"""Cross-process file locks.

Why this exists
---------------
A thread lock only protects one Python process. In practice two processes often touch the same run:
a `monitor --yes` in a terminal while a notebook downloads finished chunks, or a second notebook
opened by mistake. Without a shared lock both can read the manifest, change different rows and
write it back, and one update is silently lost (for example a task id), which later causes a
duplicate Earth Engine export.

How it works
------------
The lock is a small file created with O_CREAT | O_EXCL: the operating system guarantees that only
one process can create it. That works on local disks, WSL's /mnt/c and network file systems such
as EFS/NFS, where `fcntl` locks are unreliable. The file records who holds it.

When is a lock left behind by a crash broken?
- Holder in the SAME process space as us (same host name, kernel boot and PID namespace): the lock
  is broken only if that process no longer exists (or its pid was reused by a newer process).
  A live holder is NEVER broken, however old the file is.
- Holder elsewhere (another container, machine or PID namespace, e.g. a second JupyterHub server
  on shared EFS): we cannot see its processes, so the lock is broken only when the holder stopped
  refreshing it for `stale_after` seconds. Long-held locks refresh themselves with a heartbeat thread.

Breaking a stale lock is itself serialised by a second O_EXCL file (`<lock>.break`) and re-checks
that the file still belongs to the same stale holder, so two processes can never both "win".

Locks are re-entrant inside one process (threads of the same process share one lock file).
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path

from .errors import PipelineError

_PROCESS_STATE: dict[str, list] = {}  # path -> [threading.RLock, depth, token, heartbeat_stop_event]
_STATE_GUARD = threading.Lock()
_BREAK_FILE_STALE_SECONDS = 30.0


class LockTimeout(PipelineError):
    """The lock is held by another live process for longer than the timeout."""


# ---------------------------------------------------------------- process identity
def _read_text(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def _space_id() -> str:
    """Identifies the process space in which pids are meaningful: host + kernel boot + PID namespace."""
    try:
        pid_ns = str(os.stat("/proc/self/ns/pid").st_ino)
    except OSError:
        pid_ns = ""
    return f"{socket.gethostname()}|{_read_text('/proc/sys/kernel/random/boot_id')}|{pid_ns}"


def _process_start(pid: int) -> str:
    """Start time of a process (distinguishes a reused pid); '' when unknown."""
    try:
        import psutil

        return f"{psutil.Process(pid).create_time():.3f}"
    except Exception:  # noqa: BLE001 - psutil missing, process gone, access denied
        stat = _read_text(f"/proc/{pid}/stat")
        if stat and ")" in stat:
            fields = stat.rsplit(")", 1)[1].split()
            return fields[19] if len(fields) > 19 else ""
        return ""


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _read_holder(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text() or "{}")
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- lock
class FileLock:
    def __init__(self, path: str | Path, timeout: float | None = 600.0, stale_after: float = 300.0,
                 poll: float = 0.05, heartbeat: bool = False):
        self.path = Path(path)
        self.timeout = timeout
        self.stale_after = stale_after
        self.poll = poll
        self.heartbeat = heartbeat
        key = str(self.path.resolve())
        with _STATE_GUARD:
            if key not in _PROCESS_STATE:
                _PROCESS_STATE[key] = [threading.RLock(), 0, None, None]
            self._state = _PROCESS_STATE[key]

    # ------------------------------------------------------------ public
    def acquire(self) -> "FileLock":
        rlock = self._state[0]
        wait = -1 if self.timeout is None or self.timeout < 0 else self.timeout
        if not rlock.acquire(timeout=wait):
            raise LockTimeout(f"timed out waiting for {self.path} (held by another thread)")
        try:
            if self._state[1] == 0:
                self._state[2] = self._acquire_file()
                if self.heartbeat:
                    self._start_heartbeat()
            self._state[1] += 1
        except BaseException:
            rlock.release()
            raise
        return self

    def release(self) -> None:
        self._state[1] -= 1
        if self._state[1] == 0:
            stop = self._state[3]
            if stop is not None:
                stop.set()
                self._state[3] = None
            holder = _read_holder(self.path)
            if holder and holder.get("token") == self._state[2]:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
            self._state[2] = None
        self._state[0].release()

    def touch(self) -> None:
        """Mark a long-held lock as alive (the heartbeat thread does this automatically)."""
        try:
            os.utime(self.path, None)
        except OSError:
            pass

    def holder(self) -> dict | None:
        return _read_holder(self.path)

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()

    # ------------------------------------------------------------ internals
    def _start_heartbeat(self) -> None:
        stop = threading.Event()
        interval = max(1.0, (self.stale_after or 60.0) / 4)

        def beat():
            while not stop.wait(interval):
                self.touch()

        threading.Thread(target=beat, name=f"lock-heartbeat:{self.path.name}", daemon=True).start()
        self._state[3] = stop

    def _stale_token(self) -> str | None:
        """Token of the current holder if its lock is stale, else None."""
        holder = _read_holder(self.path)
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            return None
        if not holder:
            # Unreadable/empty: a writer may be between create and write. Only very old files are stale.
            return "" if self.stale_after is not None and age > self.stale_after else None
        token = str(holder.get("token", ""))
        if holder.get("space") == _space_id() and isinstance(holder.get("pid"), int):
            pid = holder["pid"]
            if not _pid_alive(pid):
                return token
            started = holder.get("pid_start") or ""
            if started and _process_start(pid) and _process_start(pid) != started:
                return token  # pid reused by another process
            return None  # live holder in our process space: never stale
        return token if self.stale_after is not None and age > self.stale_after else None

    def _break_stale(self, observed_token: str) -> None:
        breaker = self.path.with_name(self.path.name + ".break")
        try:
            fd = os.open(breaker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                if time.time() - breaker.stat().st_mtime > _BREAK_FILE_STALE_SECONDS:
                    breaker.unlink(missing_ok=True)  # a breaker crashed mid-break
            except OSError:
                pass
            return
        os.close(fd)
        try:
            # Re-check under the break file: still the same stale holder?
            if self._stale_token() == observed_token:
                holder = _read_holder(self.path)
                if (holder or {}).get("token", "") == observed_token or not holder:
                    self.path.unlink(missing_ok=True)
        finally:
            breaker.unlink(missing_ok=True)

    def _acquire_file(self) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        payload = json.dumps({
            "pid": os.getpid(), "pid_start": _process_start(os.getpid()), "space": _space_id(),
            "host": socket.gethostname(), "token": token,
            "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }).encode()
        deadline = None if self.timeout is None or self.timeout < 0 else time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                stale = self._stale_token()
                if stale is not None:
                    self._break_stale(stale)
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    raise LockTimeout(f"{self.path} is held by {self.holder()}; timed out after {self.timeout:g} s")
                time.sleep(self.poll)
                continue
            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            holder = _read_holder(self.path)
            if holder and holder.get("token") == token:
                return token


def run_lock(run_dir: str | Path, name: str, wait: bool = False) -> FileLock:
    """Exclusive, self-refreshing lock for a long-running stage of one run (one `monitor`, one `download`).

    With ``wait=False`` a second process fails immediately with a message naming the holder.
    """
    lock = FileLock(Path(run_dir) / f".{name}.lock", timeout=(None if wait else 0.0), stale_after=900.0,
                    heartbeat=True)
    try:
        return lock.acquire()
    except LockTimeout as exc:
        raise PipelineError(
            f"Another '{name}' process is already working on this run ({lock.holder()}). "
            "Wait for it to finish or stop it. A crashed holder on this machine is detected immediately; "
            "on another machine its lock expires 15 minutes after its last heartbeat."
        ) from exc
