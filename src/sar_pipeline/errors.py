"""Pipeline exceptions.

Each exception type means "stop and let a human look", with a clear reason. They are never
swallowed silently: the CLI prints them, notebooks show them.
"""


class PipelineError(Exception):
    """Base class for all expected pipeline failures."""


class ConfirmationRequired(PipelineError):
    """An Earth Engine export or a bulk download was requested without confirmed=True."""


class DecisionRequired(PipelineError):
    """QA found issues (missing dates, low valid pixels, failed chunks) that the user must decide on.

    The details are written to runs/<run>/decisions_required.md before this is raised.
    """


class GridMismatch(PipelineError):
    """An existing grid_def.json differs from the grid recomputed from the current config/AOI."""


class AuditRequired(PipelineError):
    """A stage needs the audit (and a track selection in s1.tracks) but it is missing."""
