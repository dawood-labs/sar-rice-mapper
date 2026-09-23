

def test_repo_root_is_found_from_a_subdirectory(tmp_path, monkeypatch):
    """Library code must not depend on the working directory; a notebook has its own."""
    from sar_pipeline import config as config_mod

    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    deep = tmp_path / "notebooks" / "sub"
    deep.mkdir(parents=True)
    assert config_mod.repo_root(deep) == tmp_path
    monkeypatch.chdir(deep)
    assert config_mod.repo_root(deep) == tmp_path


def test_repo_root_falls_back_to_the_working_directory(tmp_path, monkeypatch):
    from sar_pipeline import config as config_mod

    monkeypatch.chdir(tmp_path)
    assert config_mod.repo_root(tmp_path) == tmp_path
