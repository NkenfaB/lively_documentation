from src.config.paths import DATASET_DIRS, PROJECT_ROOT, ensure_project_dirs


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_dataset_dirs_created():
    ensure_project_dirs()
    assert set(DATASET_DIRS) == {"coswara", "coughvid", "tb"}
    for path in DATASET_DIRS.values():
        assert path.exists()
