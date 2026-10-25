from app.settings import PROJECT_ROOT, configured_path


def test_configured_path_uses_project_root_for_default(monkeypatch):
    monkeypatch.delenv("VINBIGDATA_TEST_PATH", raising=False)

    assert configured_path(
        "VINBIGDATA_TEST_PATH",
        "checkpoints/model.pth",
    ) == PROJECT_ROOT / "checkpoints" / "model.pth"


def test_configured_path_resolves_relative_environment_value(monkeypatch):
    monkeypatch.setenv("VINBIGDATA_TEST_PATH", "custom/model.pth")

    assert configured_path(
        "VINBIGDATA_TEST_PATH",
        "checkpoints/model.pth",
    ) == PROJECT_ROOT / "custom" / "model.pth"


def test_configured_path_preserves_absolute_environment_value(monkeypatch, tmp_path):
    configured = tmp_path / "model.pth"
    monkeypatch.setenv("VINBIGDATA_TEST_PATH", str(configured))

    assert configured_path(
        "VINBIGDATA_TEST_PATH",
        "checkpoints/model.pth",
    ) == configured


def test_posthoc_checkpoint_defaults_to_project_convnext(monkeypatch):
    monkeypatch.delenv("VINBIGDATA_CHECKPOINT", raising=False)

    from src.xai.production.request_visual_xai import _checkpoint_path

    assert _checkpoint_path() == (
        PROJECT_ROOT
        / "checkpoints"
        / "convnext_tiny"
        / "convnext_tiny_weak_classes_best.pth"
    )
