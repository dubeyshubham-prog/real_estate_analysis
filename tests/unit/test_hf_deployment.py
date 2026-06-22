from scripts.check_hf_deployment import find_missing_deployment_files


def test_required_hugging_face_files_are_available() -> None:
    assert find_missing_deployment_files() == []


def test_hugging_face_docker_configuration_is_free_tier_friendly() -> None:
    dockerfile = open("Dockerfile", encoding="utf-8").read()
    dockerignore = open(".dockerignore", encoding="utf-8").read()

    assert "PORT=7860" in dockerfile
    assert "RUNTIME_DIR=/tmp/proplens" in dockerfile
    assert "ENABLE_VISION=false" in dockerfile
    assert "requirements-core.txt" in dockerfile
    assert "requirements.txt" not in dockerfile
    assert "artifacts/models/vision" in dockerignore
