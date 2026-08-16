"""Tests for local `.env` loading (python-dotenv)."""

import inspect
import os

from dotenv import load_dotenv


def test_provider_module_loads_dotenv_at_import_time():
    """The provider module loads local .env configuration at import time."""
    from app.classification import provider

    source = inspect.getsource(provider)
    assert "from dotenv import load_dotenv" in source
    assert "load_dotenv()" in source


def test_load_dotenv_populates_os_environ_from_temp_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-placeholder-not-a-real-key\n"
        "INBOXGUARDIAN_LLM_MODEL=gemini-test-model\n"
    )

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("INBOXGUARDIAN_LLM_MODEL", raising=False)

    try:
        loaded = load_dotenv(dotenv_path=env_file)
        assert loaded is True
        assert os.environ["GEMINI_API_KEY"] == "test-placeholder-not-a-real-key"
        assert os.environ["INBOXGUARDIAN_LLM_MODEL"] == "gemini-test-model"
    finally:
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("INBOXGUARDIAN_LLM_MODEL", None)


def test_load_dotenv_does_not_override_existing_environment_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=value-from-dotenv-should-be-ignored\n")

    monkeypatch.setenv("GEMINI_API_KEY", "value-already-in-real-environment")

    try:
        load_dotenv(dotenv_path=env_file)
        assert os.environ["GEMINI_API_KEY"] == "value-already-in-real-environment"
    finally:
        os.environ.pop("GEMINI_API_KEY", None)


def test_load_dotenv_is_a_safe_noop_when_no_env_file_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_VAR_THAT_SHOULD_NOT_EXIST", raising=False)
    missing_path = tmp_path / ".env"

    loaded = load_dotenv(dotenv_path=missing_path)

    assert loaded is False
    assert "SOME_VAR_THAT_SHOULD_NOT_EXIST" not in os.environ


def test_env_example_documents_the_same_variable_names_as_the_provider():
    from app.classification.provider import GEMINI_API_KEY_ENV_VAR, MODEL_ENV_VAR

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_example_path = os.path.join(repo_root, ".env.example")

    with open(env_example_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert GEMINI_API_KEY_ENV_VAR in content
    assert MODEL_ENV_VAR in content
    assert "AIza" not in content
    assert "sk-ant-" not in content


def test_gitignore_excludes_the_real_env_file():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gitignore_path = os.path.join(repo_root, ".gitignore")

    with open(gitignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert ".env" in content.splitlines()
