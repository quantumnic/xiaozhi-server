"""Tests for configuration."""

import os
from unittest.mock import patch


class TestConfig:
    def test_defaults(self):
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            # Re-import to pick up env
            import importlib
            import config
            importlib.reload(config)

            assert config.LLM_PROVIDER == "openai"
            assert config.SERVER_PORT == 8765
            assert config.SERVER_HOST == "0.0.0.0"
            assert config.TTS_VOICE == "alloy"
            assert config.TTS_MODEL == "tts-1"
            assert config.WHISPER_MODEL == "whisper-1"
            assert config.MAX_HISTORY == 10
            assert config.AUTH_TOKEN == ""

    def test_custom_values(self):
        env = {
            "OPENAI_API_KEY": "sk-test",
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "LLM_MODEL": "claude-opus-4-6",
            "SERVER_PORT": "9000",
            "TTS_VOICE": "nova",
            "AUTH_TOKEN": "secret123",
            "MAX_HISTORY": "20",
        }
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import config
            importlib.reload(config)

            assert config.LLM_PROVIDER == "anthropic"
            assert config.ANTHROPIC_API_KEY == "sk-ant-test"
            assert config.SERVER_PORT == 9000
            assert config.TTS_VOICE == "nova"
            assert config.AUTH_TOKEN == "secret123"
            assert config.MAX_HISTORY == 20

    def test_get_llm_model_default_openai(self):
        env = {"OPENAI_API_KEY": "sk-test", "LLM_PROVIDER": "openai"}
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import config
            importlib.reload(config)
            assert config.get_llm_model() == "gpt-4o"

    def test_get_llm_model_default_anthropic(self):
        env = {
            "OPENAI_API_KEY": "sk-test",
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import config
            importlib.reload(config)
            assert config.get_llm_model() == "claude-sonnet-4-20250514"

    def test_get_llm_model_explicit(self):
        env = {"OPENAI_API_KEY": "sk-test", "LLM_MODEL": "gpt-4o-mini"}
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import config
            importlib.reload(config)
            assert config.get_llm_model() == "gpt-4o-mini"

    def test_validate_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import config
            importlib.reload(config)
            errors = config.validate()
            assert len(errors) > 0
            assert any("OPENAI_API_KEY" in e for e in errors)

    def test_validate_openai_ok(self):
        env = {"OPENAI_API_KEY": "sk-test", "LLM_PROVIDER": "openai"}
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import config
            importlib.reload(config)
            errors = config.validate()
            assert errors == []

    def test_validate_anthropic_missing_key(self):
        env = {"OPENAI_API_KEY": "sk-test", "LLM_PROVIDER": "anthropic"}
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import config
            importlib.reload(config)
            errors = config.validate()
            assert any("ANTHROPIC_API_KEY" in e for e in errors)

    def test_validate_invalid_provider(self):
        env = {"OPENAI_API_KEY": "sk-test", "LLM_PROVIDER": "gemini"}
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import config
            importlib.reload(config)
            errors = config.validate()
            assert any("LLM_PROVIDER" in e for e in errors)


class TestSystemPrompt:
    def test_default_prompt(self):
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import config
            importlib.reload(config)
            assert "hilfreicher" in config.SYSTEM_PROMPT

    def test_custom_prompt(self):
        env = {"OPENAI_API_KEY": "sk-test", "SYSTEM_PROMPT": "You are a pirate."}
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import config
            importlib.reload(config)
            assert config.SYSTEM_PROMPT == "You are a pirate."
