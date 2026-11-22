import json
import sys

from src.llm import (
    ANTHROPIC_API_VERSION,
    ANTHROPIC_MESSAGES_URL,
    CLAUDE_MODEL_IDS,
    COLAB_MODEL_ID,
    LOCAL_MODEL_ID,
    AnthropicClaudeModel,
    Qwen3AgentModel,
    default_device,
    default_model_id,
)
from src.models import MerchantResolution


class FakeAnthropicResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        return self.payload


def test_claude_model_sends_key_in_header_and_returns_text(monkeypatch):
    observed = {}

    def fake_post(url, **kwargs):
        observed["url"] = url
        observed["kwargs"] = kwargs
        return FakeAnthropicResponse(
            200,
            {
                "content": [
                    {"type": "text", "text": "A concise"},
                    {"type": "text", "text": " recommendation."},
                ]
            },
        )

    monkeypatch.setattr("src.llm.requests.post", fake_post)
    model = AnthropicClaudeModel(
        model_id=CLAUDE_MODEL_IDS["Claude Sonnet 4.5"],
        api_key="sk-ant-test",
    )

    result = model.explain("{\"merchant\":\"Trader Joe's\"}")

    assert result == "A concise recommendation."
    assert observed["url"] == ANTHROPIC_MESSAGES_URL
    assert observed["kwargs"]["headers"]["x-api-key"] == "sk-ant-test"
    assert observed["kwargs"]["headers"]["anthropic-version"] == ANTHROPIC_API_VERSION
    assert observed["kwargs"]["json"]["model"] == "claude-sonnet-4-5"
    assert "sk-ant-test" not in str(observed["kwargs"]["json"])


def test_claude_model_reports_http_status_without_exposing_api_key(monkeypatch):
    monkeypatch.setattr(
        "src.llm.requests.post",
        lambda *_args, **_kwargs: FakeAnthropicResponse(401, {"error": "unauthorized"}),
    )
    model = AnthropicClaudeModel(
        model_id=CLAUDE_MODEL_IDS["Claude Haiku 4.5"],
        api_key="sk-ant-private",
    )

    try:
        model.explain("{}")
    except RuntimeError as error:
        assert "HTTP 401" in str(error)
        assert "sk-ant-private" not in str(error)
    else:
        raise AssertionError("Unauthorized API response should fail explicitly.")


def test_claude_model_rejects_malformed_response(monkeypatch):
    monkeypatch.setattr(
        "src.llm.requests.post",
        lambda *_args, **_kwargs: FakeAnthropicResponse(200, {"content": "not a list"}),
    )
    model = AnthropicClaudeModel(
        model_id=CLAUDE_MODEL_IDS["Claude Haiku 4.5"],
        api_key="sk-ant-test",
    )

    try:
        model.explain("{}")
    except RuntimeError as error:
        assert "did not contain text" in str(error)
    else:
        raise AssertionError("Malformed API response should fail explicitly.")


def test_claude_merchant_resolution_rejects_names_outside_catalog(monkeypatch):
    model = AnthropicClaudeModel(
        model_id=CLAUDE_MODEL_IDS["Claude Sonnet 4.5"],
        api_key="sk-ant-test",
    )
    monkeypatch.setattr(
        model,
        "_generate",
        lambda *_args, **_kwargs: (
            '{"canonical_name":"Unlisted Merchant","confidence":1.0}'
        ),
    )

    assert model.resolve_merchant("Unlisted Merchant", ["Trader Joe's"]) is None


def test_claude_model_requires_supported_model_and_non_empty_api_key():
    try:
        AnthropicClaudeModel(model_id="claude-unknown", api_key="key")
    except ValueError as error:
        assert "Unsupported Claude model" in str(error)
    else:
        raise AssertionError("Unsupported model should be rejected.")

    try:
        AnthropicClaudeModel(
            model_id=CLAUDE_MODEL_IDS["Claude Sonnet 4.5"],
            api_key=" ",
        )
    except ValueError as error:
        assert "API key is required" in str(error)
    else:
        raise AssertionError("Empty API key should be rejected.")


def test_local_defaults_select_small_cpu_model(monkeypatch):
    monkeypatch.delenv("WALLET_AGENT_MODEL", raising=False)
    monkeypatch.delenv("WALLET_AGENT_DEVICE", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)

    assert default_model_id() == LOCAL_MODEL_ID
    assert default_device() == "cpu"
    model = Qwen3AgentModel()
    assert model.model_id == LOCAL_MODEL_ID
    assert model.device == "cpu"


def test_colab_defaults_select_larger_model_and_auto_device(monkeypatch):
    monkeypatch.delenv("WALLET_AGENT_MODEL", raising=False)
    monkeypatch.delenv("WALLET_AGENT_DEVICE", raising=False)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release")

    assert default_model_id() == COLAB_MODEL_ID
    assert default_device() == "auto"


def test_model_and_device_can_be_overridden(monkeypatch):
    monkeypatch.setenv("WALLET_AGENT_MODEL", "org/custom-model")
    monkeypatch.setenv("WALLET_AGENT_DEVICE", "cpu")

    assert default_model_id() == "org/custom-model"
    assert default_device() == "cpu"
    model = Qwen3AgentModel()
    assert model.model_id == "org/custom-model"
    assert model.device == "cpu"


def test_explicit_constructor_values_override_environment(monkeypatch):
    monkeypatch.setenv("WALLET_AGENT_MODEL", "org/environment-model")
    monkeypatch.setenv("WALLET_AGENT_DEVICE", "auto")

    model = Qwen3AgentModel(model_id="org/explicit-model", device="cpu")

    assert model.model_id == "org/explicit-model"
    assert model.device == "cpu"


def test_four_bit_quantization_can_be_enabled_from_environment(monkeypatch):
    monkeypatch.setenv("WALLET_AGENT_4BIT", "true")

    assert Qwen3AgentModel().load_in_4bit
    assert not Qwen3AgentModel(load_in_4bit=False).load_in_4bit


def test_merchant_resolver_parses_only_catalog_candidate(monkeypatch):
    model = Qwen3AgentModel(model_id="test-model")
    monkeypatch.setattr(
        model,
        "_generate",
        lambda *_args, **_kwargs: json.dumps(
            {"canonical_name": "Trader Joe's", "confidence": 0.91}
        ),
    )

    match = model.resolve_merchant("TJ's", ["Trader Joe's", "Walmart"])

    assert match == MerchantResolution(
        canonical_name="Trader Joe's",
        confidence=0.91,
    )


def test_merchant_resolver_rejects_model_invented_catalog_candidate(monkeypatch):
    model = Qwen3AgentModel(model_id="test-model")
    monkeypatch.setattr(
        model,
        "_generate",
        lambda *_args, **_kwargs: json.dumps(
            {"canonical_name": "Invented Merchant", "confidence": 0.99}
        ),
    )

    assert model.resolve_merchant("Some shop", ["Trader Joe's"]) is None


def test_merchant_resolver_returns_none_for_invalid_model_json(monkeypatch):
    model = Qwen3AgentModel(model_id="test-model")
    monkeypatch.setattr(model, "_generate", lambda *_args, **_kwargs: "not JSON")

    assert model.resolve_merchant("Some shop", ["Trader Joe's"]) is None
