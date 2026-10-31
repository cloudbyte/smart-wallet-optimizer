import json
import sys

from src.llm import (
    COLAB_MODEL_ID,
    LOCAL_MODEL_ID,
    Qwen3AgentModel,
    default_device,
    default_model_id,
)
from src.models import MerchantResolution


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
