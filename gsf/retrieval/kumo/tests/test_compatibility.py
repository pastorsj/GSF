from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from gsf.retrieval.kumo import compatibility
from gsf.retrieval.kumo.compatibility import LegacyKumoRelationalAdapter
from gsf.retrieval.kumo.compatibility import _RelationalModelClient
from gsf.retrieval.kumo.compatibility import compatibility_registry
from nvidia_sdfm.errors import SdfmError
from nvidia_sdfm.requests import KumoRFMRequest


def _request() -> KumoRFMRequest:
    return KumoRFMRequest(
        graph=object(),
        query="PREDICT 1 FOR EACH entity.id",
        num_retries=0,
    )


def _transport() -> SimpleNamespace:
    return SimpleNamespace(
        url="https://provider.invalid",
        api_key="test-only",
        verify_ssl=True,
        timeout=1,
        max_retries=0,
    )


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, raw_client: MagicMock) -> None:
    import kumorfm.client.client

    monkeypatch.setattr(kumorfm.client.client, "KumoClient", MagicMock(return_value=raw_client))


def test_compatibility_client_rewrites_only_the_legacy_top_level_model() -> None:
    client = MagicMock()
    wrapped = _RelationalModelClient(client)
    payload = {"model": "kumo-rfm", "schema": {"model": "business-model"}}

    wrapped._request("prediction-endpoint", json=payload, timeout=3)

    client._request.assert_called_once_with(
        "prediction-endpoint",
        json={"model": "kumo-relational", "schema": {"model": "business-model"}},
        timeout=3,
    )
    assert payload["model"] == "kumo-rfm"


def test_compatibility_client_fails_closed_for_an_unknown_wire_model() -> None:
    wrapped = _RelationalModelClient(MagicMock())

    with pytest.raises(SdfmError) as raised:
        wrapped._request("prediction-endpoint", json={"model": "future-model"})

    assert raised.value.code == "INCOMPATIBLE_MODEL"


def test_compatibility_registry_preserves_normal_tabicl_and_kumo_dispatch() -> None:
    registry = compatibility_registry()

    assert registry.names() == ["kumo-rfm", "tabicl"]
    assert registry.get("kumo-rfm").capabilities().model == "kumo-rfm"


def test_compatibility_adapter_closes_client_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_client = MagicMock()
    _install_fake_client(monkeypatch, raw_client)
    result = object()
    model = MagicMock()
    model.predict.return_value = result
    engine = SimpleNamespace(KumoRFM=MagicMock(return_value=model))
    monkeypatch.setattr(compatibility, "_load_engine", lambda: engine)
    monkeypatch.setattr(compatibility, "_coerce_result", lambda value, _explain: value)

    assert LegacyKumoRelationalAdapter().predict(_transport(), _request()) is result

    raw_client.close.assert_called_once_with()


def test_compatibility_adapter_closes_client_after_model_constructor_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_client = MagicMock()
    _install_fake_client(monkeypatch, raw_client)
    engine = SimpleNamespace(KumoRFM=MagicMock(side_effect=ValueError("invalid graph")))
    monkeypatch.setattr(compatibility, "_load_engine", lambda: engine)

    with pytest.raises(ValueError, match="invalid graph"):
        LegacyKumoRelationalAdapter().predict(_transport(), _request())

    raw_client.close.assert_called_once_with()


def test_compatibility_adapter_closes_client_after_prediction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_client = MagicMock()
    _install_fake_client(monkeypatch, raw_client)
    model = MagicMock()
    model.predict.side_effect = RuntimeError("provider failed")
    engine = SimpleNamespace(KumoRFM=MagicMock(return_value=model))
    monkeypatch.setattr(compatibility, "_load_engine", lambda: engine)

    with pytest.raises(SdfmError, match="provider failed"):
        LegacyKumoRelationalAdapter().predict(_transport(), _request())

    raw_client.close.assert_called_once_with()
