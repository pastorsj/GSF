import sys
from unittest.mock import patch

import httpx
import pytest
from gsf.retrieval.kumo.provider import KumoProviderCompatibilityError
from gsf.retrieval.kumo.provider import KumoProviderReadiness
from gsf.retrieval.kumo.provider import KumoProviderUnavailableError
from gsf.retrieval.kumo.provider import check_kumo_provider
from gsf.retrieval.kumo.provider import is_nonrepairable_provider_error
from gsf.retrieval.kumo.provider import require_kumo_provider_ready


def _transport(
    model: str = "kumo-relational",
    *,
    health_status: int = 200,
    health_body: dict | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-API-Key") == "private-test-key"
        if request.url.path == "/v1/health/ready":
            return httpx.Response(
                health_status, json=health_body or {"status": "ready"}
            )
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": model, "object": "model"}]}
            )
        raise AssertionError(f"unexpected route: {request.url.path}")

    return httpx.MockTransport(handler)


def test_provider_readiness_proves_installed_model_is_advertised() -> None:
    with patch(
        "gsf.retrieval.kumo.provider.installed_kumo_model",
        return_value="kumo-relational",
    ):
        readiness = check_kumo_provider(
            "https://provider.example.test",
            "private-test-key",
            transport=_transport(),
        )

    assert readiness.ready is True
    assert readiness.status == "ready"
    assert readiness.expected_model == "kumo-relational"
    assert readiness.advertised_models == ("kumo-relational",)
    assert "private-test-key" not in str(readiness.to_receipt())
    assert "provider.example.test" not in str(readiness.to_receipt())


def test_provider_readiness_returns_typed_nonretryable_model_mismatch() -> None:
    with (
        patch(
            "gsf.retrieval.kumo.provider.installed_kumo_model", return_value="kumo-rfm"
        ),
        patch(
            "gsf.retrieval.kumo.provider._package_version",
            return_value="unreviewed-version",
        ),
    ):
        readiness = check_kumo_provider(
            "https://provider.example.test",
            "private-test-key",
            transport=_transport(),
        )

    assert readiness.ready is False
    assert readiness.status == "incompatible"
    assert readiness.error_code == "KUMO_PROVIDER_MODEL_INCOMPATIBLE"
    assert readiness.retryable is False


def test_unreviewed_version_mismatch_does_not_import_private_adapter_module() -> None:
    with (
        patch(
            "gsf.retrieval.kumo.provider.installed_kumo_model", return_value="kumo-rfm"
        ),
        patch(
            "gsf.retrieval.kumo.provider._package_version",
            return_value="future-version",
        ),
        patch.dict(sys.modules, {"gsf.retrieval.kumo.compatibility": None}),
    ):
        readiness = check_kumo_provider(
            "https://provider.example.test",
            "private-test-key",
            transport=_transport(),
        )

    assert readiness.status == "incompatible"
    assert readiness.error_code == "KUMO_PROVIDER_MODEL_INCOMPATIBLE"


def test_provider_readiness_enables_only_the_exact_reviewed_legacy_adapter() -> None:
    versions = {"nvidia-sdfm": "0.2.1", "kumorfm": "2.28.0"}
    with (
        patch(
            "gsf.retrieval.kumo.provider.installed_kumo_model", return_value="kumo-rfm"
        ),
        patch(
            "gsf.retrieval.kumo.provider._package_version",
            side_effect=versions.__getitem__,
        ),
    ):
        readiness = check_kumo_provider(
            "https://provider.example.test",
            "private-test-key",
            transport=_transport(),
        )

    assert readiness.ready is True
    assert readiness.wire_model == "kumo-relational"
    assert (
        readiness.compatibility_adapter
        == "nvidia-sdfm-0.2.1-kumorfm-2.28.0-relational-model"
    )


def test_require_provider_ready_raises_safe_typed_mismatch() -> None:
    with patch(
        "gsf.retrieval.kumo.provider._package_version",
        return_value="unreviewed-version",
    ):
        mismatch = check_kumo_provider(
            "https://provider.example.test",
            "private-test-key",
            transport=_transport(),
        )
    with (
        patch("gsf.retrieval.kumo.provider.check_kumo_provider", return_value=mismatch),
        pytest.raises(KumoProviderCompatibilityError) as raised,
    ):
        require_kumo_provider_ready("https://provider.example.test", "private-test-key")

    assert raised.value.code == "KUMO_PROVIDER_MODEL_INCOMPATIBLE"
    assert raised.value.retryable is False
    assert "private-test-key" not in str(raised.value)
    assert "provider.example.test" not in str(raised.value)


def test_provider_auth_failure_is_nonretryable_and_does_not_probe_models() -> None:
    readiness = check_kumo_provider(
        "https://provider.example.test",
        "private-test-key",
        transport=_transport(health_status=403),
    )

    assert readiness.ready is False
    assert readiness.error_code == "KUMO_PROVIDER_AUTHENTICATION_FAILED"
    assert readiness.retryable is False


def test_provider_rejects_http_200_without_typed_ready_health_state() -> None:
    readiness = check_kumo_provider(
        "https://provider.example.test",
        "private-test-key",
        transport=_transport(health_body={"ready": True}),
    )

    assert readiness.ready is False
    assert readiness.error_code == "KUMO_PROVIDER_NOT_READY"
    assert readiness.retryable is True


def test_provider_accepts_healthy_ready_health_variant() -> None:
    with patch(
        "gsf.retrieval.kumo.provider.installed_kumo_model",
        return_value="kumo-relational",
    ):
        readiness = check_kumo_provider(
            "https://provider.example.test",
            "private-test-key",
            transport=_transport(health_body={"status": "healthy", "check": "ready"}),
        )

    assert readiness.ready is True


@pytest.mark.parametrize(("retryable", "expected"), [(False, True), (True, False)])
def test_typed_provider_unavailability_controls_repairability(
    retryable: bool, expected: bool
) -> None:
    readiness = KumoProviderReadiness(
        status="unavailable",
        ready=False,
        expected_model="kumo-rfm",
        advertised_models=(),
        nvidia_sdfm_version="0.2.1",
        kumorfm_version="2.28.0",
        error_code="KUMO_PROVIDER_NOT_READY",
        retryable=retryable,
    )

    assert (
        is_nonrepairable_provider_error(KumoProviderUnavailableError(readiness))
        is expected
    )
