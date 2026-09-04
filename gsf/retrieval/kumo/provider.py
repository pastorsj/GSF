# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-fast compatibility checks for the configured Kumo Universal TFM NIM.

The Kumo Python driver normally owns the wire-level model identifier. This
module compares that identifier with the NIM's advertised models and returns a
small, credential-free readiness receipt. One reviewed transition is supported
through a client-local GSF adapter: exactly ``nvidia-sdfm==0.2.1`` with
``kumorfm==2.28.0`` may translate ``kumo-rfm`` to ``kumo-relational``. Unknown
version/model combinations remain incompatible and fail closed.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from gsf.retrieval.kumo.compatibility_contract import LEGACY_ADAPTER_ID
from gsf.retrieval.kumo.compatibility_contract import LEGACY_KUMORFM_VERSION
from gsf.retrieval.kumo.compatibility_contract import LEGACY_MODEL
from gsf.retrieval.kumo.compatibility_contract import LEGACY_SDFM_VERSION
from gsf.retrieval.kumo.compatibility_contract import RELATIONAL_MODEL

_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_MAX_MODELS = 16
_MAX_MODEL_CHARS = 128


@dataclass(frozen=True)
class KumoProviderReadiness:
    """Display-safe proof that the installed SDK and configured NIM agree."""

    status: str
    ready: bool
    expected_model: str
    advertised_models: tuple[str, ...]
    nvidia_sdfm_version: str
    kumorfm_version: str
    wire_model: str | None = None
    compatibility_adapter: str | None = None
    error_code: str | None = None
    retryable: bool = False

    def to_receipt(self) -> dict[str, Any]:
        """Return an allow-listed receipt with no endpoint or credential data."""

        return {
            "schema_version": 1,
            "status": self.status,
            "ready": self.ready,
            "sdk": {
                "nvidia_sdfm": self.nvidia_sdfm_version,
                "kumorfm": self.kumorfm_version,
                "expected_model": self.expected_model,
            },
            "provider": {"advertised_models": list(self.advertised_models)},
            "compatibility": (
                {
                    "adapter": self.compatibility_adapter,
                    "wire_model": self.wire_model,
                }
                if self.compatibility_adapter is not None
                else None
            ),
            "error": ({"code": self.error_code, "retryable": self.retryable} if self.error_code is not None else None),
        }


class KumoProviderCompatibilityError(RuntimeError):
    """A deterministic SDK/NIM contract mismatch that retries cannot repair."""

    code = "KUMO_PROVIDER_MODEL_INCOMPATIBLE"
    retryable = False

    def __init__(self, readiness: KumoProviderReadiness) -> None:
        self.readiness = readiness
        super().__init__(
            "The installed Kumo SDK model contract does not match the model "
            "advertised by the configured NIM. Install a compatible official SDK."
        )


class KumoProviderUnavailableError(RuntimeError):
    """A safe provider readiness failure, with retryability made explicit."""

    def __init__(self, readiness: KumoProviderReadiness) -> None:
        self.readiness = readiness
        self.code = readiness.error_code or "KUMO_PROVIDER_UNAVAILABLE"
        self.retryable = readiness.retryable
        super().__init__("The configured Kumo provider did not pass readiness checks.")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def installed_kumo_model() -> str:
    """Return the exact model identifier serialized by the installed driver."""

    try:
        from kumorfm.client.generated.tfm_api import TFM_MODEL_KUMO_RFM
    except Exception:
        return "unavailable"
    return str(TFM_MODEL_KUMO_RFM)


def _base_url(url: str, *, has_api_key: bool) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("KUMO_RFM_API_URL must be an HTTP(S) origin without credentials, query, or fragment.")
    if has_api_key and parsed.scheme != "https" and parsed.hostname.casefold() not in _LOCAL_HOSTS:
        raise ValueError("KUMO_RFM_API_URL must use HTTPS when an API key is configured.")
    return url.rstrip("/")


def _models(document: Any) -> tuple[str, ...] | None:
    if not isinstance(document, dict) or not isinstance(document.get("data"), list):
        return None
    values: list[str] = []
    for item in document["data"]:
        if not isinstance(item, dict):
            continue
        model = item.get("id")
        if not isinstance(model, str) or not model or len(model) > _MAX_MODEL_CHARS:
            continue
        if model not in values:
            values.append(model)
        if len(values) == _MAX_MODELS:
            break
    return tuple(values)


def _health_ready(document: Any) -> bool:
    """Match the Universal TFM typed readiness document used by the SDK."""

    if not isinstance(document, dict):
        return False
    status = document.get("status")
    check = document.get("check")
    normalized_status = status.casefold() if isinstance(status, str) else ""
    normalized_check = check.casefold() if isinstance(check, str) else ""
    return normalized_status == "ready" or (normalized_status == "healthy" and normalized_check == "ready")


def check_kumo_provider(
    url: str,
    api_key: str | None,
    *,
    timeout: float = 10.0,
    transport: httpx.BaseTransport | None = None,
) -> KumoProviderReadiness:
    """Probe health and model compatibility without emitting sensitive data."""

    expected_model = installed_kumo_model()
    versions = {
        "nvidia_sdfm_version": _package_version("nvidia-sdfm"),
        "kumorfm_version": _package_version("kumorfm"),
    }
    try:
        base = _base_url(url, has_api_key=bool(api_key))
    except ValueError:
        return KumoProviderReadiness(
            status="unavailable",
            ready=False,
            expected_model=expected_model,
            advertised_models=(),
            error_code="KUMO_PROVIDER_INVALID_CONFIGURATION",
            retryable=False,
            **versions,
        )

    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        with httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,
            transport=transport,
        ) as client:
            health = client.get(f"{base}/v1/health/ready")
            if health.status_code in {401, 403}:
                code, retryable = "KUMO_PROVIDER_AUTHENTICATION_FAILED", False
            elif health.status_code != 200:
                code, retryable = "KUMO_PROVIDER_NOT_READY", health.status_code >= 500 or health.status_code == 429
            elif not _health_ready(health.json()):
                code, retryable = "KUMO_PROVIDER_NOT_READY", True
            else:
                code, retryable = "", False
            if code:
                return KumoProviderReadiness(
                    status="unavailable",
                    ready=False,
                    expected_model=expected_model,
                    advertised_models=(),
                    error_code=code,
                    retryable=retryable,
                    **versions,
                )

            response = client.get(f"{base}/v1/models")
            if response.status_code in {401, 403}:
                code, retryable = "KUMO_PROVIDER_AUTHENTICATION_FAILED", False
            elif response.status_code != 200:
                code, retryable = (
                    "KUMO_PROVIDER_MODELS_UNAVAILABLE",
                    response.status_code >= 500 or response.status_code == 429,
                )
            else:
                code, retryable = "", False
            if code:
                return KumoProviderReadiness(
                    status="unavailable",
                    ready=False,
                    expected_model=expected_model,
                    advertised_models=(),
                    error_code=code,
                    retryable=retryable,
                    **versions,
                )
            advertised = _models(response.json())
    except (httpx.HTTPError, ValueError):
        return KumoProviderReadiness(
            status="unavailable",
            ready=False,
            expected_model=expected_model,
            advertised_models=(),
            error_code="KUMO_PROVIDER_PROBE_FAILED",
            retryable=True,
            **versions,
        )

    if advertised is None or not advertised:
        return KumoProviderReadiness(
            status="unavailable",
            ready=False,
            expected_model=expected_model,
            advertised_models=(),
            error_code="KUMO_PROVIDER_INVALID_MODELS_RESPONSE",
            retryable=False,
            **versions,
        )
    if expected_model not in advertised:
        if (
            expected_model == LEGACY_MODEL
            and versions["nvidia_sdfm_version"] == LEGACY_SDFM_VERSION
            and versions["kumorfm_version"] == LEGACY_KUMORFM_VERSION
            and RELATIONAL_MODEL in advertised
        ):
            return KumoProviderReadiness(
                status="ready",
                ready=True,
                expected_model=expected_model,
                wire_model=RELATIONAL_MODEL,
                compatibility_adapter=LEGACY_ADAPTER_ID,
                advertised_models=advertised,
                **versions,
            )
        return KumoProviderReadiness(
            status="incompatible",
            ready=False,
            expected_model=expected_model,
            advertised_models=advertised,
            error_code=KumoProviderCompatibilityError.code,
            retryable=False,
            **versions,
        )
    return KumoProviderReadiness(
        status="ready",
        ready=True,
        expected_model=expected_model,
        wire_model=expected_model,
        advertised_models=advertised,
        **versions,
    )


def require_kumo_provider_ready(
    url: str,
    api_key: str | None,
    *,
    timeout: float = 10.0,
) -> KumoProviderReadiness:
    """Return readiness or raise a typed, non-secret-bearing provider error."""

    readiness = check_kumo_provider(url, api_key, timeout=timeout)
    if readiness.ready:
        return readiness
    if readiness.error_code == KumoProviderCompatibilityError.code:
        raise KumoProviderCompatibilityError(readiness)
    raise KumoProviderUnavailableError(readiness)


def is_nonrepairable_provider_error(exc: BaseException) -> bool:
    """Identify structured provider-contract failures that PQL repair cannot fix.

    This deliberately inspects exception type/status/code metadata only.  It
    never parses model-authored text or provider error prose to steer control
    flow.
    """

    if isinstance(exc, KumoProviderCompatibilityError):
        return True
    if isinstance(exc, KumoProviderUnavailableError):
        return not exc.retryable
    status = getattr(exc, "status_code", None)
    if status in {401, 403, 404, 405, 415}:
        return True
    code = getattr(exc, "code", None)
    if not isinstance(code, str):
        return False
    return code.upper() in {
        "INCOMPATIBLE_MODEL",
        "INVALID_MODEL",
        "MISSING_EXTRA",
        "MODEL_NOT_FOUND",
        "UNKNOWN_MODEL",
        "UNSUPPORTED_MODEL",
    }


__all__ = [
    "KumoProviderCompatibilityError",
    "KumoProviderReadiness",
    "KumoProviderUnavailableError",
    "check_kumo_provider",
    "installed_kumo_model",
    "is_nonrepairable_provider_error",
    "require_kumo_provider_ready",
]
