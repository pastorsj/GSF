# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version-gated adapter for one known Kumo SDK/NIM model-name transition.

This is an ordinary ``nvidia-sdfm`` adapter registered on one GSF-owned client,
not a process monkeypatch.  It preserves the legacy SDK's graph sampling,
payload generation, transport limits, retries, and result coercion, changing
only the top-level wire model from ``kumo-rfm`` to ``kumo-relational`` at its
HTTP boundary.  The provider readiness module enables it only for the exact
reviewed SDK versions and advertised NIM model.
"""

from __future__ import annotations

import contextlib
from typing import Any

from nvidia_sdfm.adapters import TabICLAdapter
from nvidia_sdfm.adapters.kumorfm import _build_task_table
from nvidia_sdfm.adapters.kumorfm import _coerce_result
from nvidia_sdfm.adapters.kumorfm import _load_engine
from nvidia_sdfm.adapters.kumorfm import _reject_reserved_options
from nvidia_sdfm.adapters.kumorfm import _resolve_explain
from nvidia_sdfm.adapters.kumorfm import _translate_engine_error
from nvidia_sdfm.adapters.kumorfm import _validate_batch_size
from nvidia_sdfm.adapters.kumorfm import _validate_num_retries
from nvidia_sdfm.adapters.kumorfm import _validate_task_request
from nvidia_sdfm.base import AdapterRegistry
from nvidia_sdfm.base import ModelCapabilities
from nvidia_sdfm.base import request_type_names
from nvidia_sdfm.core.serving import ServingTarget
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import SdfmError
from nvidia_sdfm.requests import KumoRFMRequest
from nvidia_sdfm.requests import KumoRFMTaskRequest

from gsf.retrieval.kumo.compatibility_contract import LEGACY_ADAPTER_ID
from gsf.retrieval.kumo.compatibility_contract import LEGACY_KUMORFM_VERSION
from gsf.retrieval.kumo.compatibility_contract import LEGACY_MODEL
from gsf.retrieval.kumo.compatibility_contract import LEGACY_SDFM_VERSION
from gsf.retrieval.kumo.compatibility_contract import RELATIONAL_MODEL


class _RelationalModelClient:
    """Delegate Kumo HTTP calls while rewriting only the reviewed model field."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def close(self) -> None:
        self._client.close()

    def _request(self, endpoint: Any, **kwargs: Any) -> Any:
        payload = kwargs.get("json")
        if isinstance(payload, dict) and "model" in payload:
            if payload["model"] != LEGACY_MODEL:
                raise SdfmError(
                    "The compatibility adapter received an unexpected SDK wire model.",
                    code="INCOMPATIBLE_MODEL",
                )
            kwargs["json"] = {**payload, "model": RELATIONAL_MODEL}
        return self._client._request(endpoint, **kwargs)


class LegacyKumoRelationalAdapter:
    """Kumo adapter for the exact legacy-client/current-NIM contract pair."""

    name = LEGACY_MODEL
    request_type = (KumoRFMRequest, KumoRFMTaskRequest)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self.name,
            request_type=request_type_names(self.request_type),
            tasks=(
                "binary_classification",
                "multiclass_classification",
                "regression",
                "forecasting",
                "temporal_link_prediction",
            ),
            outputs=("prediction", "probabilities", "explanation"),
        )

    def predict(self, transport: Transport | ServingTarget, request: Any) -> Any:
        if isinstance(transport, ServingTarget):
            raise SdfmError(
                "The version-gated Kumo NIM adapter does not support managed serving targets.",
                code="INVALID_CONFIGURATION",
            )
        engine = _load_engine()
        _validate_batch_size(request.batch_size)
        _validate_num_retries(request.num_retries)
        options = dict(request.options)
        _reject_reserved_options(options)
        explain = _resolve_explain(request.explain, options)
        if isinstance(request, KumoRFMTaskRequest):
            _validate_task_request(request)

        # Construct the pinned driver's ordinary HTTP client directly. KumoRFM
        # explicitly supports binding this client through ``_client`` and
        # RFMAPI requires only its ``_request`` operation. The stronger GSF
        # probe has already authenticated, validated the typed health body, and
        # proved the advertised model. It replaces ``KumoClient.authenticate``
        # here because that legacy method hardcodes ``kumo-rfm``. The wrapper
        # below changes only the outbound top-level model field.
        from kumorfm.client.client import KumoClient

        api_client = _RelationalModelClient(
            KumoClient(
                url=transport.url,
                api_key=transport.api_key,
                verify_ssl=transport.verify_ssl,
                timeout=transport.timeout,
                max_retries=transport.max_retries,
            )
        )
        try:
            model = (
                engine.KumoRFM(request.graph, verbose=options["verbose"], _client=api_client)
                if "verbose" in options
                else engine.KumoRFM(request.graph, _client=api_client)
            )
            if request.batch_size is not None:
                batch_context = model.batch_mode(request.batch_size, num_retries=request.num_retries)
            elif request.num_retries:
                batch_context = model.retry(request.num_retries)
            else:
                batch_context = contextlib.nullcontext()
            try:
                with batch_context:
                    if isinstance(request, KumoRFMTaskRequest):
                        result = model.predict_task(
                            _build_task_table(engine, request),
                            run_mode=request.run_mode,
                            explain=explain,
                            **options,
                        )
                    else:
                        result = model.predict(
                            request.query,
                            indices=request.indices,
                            run_mode=request.run_mode,
                            explain=explain,
                            **options,
                        )
            except SdfmError:
                raise
            except Exception as error:
                raise _translate_engine_error(error, transport.url) from error
            return _coerce_result(result, explain is not False)
        finally:
            # This compatibility path constructs the pinned client directly,
            # so it must release that client on every result and exception
            # path. A close failure must not replace the prediction result or
            # the driver's translated exception.
            with contextlib.suppress(Exception):
                api_client.close()


def compatibility_registry() -> AdapterRegistry:
    """Return a client-local registry with normal TabICL and adapted Kumo."""

    registry = AdapterRegistry()
    registry.register(TabICLAdapter())
    registry.register(LegacyKumoRelationalAdapter())
    return registry


__all__ = [
    "LEGACY_ADAPTER_ID",
    "LEGACY_KUMORFM_VERSION",
    "LEGACY_MODEL",
    "LEGACY_SDFM_VERSION",
    "RELATIONAL_MODEL",
    "LegacyKumoRelationalAdapter",
    "compatibility_registry",
]
