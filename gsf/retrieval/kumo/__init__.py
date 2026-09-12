# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KumoRFM-backed prediction for the text-to-SQL agent."""

from gsf.retrieval.kumo.graph_contract import GraphContractPredictionScope
from gsf.retrieval.kumo.predictor import PredictionContext
from gsf.retrieval.kumo.predictor import build_prediction_context
from gsf.retrieval.kumo.predictor import run_prediction

__all__ = [
    "GraphContractPredictionScope",
    "PredictionContext",
    "build_prediction_context",
    "run_prediction",
]
