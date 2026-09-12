# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ask_question`` — GSF structured retrieval and prediction as one MCP tool.

Hand-written rather than generated, because ``POST /api/chat/completions``
answers with ``text/event-stream``: the agent emits a step event per graph node
and only then a result. ``FastMCP.from_openapi`` produces request/response
tools and has nothing useful to say about a stream.

Streaming is not merely an obstacle here. A run is many sequential model calls
and routinely takes tens of seconds, which is long enough that a silent tool
looks hung, so step boundaries are forwarded as finite, non-reasoning progress
labels.

The response the UI receives is markdown with fenced SQL, shaped for a renderer.
This returns the parts separately instead, so a calling agent can use the SQL
and the rows without parsing prose.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from gsf_mcp.config import Settings

logger = logging.getLogger(__name__)

# An agent's context is finite and a question can legitimately match thousands
# of rows. Return a readable prefix and say how much was withheld, rather than
# flooding the caller or silently truncating.
MAX_ROWS = 100

_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"
_SAFE_PROGRESS_LABELS = {
    "question_extraction": "Understanding the question",
    "classify_prediction": "Checking the query path",
    "prepare_prediction_graph": "Preparing prediction",
    "kumo_predict": "Running prediction",
    "retrieve_candidates": "Retrieving ontology candidates",
    "prepare_candidates": "Preparing ontology candidates",
    "precheck_combined": "Checking joins and filter values",
    "construct_sql_from_candidates": "Constructing query",
    "reconstruct_sql": "Repairing query",
    "validate_sql_query": "Validating query",
    "validate_intent": "Validating requested outcome",
    "execute_sql_query": "Executing query",
    "check_empty_like_result": "Checking results",
    "check_value_repair": "Checking values",
    "format_and_respond": "Formatting response",
    "unconstructable_sql_response": "Query could not be constructed",
}
_UNKNOWN_PROGRESS_LABEL = "Processing structured-data step"


class DataAnswer(BaseModel):
    """Structured result of one ``ask_question`` call."""

    answer: str = Field(description="Natural-language answer to the question.")
    sql: str = Field(
        default="",
        description="The SQL or PQL query that produced the rows.",
    )
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description=f"Result rows, at most {MAX_ROWS} of them.",
    )
    row_count: int = Field(
        default=0, description="Rows returned by the query before truncation."
    )
    truncated: bool = Field(
        default=False, description=f"True when more than {MAX_ROWS} rows matched."
    )
    resolution_lineage: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Observed mappings from extracted business phrases to governed "
            "ontology objects and their physical table/column bindings."
        ),
    )


def _parse_rows(raw: Any) -> list[dict[str, Any]]:
    """Normalise the agent's result payload into a list of row dicts.

    The backend sends either a one-item list holding a JSON-records string or
    the records themselves, depending on which path produced the answer.
    """
    if raw is None:
        return []

    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        try:
            raw = json.loads(raw[0])
        except json.JSONDecodeError:
            logger.warning("Result payload was a string but not JSON; dropping rows")
            return []

    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def _build_answer(payload: dict[str, Any]) -> DataAnswer:
    # A successful run reports rows under `sql_response_from_db`. `result` is
    # only present in the failure shape `_extract_answer` falls back to, where
    # it is None — so reading it first would discard every real answer.
    rows = _parse_rows(payload.get("sql_response_from_db") or payload.get("result"))
    return DataAnswer(
        answer=str(payload.get("response") or ""),
        sql=str(payload.get("sql_code") or ""),
        rows=rows[:MAX_ROWS],
        row_count=len(rows),
        truncated=len(rows) > MAX_ROWS,
        resolution_lineage=[
            {
                str(key): str(value)
                for key, value in item.items()
                if key in {"phrase", "ontology_object", "table", "column"}
                and isinstance(value, str)
            }
            for item in list(payload.get("resolution_lineage") or [])[:40]
            if isinstance(item, dict)
        ],
    )


def _decode_event(line: str) -> dict[str, Any] | None:
    """Return the JSON object on an SSE ``data:`` line, if there is one.

    Heartbeats arrive as ``: ping`` comments and blank lines separate events;
    both are skipped. A malformed payload is dropped rather than failing the
    run — losing one progress update is better than losing the answer.
    """
    if not line.startswith(_SSE_DATA_PREFIX):
        return None
    payload = line[len(_SSE_DATA_PREFIX) :].strip()
    if not payload or payload == _SSE_DONE:
        return None
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Skipping unparseable SSE payload: %.200s", payload)
        return None
    return event if isinstance(event, dict) else None


async def _raise_for_status(response: httpx.Response) -> None:
    """Turn a non-200 chat response into a ToolError the caller can act on."""
    if response.status_code == 200:
        return

    await response.aread()
    try:
        detail = response.json().get("detail") or response.json().get("error")
    except (json.JSONDecodeError, ValueError):
        detail = response.text.strip()[:300]

    if response.status_code in (401, 403):
        raise ToolError(
            "GSF rejected the credentials for this question. Sign in again, "
            "and check that the account is allowed to use chat. "
            f"({detail})"
        )
    if response.status_code == 409:
        # Either the semantic layer was never built, or this conversation
        # already has a turn in flight. The backend's detail distinguishes them.
        raise ToolError(f"GSF cannot answer right now: {detail}")
    raise ToolError(f"GSF returned HTTP {response.status_code}: {detail}")


def register(mcp: FastMCP, settings: Settings, client: httpx.AsyncClient) -> None:
    """Attach ``ask_question`` to *mcp*."""

    @mcp.tool(
        name="ask_question",
        description=(
            "Ask a natural-language question about the data connected to this "
            "GSF deployment. GSF resolves the question against its semantic "
            "layer, runs structured retrieval or prediction, and returns the "
            "answer together with the SQL or PQL it used.\n\n"
            "This is the primary tool and the reason GSF exists. It is also "
            "slow — many sequential model calls, typically tens of seconds — "
            "so use check_answerable first when you are unsure the question is "
            "in scope, and prefer one well-formed question over several "
            "narrow ones."
        ),
    )
    async def ask_question(
        question: str,
        ctx: Context,
        conversation_id: str | None = None,
        evidence: str | None = None,
        target_db: str | None = None,
        prediction: bool | None = None,
    ) -> DataAnswer:
        """Run one structured-data turn and return its structured result.

        Args:
            question: The question, in plain language.
            conversation_id: Continue an existing thread, so follow-up
                questions resolve against earlier turns. Requires the caller
                to hold conversation-write permission, and rejects a second
                question while the first is still running.
            evidence: Optional authoritative evidence supplied separately
                from the question.
            target_db: Exact configured database scope. In managed agent
                deployments this is injected from the immutable run scope.
            prediction: ``True`` forces prediction, ``False`` forces SQL, and
                ``None`` lets GSF classify. Managed deployments inject this
                only when the work item already selected a branch.
        """
        body: dict[str, Any] = {"question": question}
        if conversation_id:
            body["conversation_id"] = conversation_id
        if evidence:
            body["evidence"] = evidence
        if target_db:
            body["target_db"] = target_db
        if prediction is not None:
            body["prediction"] = prediction

        answer: DataAnswer | None = None
        steps = 0

        try:
            async with client.stream(
                "POST",
                "/api/chat/completions",
                json=body,
                headers={"Accept": "text/event-stream"},
                timeout=settings.chat_timeout_s,
            ) as response:
                await _raise_for_status(response)

                async for line in response.aiter_lines():
                    event = _decode_event(line)
                    if event is None:
                        continue

                    kind = event.get("type")
                    if kind == "step":
                        steps += 1
                        node = event.get("node")
                        label = _SAFE_PROGRESS_LABELS.get(
                            node if isinstance(node, str) else "",
                            _UNKNOWN_PROGRESS_LABEL,
                        )
                        # No total: the graph's path depends on the question,
                        # so the step count is not known ahead of time.
                        await ctx.report_progress(
                            progress=steps,
                            message=label,
                        )
                    elif kind == "result":
                        answer = _build_answer(event.get("answer") or {})
                    elif kind == "error":
                        raise ToolError(
                            f"GSF agent failed: {event.get('message') or 'unknown'}"
                        )
        except httpx.TimeoutException as exc:
            raise ToolError(
                f"GSF did not answer within {settings.chat_timeout_s:.0f}s. "
                "Raise GSF_MCP_CHAT_TIMEOUT_S, or ask a narrower question."
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolError(
                f"Could not reach GSF at {settings.api_url}: {exc}"
            ) from exc

        if answer is None:
            # The stream ended with no result and no error. This is what a
            # cancelled run looks like from the outside.
            raise ToolError(
                "GSF ended the run without producing an answer. It may have "
                "been cancelled, or the question may not map to the data."
            )
        return answer


__all__ = ["DataAnswer", "MAX_ROWS", "register"]
