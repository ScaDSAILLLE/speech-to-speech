from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any, cast

from openai import Stream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_named_tool_choice_param import Function as NamedToolChoiceFunction
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import ResponseFunctionToolCall
from openai.types.shared_params import FunctionDefinition

from speech_to_speech.LLM.base_openai_compatible_language_model import (
    WARMUP_MAX_RETRIES,
    AssistantMessage,
    BaseOpenAICompatibleHandler,
    ProviderEvent,
    TextDelta,
    ToolCall,
    Usage,
)
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)


def _to_chat_tools(req_tools: Any) -> list[ChatCompletionToolParam] | None:
    """Convert Responses-API function tools to Chat-Completions tool format.

    Responses tools are flat ``{type:"function", name, description, parameters}``;
    Chat Completions nests them under a ``function`` key. Items already in the
    nested form (or non-function tools) are passed through untouched.
    """
    if not req_tools:
        return None
    chat_tools: list[ChatCompletionToolParam] = []
    for t in req_tools:
        d = t if isinstance(t, dict) else t.model_dump(exclude_none=True)
        if d.get("type") == "function" and "function" not in d:
            fn = FunctionDefinition(name=d["name"])
            if d.get("description") is not None:
                fn["description"] = d["description"]
            if d.get("parameters") is not None:
                fn["parameters"] = d["parameters"]
            chat_tools.append(ChatCompletionToolParam(type="function", function=fn))
        else:
            chat_tools.append(cast("ChatCompletionToolParam", d))
    return chat_tools


def _to_chat_tool_choice(tool_choice: Any) -> ChatCompletionToolChoiceOptionParam:
    """Convert a Responses-API tool_choice to Chat-Completions form.

    The string forms ("auto"/"required"/"none") are identical across both APIs;
    only the forced-function object differs (flat ``name`` vs nested ``function``).
    """
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" and "name" in tool_choice:
        return ChatCompletionNamedToolChoiceParam(
            type="function", function=NamedToolChoiceFunction(name=tool_choice["name"])
        )
    if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
        d = tool_choice.model_dump(exclude_none=True)
        if d.get("type") == "function" and "name" in d:
            return ChatCompletionNamedToolChoiceParam(type="function", function=NamedToolChoiceFunction(name=d["name"]))
        return cast("ChatCompletionToolChoiceOptionParam", d)
    return cast("ChatCompletionToolChoiceOptionParam", tool_choice)


class ChatCompletionsApiModelHandler(BaseOpenAICompatibleHandler):
    """LLM handler that talks to an OpenAI-compatible ``/v1/chat/completions`` server.

    Functionally mirrors :class:`ResponsesApiModelHandler` but uses the mature
    Chat Completions streaming tool-call protocol (``choices[].delta.tool_calls``)
    instead of ``/v1/responses``. This is the robust path for vLLM + Qwen tool
    calling. The conversation is serialised with :meth:`Chat.to_transformers_chat`,
    which already emits OpenAI chat messages including ``tool_calls``/``tool`` roles.
    """

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()

        # LiteRT-LM (and similar model servers) load the model on first
        # request — that can take 30-60 s on a Pi. The OpenAI SDK retries with
        # exponential backoff but only ~21 s total, which races the load.
        # Poll /v1/models first until the endpoint responds, then run the
        # actual warmup call against an already-warm server.
        if self.client.base_url and self._wait_for_endpoint():
            logger.info(
                f"{self.__class__.__name__}: model endpoint is up, sending warmup request"
            )
        else:
            logger.warning(
                f"{self.__class__.__name__}: endpoint not reachable after timeout, "
                "warmup may fail — proceeding anyway"
            )

        self.client.with_options(max_retries=WARMUP_MAX_RETRIES).chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            extra_body=self._extra_body,
            timeout=self.request_timeout,
        )
        end = time.time()
        logger.info(f"{self.__class__.__name__}:  warmed up! time: {(end - start):.3f} s")

    def _wait_for_endpoint(self, timeout: float = 180.0, interval: float = 1.0) -> bool:
        """Block until GET /v1/models returns 2xx or `timeout` seconds elapse.

        Returns True if the endpoint is reachable, False on timeout. Cheap
        HEAD probe so we don't trigger model load twice.
        """
        import httpx

        deadline = time.time() + timeout
        base_url = str(self.client.base_url).rstrip('/')
        url = f"{base_url}/models"
        while time.time() < deadline:
            try:
                resp = httpx.get(url, timeout=2.0)
                if 200 <= resp.status_code < 300:
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return False

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        """Return a generate fn that calls Chat Completions for compaction."""
        client = self.client
        model_name = self.model_name
        timeout = self.request_timeout
        extra_body = self._extra_body

        def generate(system: str, user: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra_body=extra_body,
                timeout=timeout,
            )
            return response.choices[0].message.content or ""

        return generate

    @staticmethod
    def _to_chat_content_part(part: dict[str, Any]) -> ChatCompletionContentPartParam:
        """Convert one transformers content part to Chat-Completions shape.

        ``to_transformers_chat`` keeps Realtime-style parts (``input_text`` /
        ``input_image`` with a bare-string ``image_url``). The Chat Completions
        HTTP API instead wants ``{type:"text", text}`` and
        ``{type:"image_url", image_url:{url, detail}}``. Unknown parts pass through.
        """
        ptype = part.get("type")
        if ptype == "input_text":
            return ChatCompletionContentPartTextParam(type="text", text=part.get("text") or "")
        if ptype == "input_image":
            raw_url: Any = part.get("image_url")
            if isinstance(raw_url, dict):
                image_url = cast("ImageURL", raw_url)
            else:
                image_url = ImageURL(url=raw_url)
                detail = part.get("detail")
                if detail is not None:
                    image_url["detail"] = detail
            return ChatCompletionContentPartImageParam(type="image_url", image_url=image_url)
        return cast("ChatCompletionContentPartParam", part)

    @classmethod
    def _chat_messages(cls, chat: Chat) -> list[dict[str, Any]]:
        """Serialise the chat for the Chat Completions API.

        ``Chat.to_transformers_chat`` targets HuggingFace ``apply_chat_template``,
        so two shapes need fixing up for the OpenAI Chat Completions HTTP API:
        tool-call ``arguments`` must be a JSON *string* (not a parsed object), and
        multimodal ``content`` parts must use the Chat Completions ``text`` /
        ``image_url`` shape rather than the Realtime ``input_text`` /
        ``input_image`` shape.
        """
        messages = chat.to_transformers_chat()
        for message in messages:
            for tool_call in message.get("tool_calls") or []:
                fn = tool_call.get("function")
                if fn is not None and not isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.dumps(fn.get("arguments") or {}, ensure_ascii=False)
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [cls._to_chat_content_part(p) for p in content]
            if message.get("role") == "tool":
                message.pop("name", None)
        return messages

    # ── base hooks ──────────────────────────────────────────────────────────--

    def _serialize(self, active_chat: Chat) -> list[dict[str, Any]]:
        return self._chat_messages(active_chat)

    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        optional_kwargs: dict[str, Any] = {}
        chat_tools = _to_chat_tools(req_tools)
        if chat_tools is not None:
            optional_kwargs["tools"] = chat_tools
        if req_tool_choice is not None:
            optional_kwargs["tool_choice"] = _to_chat_tool_choice(req_tool_choice)
        return optional_kwargs

    def _request(self, api_input: list[dict[str, Any]], optional_kwargs: dict[str, Any]) -> Any:
        create_kwargs: dict[str, Any] = dict(optional_kwargs)
        session_id = create_kwargs.pop("__session_id", None)
        if self.stream:
            create_kwargs["stream_options"] = {"include_usage": True}
        extra_body: dict[str, Any] = dict(self._extra_body) if self._extra_body else {}
        if session_id:
            extra_body["session_id"] = session_id
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=api_input,  # type: ignore[arg-type]  # runtime dicts match the Chat Completions message shape
            stream=self.stream,
            extra_body=extra_body or None,
            timeout=self.request_timeout,
            **create_kwargs,
        )

    def _iter_stream_events(self, api_response: Stream[ChatCompletionChunk]) -> Iterator[ProviderEvent]:
        # Accumulate streamed tool-call deltas, keyed by their stream index, and the
        # raw assistant text, then emit assistant message + tool calls + usage once
        # the stream is exhausted.
        tool_accum: dict[int, dict[str, str]] = {}
        usage: Usage | None = None
        raw_text = ""
        for chunk in api_response:
            # Usage-only trailing chunk (choices == []) when include_usage is set.
            if chunk.usage is not None:
                usage = Usage(
                    input_tokens=chunk.usage.prompt_tokens or 0, output_tokens=chunk.usage.completion_tokens or 0
                )
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    entry = tool_accum.setdefault(tc.index, {"name": "", "args": "", "id": ""})
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            entry["name"] = tc.function.name
                        if tc.function.arguments:
                            entry["args"] += tc.function.arguments
            # A refusal streams as `delta.refusal` with `delta.content` None;
            # surface it as assistant text so it is spoken and stored.
            text_piece = delta.content or getattr(delta, "refusal", None)
            if text_piece:
                raw_text += text_piece
                yield TextDelta(text=text_piece)

        if raw_text.strip():
            yield AssistantMessage(content=[AssistantContent(type="output_text", text=raw_text)])
        yield from self._tool_calls_from_accum(tool_accum)
        if usage is not None:
            yield usage

    def _iter_response_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        usage = api_response.usage
        if usage:
            yield Usage(input_tokens=usage.prompt_tokens or 0, output_tokens=usage.completion_tokens or 0)
        # A valid-but-empty response (e.g. content filter) returns no choices;
        # complete cleanly with no assistant text rather than raising IndexError.
        message = api_response.choices[0].message if api_response.choices else None
        if message is None:
            return
        # A refusal arrives as `message.refusal` with `message.content` None; treat
        # it as assistant text so it is spoken and stored.
        raw_content = message.content or getattr(message, "refusal", None)
        if raw_content:
            yield AssistantMessage(content=[AssistantContent(type="output_text", text=raw_content)])
            yield TextDelta(text=raw_content)
        tool_accum: dict[int, dict[str, str]] = {}
        for tc in message.tool_calls or []:
            tool_accum[len(tool_accum)] = {
                "name": tc.function.name or "",
                "args": tc.function.arguments or "",
                "id": tc.id or "",
            }
        yield from self._tool_calls_from_accum(tool_accum)

    @staticmethod
    def _tool_calls_from_accum(tool_accum: dict[int, dict[str, str]]) -> Iterator[ToolCall]:
        """Turn accumulated tool-call deltas into ToolCall events.

        IDs are regenerated (mirroring the Responses handler) so the rest of the
        pipeline pairs each call_id with its function_call_output consistently.
        """
        for index in sorted(tool_accum):
            entry = tool_accum[index]
            if not entry["name"]:
                continue
            yield ToolCall(
                item=ResponseFunctionToolCall(
                    type="function_call",
                    name=entry["name"],
                    arguments=entry["args"] or "{}",
                    call_id=_generate_id("call"),
                    id=_generate_id("fc"),
                    status="completed",
                )
            )

    def on_session_end(self) -> None:
        logger.debug("Chat Completions API language model session state reset")
