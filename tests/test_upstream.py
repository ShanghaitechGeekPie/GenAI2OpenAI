import unittest
import json

from genai_proxy.tools import build_tool_calling_messages, parse_tool_calls_from_content
from genai_proxy.upstream import iter_sse_events


class UpstreamSseTests(unittest.TestCase):
    def test_eof_is_not_reported_as_normal_completion(self):
        events = list(iter_sse_events(['data: {"choices":[{"delta":{"content":"partial"}}]}']))
        self.assertEqual(events[-1]["type"], "error")

    def test_length_and_done_marker_are_preserved(self):
        events = list(iter_sse_events(['data: {"choices":[{"delta":{"content":"last"},"finish_reason":"length"}]}']))
        self.assertEqual(events[0]["content"], "last")
        self.assertEqual(events[-1]["finish_reason"], "length")
        events = list(iter_sse_events([': keep-alive', 'data: [DONE]']))
        self.assertEqual(events[-1]["finish_reason"], "stop")

    def test_tool_protocol_follows_client_context_without_mutating_it(self):
        messages = [
            {"role": "system", "content": "Use native tools to inspect files."},
            {"role": "user", "content": "Inspect this project."},
        ]
        tools = [{"type": "function", "function": {"name": "glob"}}]

        result = build_tool_calling_messages(messages, tools, "auto")

        self.assertEqual(result[:-1], messages)
        self.assertEqual(len(messages), 2)
        self.assertEqual(result[-1]["role"], "system")
        self.assertIn('"tool_calls"', result[-1]["content"])
        self.assertIn('"name": "glob"', result[-1]["content"])

    def test_last_delta_is_kept_when_finish_reason_is_present(self):
        fragments = ['{"tool_calls":[', '{"name":"weather","arguments":{}}]}']
        lines = [
            'data: ' + json.dumps({"choices": [{
                "delta": {"content": fragment},
                "finish_reason": "stop" if index else None,
            }]})
            for index, fragment in enumerate(fragments)
        ]

        events = list(iter_sse_events(lines))
        content = "".join(
            event.get("content") or ""
            for event in events
            if event["type"] == "delta"
        )

        self.assertEqual(content, "".join(fragments))
        self.assertEqual(parse_tool_calls_from_content(content)[0]["function"]["name"], "weather")
        self.assertEqual(events[-1]["type"], "done")

    def test_business_error_is_not_an_empty_success(self):
        for payload in [
            {"code": 500, "errMsg": "未找到模型"},
            {"success": False, "message": "Token失效，请重新登录"},
        ]:
            with self.subTest(payload=payload):
                events = list(iter_sse_events(['data: ' + json.dumps(payload)]))
                self.assertEqual([event["type"] for event in events], ["error"])
                self.assertEqual(events[0]["error"], payload.get("errMsg") or payload["message"])

    def test_tool_call_json_is_parsed_after_stream_aggregation(self):
        content = '{"tool_calls":[{"name":"weather","arguments":{"city":"上海"}}]}'

        calls = parse_tool_calls_from_content(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "weather")
        self.assertEqual(calls[0]["function"]["arguments"], '{"city": "上海"}')


if __name__ == "__main__":
    unittest.main()
