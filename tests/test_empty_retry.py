import importlib
import json
import sys
import unittest
from unittest.mock import Mock, patch

from genai_proxy.upstream import ToolCallAccumulator, iter_sse_events


class EmptyRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, 'argv', ['main.py', '--token', 'test-only']):
            cls.server = importlib.import_module('main')

    def run_attempts(self, attempts):
        with patch.object(self.server, '_stream_genai_attempt', side_effect=[iter(x) for x in attempts]) as call:
            result = list(self.server.stream_genai_events([{'role': 'tool', 'content': 'ALREADY_EXECUTED'}], 'deepseek-pro', 100))
            return result, call.call_args_list

    def test_empty_then_content_retries_once_without_changing_history(self):
        result, calls = self.run_attempts([
            [{'type': 'done', 'finish_reason': 'stop'}],
            [{'type': 'delta', 'content': 'OK'}, {'type': 'done', 'finish_reason': 'stop'}],
        ])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result[0]['content'], 'OK')
        self.assertIs(calls[0].args[0], calls[1].args[0])
        self.assertEqual(calls[1].args[0], [{'role': 'tool', 'content': 'ALREADY_EXECUTED'}])

    def test_two_empty_attempts_end_with_error(self):
        result, calls = self.run_attempts([[{'type': 'done'}], [{'type': 'done'}]])
        self.assertEqual(len(calls), 2)
        self.assertEqual([e['type'] for e in result], ['error'])
        self.assertIn('after one retry', result[0]['error'])

    def test_errors_and_length_are_not_retried(self):
        for events in [[{'type': 'error', 'error': 'network error'}], [{'type': 'done', 'finish_reason': 'length'}]]:
            result, calls = self.run_attempts([events])
            self.assertEqual(len(calls), 1)
            self.assertEqual(result, events)

    def test_partial_output_is_not_replayed_after_an_error(self):
        events = [{'type': 'delta', 'content': 'partial'}, {'type': 'error', 'error': 'EOF'}]
        result, calls = self.run_attempts([events])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result, events)

    def test_reasoning_only_attempt_is_discarded_before_retry(self):
        result, calls = self.run_attempts([
            [{'type': 'delta', 'reasoning': 'private thought', 'content': ''}, {'type': 'done'}],
            [{'type': 'delta', 'content': 'OK'}, {'type': 'done'}],
        ])
        self.assertEqual(len(calls), 2)
        self.assertFalse(any(e.get('reasoning') for e in result))

    def test_native_call_counts_as_output_and_is_not_retried(self):
        events = [{'type': 'delta', 'tool_calls': [{'index': 0, 'function': {'name': 'read'}}]},
                  {'type': 'done', 'finish_reason': 'tool_calls'}]
        result, calls = self.run_attempts([events])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result, events)

    def test_close_does_not_start_another_attempt(self):
        with patch.object(self.server, '_stream_genai_attempt', return_value=iter([
            {'type': 'delta', 'content': 'OK'}, {'type': 'done'},
        ])) as attempt:
            events = self.server.stream_genai_events([], 'deepseek-pro', 10)
            next(events)
            events.close()
            self.assertEqual(attempt.call_count, 1)

    def test_route_recovers_from_empty_and_validates_native_fragments(self):
        def response(packets):
            fake = Mock(status_code=200)
            fake.iter_lines.return_value = iter(('data: ' + json.dumps(p)).encode() for p in packets)
            return fake
        empty = response([{'choices': [{'delta': {'content': ''}, 'finish_reason': 'stop'}]}])
        native = response([
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_native', 'function': {'name': 'read_', 'arguments': '{"path":'}}]}, 'finish_reason': None}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'file', 'arguments': '"a.py"}'}}]}, 'finish_reason': 'tool_calls'}]},
        ])
        with patch.object(self.server.requests, 'post', side_effect=[empty, native]) as upstream:
            result = self.server.app.test_client().post('/v1/chat/completions', json={
                'model': 'deepseek-pro', 'stream': True,
                'messages': [{'role': 'user', 'content': 'Read a.py'}],
                'tools': [{'type': 'function', 'function': {'name': 'read_file'}}],
            }).get_data(as_text=True)
            self.assertEqual(upstream.call_count, 2)
        chunks = [json.loads(line[6:]) for line in result.splitlines() if line.startswith('data: {')]
        call = chunks[0]['choices'][0]['delta']['tool_calls'][0]
        self.assertEqual(call['id'], 'call_native')
        self.assertEqual(call['function']['name'], 'read_file')
        self.assertEqual(json.loads(call['function']['arguments']), {'path': 'a.py'})
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'tool_calls')
        empty.close.assert_called_once()
        native.close.assert_called_once()

    def test_nonstream_native_message_uses_declared_tools(self):
        for name in ('read_file', 'undeclared'):
            packet = {'choices': [{'message': {'tool_calls': [{
                'id': 'call_message', 'type': 'function',
                'function': {'name': name, 'arguments': '{"path":"a.py"}'},
            }]}, 'finish_reason': 'tool_calls'}]}
            fake = Mock(status_code=200)
            fake.iter_lines.return_value = iter([('data: ' + json.dumps(packet)).encode()])
            with patch.object(self.server.requests, 'post', return_value=fake) as upstream:
                result = self.server.app.test_client().post('/v1/chat/completions', json={
                    'model': 'deepseek-pro', 'stream': False,
                    'messages': [{'role': 'user', 'content': 'Read a.py'}],
                    'tools': [{'type': 'function', 'function': {'name': 'read_file'}}],
                })
            self.assertEqual(upstream.call_count, 1)
            if name == 'read_file':
                choice = result.get_json()['choices'][0]
                self.assertEqual(choice['finish_reason'], 'tool_calls')
                self.assertEqual(choice['message']['tool_calls'][0]['id'], 'call_message')
            else:
                self.assertIn('Unknown tool', result.get_json()['error'])

    def test_summary_logs_shape_without_prompt_or_credentials(self):
        fake = Mock(status_code=200)
        fake.iter_lines.return_value = iter([b'data: {"choices":[{"delta":{"content":"secret-response-body"},"finish_reason":"stop"}]}'])
        with patch.object(self.server.requests, 'post', return_value=fake), self.assertLogs('genai-proxy', level='INFO') as captured:
            list(self.server.stream_genai_events([{'role': 'user', 'content': 'secret-prompt-body'}],
                                                'deepseek-pro', 100, 'secret-token-value'))
        text = '\n'.join(captured.output)
        self.assertIn('Upstream summary:', text)
        for value in ('secret-response-body', 'secret-prompt-body', 'secret-token-value'):
            self.assertNotIn(value, text)


class UpstreamFieldTests(unittest.TestCase):
    def test_message_fallback_and_reasoning_alias(self):
        packets = [{'choices': [{'message': {'content': 'OK', 'reasoning_content': 'thought'}, 'finish_reason': 'stop'}]}]
        events = list(iter_sse_events(['data: ' + json.dumps(p) for p in packets]))
        self.assertEqual(events[0]['content'], 'OK')
        self.assertEqual(events[0]['reasoning'], 'thought')

    def test_legacy_function_call_is_retained(self):
        packet = {'choices': [{'delta': {'function_call': {'name': 'read', 'arguments': '{}'}}, 'finish_reason': 'function_call'}]}
        events = list(iter_sse_events(['data: ' + json.dumps(packet)]))
        acc = ToolCallAccumulator()
        acc.add(events[0]['tool_calls'])
        self.assertEqual(acc.result()[0]['function']['name'], 'read')
