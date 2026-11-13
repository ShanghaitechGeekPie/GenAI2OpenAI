import json
import unittest

from genai_proxy.tools import normalize_messages_for_genai, parse_tool_calls_from_content


class ToolParsingTests(unittest.TestCase):
    def setUp(self):
        self.tools = [{"type": "function", "function": {"name": "read_file"}}]
        self.call = {
            "id": "call_example_0", "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({
                "filePath": "/proj/finance_match/.gitignore", "startLine": 1, "endLine": 40,
            })},
        }

    def test_screenshot_formats_preserve_id_and_arguments(self):
        for text in [json.dumps([self.call]), json.dumps([self.call]) + '\n(https://microsoft.com))',
                     json.dumps(self.call), '```json\n' + json.dumps([self.call]) + '\n```',
                     '工具调用：\n' + json.dumps([self.call]) + '\n结束。']:
            with self.subTest(text=text):
                calls = parse_tool_calls_from_content(text, self.tools)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]['id'], self.call['id'])
                self.assertEqual(calls[0]['function'], self.call['function'])

    def test_multiple_array_and_xml_calls_are_not_lost(self):
        plain = {"name": "read_file", "arguments": {"filePath": "b.py"}}
        for text in [json.dumps([self.call, plain]),
                     f'<tool_call>{json.dumps(self.call)}</tool_call><tool_call>{json.dumps(plain)}</tool_call>']:
            self.assertEqual(len(parse_tool_calls_from_content(text, self.tools)), 2)

    def test_invalid_calls_do_not_fall_back_to_text_or_execute_partial_arrays(self):
        invalid = [
            '[ ' + json.dumps(self.call) + ', {"function":',
            '[ {"name":"read_file","arguments":{}},',
            '<tool_call>{"name":"read_file","arguments":{}}',
            json.dumps({"tool_calls": "bad"}),
            json.dumps({"name": "not_declared", "arguments": {}}),
            json.dumps({"name": "read_file", "arguments": '{"path":'}),
            json.dumps({"name": "read_file", "arguments": []}),
            json.dumps([self.call, None]),
        ]
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_tool_calls_from_content(text, self.tools)

    def test_only_missing_outer_wrapper_closures_are_repaired(self):
        wrapped = json.dumps({"tool_calls": [self.call]})
        array = json.dumps([self.call])
        for text in [wrapped[:-1], wrapped[:-2], array[:-1]]:
            with self.subTest(text=text):
                calls = parse_tool_calls_from_content(text, self.tools)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]['id'], self.call['id'])
        # 不能补齐调用对象或参数、字符串、逗号后的缺失值。
        for text in [wrapped[:-3], wrapped[:-2] + ',',
                     wrapped[:-1] + ',"other":[',
                     '{"tool_calls":[{"name":"read_file","arguments":{"filePath":"abc',
                     '{"tool_calls":[{"name":"read_file","arguments":{"filePath":"abc"',
                     '[{"name":"read_file","arguments":{"filePath":"abc"}']:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_tool_calls_from_content(text, self.tools)

    def test_normal_explanations_and_json_data_are_not_tool_protocol(self):
        for text in ['已完成，字段为 `"name": "demo"`。',
                     '```json\n{"name":"demo","version":"1.0"}\n```',
                     '检查结果：{"ok":true}\n字段示例包含 "name": "read_file"。',
                     '{"tool_calls":[]}',
                     '{"function":"documentation","description":"API 示例"}',
                     '[{"name":"demo","version":"1.0"}]']:
            with self.subTest(text=text):
                self.assertEqual(parse_tool_calls_from_content(text, self.tools), [])

    def test_required_choice_fails_explicitly_but_auto_allows_final_text(self):
        self.assertEqual(parse_tool_calls_from_content('完成。', self.tools), [])
        self.assertEqual(parse_tool_calls_from_content('[{"value":1}]', self.tools), [])
        with self.assertRaises(ValueError):
            parse_tool_calls_from_content('我来检查一下。', self.tools, 'required')
        with self.assertRaises(ValueError):
            parse_tool_calls_from_content('完成。', self.tools, {"type": "function", "function": {"name": "read_file"}})

    def test_assistant_text_does_not_drop_call_history(self):
        result = normalize_messages_for_genai([{"role": "assistant", "content": "读取文件。", "tool_calls": [self.call]}])
        self.assertIn('读取文件。', result[0]['content'])
        self.assertIn('read_file', result[0]['content'])
