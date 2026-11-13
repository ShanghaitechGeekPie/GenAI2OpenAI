import importlib
import sys
import unittest
from unittest.mock import Mock, patch

from genai_proxy.models import build_remote_model_list


class RemoteModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 指定占位 token，避免导入入口时读取缓存或触发自动登录。
        with patch.object(sys, 'argv', ['main.py', '--token', 'startup-test-token']):
            cls.server = importlib.import_module('main')

    def setUp(self):
        token = patch.dict(self.server.GENAI_HEADERS, {'X-Access-Token': 'startup-test-token'})
        token.start()
        self.addCleanup(token.stop)
        self.client = self.server.app.test_client()

    def test_upstream_ids_are_preserved_and_invalid_records_skipped(self):
        result = build_remote_model_list([
            {'aiType': 'GPT-6.1-Sol', 'simpleName': 'Display name'},
            {'aiType': 'GPT-6.1-Sol'}, {'aiType': 'deepseek-pro'},
            {}, None, {'aiType': ''}, {'aiType': 123},
        ])
        self.assertEqual(result['object'], 'list')
        self.assertEqual([m['id'] for m in result['data']], ['GPT-6.1-Sol', 'deepseek-pro'])

    def test_list_uses_request_token_and_refreshes_on_each_request(self):
        response = Mock()
        response.json.side_effect = [
            {'success': True, 'result': {'records': [{'aiType': 'first-model'}]}},
            {'success': True, 'result': {'records': [{'aiType': 'new-model'}]}},
        ]
        with patch.object(self.server.requests, 'get', return_value=response) as upstream:
            first = self.client.get('/v1/models', headers={'Authorization': 'Bearer request-token'})
            second = self.client.get('/v1/models', headers={'X-API-Key': 'another-token'})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json['data'][0]['id'], 'first-model')
        self.assertEqual(second.json['data'][0]['id'], 'new-model')
        self.assertEqual(upstream.call_args_list[0].kwargs['headers']['X-Access-Token'], 'request-token')
        self.assertEqual(upstream.call_args_list[1].kwargs['headers']['X-Access-Token'], 'another-token')

    def test_list_uses_startup_token_without_request_token(self):
        response = Mock()
        response.json.return_value = {'success': True, 'result': {'records': []}}
        with patch.object(self.server.requests, 'get', return_value=response) as upstream:
            result = self.client.get('/v1/models')
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json['data'], [])
        self.assertEqual(upstream.call_args.kwargs['headers']['X-Access-Token'], 'startup-test-token')

    def test_upstream_failure_does_not_return_static_models(self):
        for payload in [
            {'success': False, 'message': 'Token invalid'},
            {'success': True, 'result': {}},
        ]:
            with self.subTest(payload=payload):
                response = Mock()
                response.json.return_value = payload
                with patch.object(self.server.requests, 'get', return_value=response):
                    result = self.client.get('/v1/models')
                self.assertEqual(result.status_code, 502)
                self.assertIn('error', result.json)
                self.assertNotIn('data', result.json)


if __name__ == '__main__':
    unittest.main()
