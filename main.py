from flask import Flask, request, jsonify, stream_with_context, Response
from flask_cors import CORS
import requests
import base64
import io
import json
import uuid
import sqlite3
import hashlib
from datetime import datetime
import argparse

app = Flask(__name__)
CORS(app)

# 解析命令行参数
parser = argparse.ArgumentParser(description='GenAI Flask API Server')
parser.add_argument('--token', type=str, default='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NjMzNjA2MzgsInVzZXJuYW1lIjoiMjAyNDEzNDAyMiJ9.b4E5VzUxkn0Kc1pxkKVipybRFCw47NcppBognTD39e8',
                    help='GenAI API Access Token')
parser.add_argument('--port', type=int, default=5000,
                    help='Flask server port (default: 5000)')
parser.add_argument('--imgtoken', type=str,
                    help='GenAI image upload token')
args = parser.parse_args()

# GenAI API 配置
GENAI_URL = "https://genai.shanghaitech.edu.cn/htk/chat/start/chat"
UPLOAD_URL = "https://genaipic.shanghaitech.edu.cn/sys/common/upload"
STATIC_BASE_URL = "https://genaipic.shanghaitech.edu.cn/sys/common/static/"
GENAI_HEADERS = {
    "Accept": "*/*, text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://genai.shanghaitech.edu.cn",
    "Referer": "https://genai.shanghaitech.edu.cn/dialogue",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "X-Access-Token": args.token,
    "Token": args.imgtoken,
    "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

# 数据库初始化
DB_FILE = "image_cache.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (hash TEXT PRIMARY KEY, url TEXT)''')
    conn.commit()
    conn.close()

init_db()

def get_url_from_db(base64_str):
    """根据Base64内容的哈希获取缓存URL"""
    img_hash = hashlib.md5(base64_str.encode('utf-8')).hexdigest()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT url FROM images WHERE hash=?", (img_hash,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def save_url_to_db(base64_str, url):
    """保存Base64哈希和URL的映射"""
    img_hash = hashlib.md5(base64_str.encode('utf-8')).hexdigest()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO images (hash, url) VALUES (?, ?)", (img_hash, url))
    conn.commit()
    conn.close()

def upload_image(base64_str):
    """上传图片"""
    try:
        original_b64 = base64_str
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]

        # 查看缓存
        cached_url = get_url_from_db(base64_str)
        if cached_url:
            print(f"DEBUG: Image found in cache: {cached_url}")
            return cached_url

        # 上传
        img_data = base64.b64decode(base64_str)

        common_headers = {
            "User-Agent": GENAI_HEADERS["User-Agent"],
            "Accept": "*/*",
            "Referer": "https://genai.shanghaitech.edu.cn/",
            "Origin": "https://genai.shanghaitech.edu.cn",
        }

        files = {'file': ('image.png', io.BytesIO(img_data), 'image/png')}
        data = {'biz': 'temp', 'uploadType': 'local'}
        post_headers = dict(common_headers, **{"token": args.imgtoken}) # 确保这里用了 imgtoken

        res = requests.post(
            UPLOAD_URL,
            files=files, data=data, headers=post_headers, verify=False
        )

        res_json = res.json()
        if res_json.get("success") and res_json.get("result"):
            full_url = STATIC_BASE_URL + res_json["result"]["url"]

            # 写入缓存
            save_url_to_db(base64_str, full_url)
            return full_url

    except Exception as e:
        print(f"Upload Critical Error: {e}")

    return None


def convert_messages_to_genai_format(messages):
    """转换格式"""
    if not messages:
        return [], "", ""

    current_msg = messages[-1]
    history_messages = messages[:-1]

    chat_info = ""
    image_url = ""

    # 历史图片base64 -> url
    processed_history = []
    for msg in history_messages:
        new_msg = msg.copy()
        content = msg.get("content")

        if isinstance(content, list):
            new_content = []
            for item in content:
                if item.get("type") == "image_url":
                    raw_url = item.get("image_url", {}).get("url", "")
                    if raw_url.startswith("data:"):
                        url = upload_image(raw_url)
                        if url:
                            item["image_url"]["url"] = url
                new_content.append(item)
            new_msg["content"] = new_content

        processed_history.append(new_msg)

    # 当前
    if current_msg.get("role") == "user":
        content = current_msg.get("content")
        if isinstance(content, str):
            chat_info = content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    raw_url = item.get("image_url", {}).get("url", "")
                    url = upload_image(raw_url)
                    if url:
                        image_url = url
            chat_info = " ".join(text_parts)

    return processed_history, chat_info, image_url


def extract_content_from_genai(response_data):
    """从GenAI API响应中提取内容"""
    try:
        if "choices" in response_data and len(response_data["choices"]) > 0:
            delta = response_data["choices"][0].get("delta", {})
            if "reasoning_content" in delta:
                return delta["reasoning_content"]
            content = delta.get("content", "")
            return content
    except (KeyError, IndexError, TypeError):
        pass
    return None

def stream_genai_response(chat_info, messages, model, max_tokens, image_url = None):
    """流式调用GenAI API并转换为OpenAI格式"""
    
    # 确定 rootAiType
    azure_models = {"GPT-5.2", "GPT-5-Pro", "GPT-5", "o4-mini", "GPT-4.1", "o3", "GPT-4.1-mini"}
    root_ai_type = "azure" if model in azure_models else "xinference"

    # 构建GenAI请求数据
    genai_data = {
        "chatInfo": chat_info,
        "messages": messages,
        "type": "3",
        "stream": True,
        "aiType": model,
        "aiSecType": "1",
        "promptTokens": 0,
        "rootAiType": root_ai_type,
        "maxToken": max_tokens or 30000
    }

    if image_url:
        genai_data["imageUrl"] = image_url
    
    try:
        
        # 调用GenAI API
        response = requests.post(
            GENAI_URL,
            headers=GENAI_HEADERS,
            json=genai_data,
            stream=True,
            timeout=60
        )
        
        # 打印原始响应状态
        # ic(f"DEBUG: GenAI API Response Status: {response.status_code}")
        
        if response.status_code != 200:
            yield f"data: {json.dumps({'error': f'GenAI API error: {response.status_code}'})}\n\n"
            return
        
        # 处理流式响应
        finished = False
        for line in response.iter_lines():
            if finished:
                break
                
            if line:
                try:
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                    
                    # 处理SSE格式
                    if line_str.startswith('data:'):
                        line_str = line_str[5:].strip()
                    
                    if line_str:
                        # ic(f"DEBUG: Raw response line: {line_str}")  # 打印原始响应行
                        genai_json = json.loads(line_str)
                        
                        # 检查是否已经完成
                        if "choices" in genai_json and len(genai_json["choices"]) > 0:
                            choice = genai_json["choices"][0]
                            if choice.get("finish_reason") is not None:
                                finished = True
                        
                        if finished:
                            # 发送完成信号后跳出循环
                            final_response = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "text_completion.chunk",
                                "created": int(datetime.now().timestamp()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop"
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(final_response)}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                        
                        content = extract_content_from_genai(genai_json)
                        
                        if content is not None:
                            # 转换为OpenAI格式
                            openai_response = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "text_completion.chunk",
                                "created": int(datetime.now().timestamp()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(openai_response)}\n\n"
                
                except json.JSONDecodeError:
                    pass
        
        # 发送完成信号
        final_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_response)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI兼容的聊天完成端点"""
    try:
        req_data = request.get_json()
        
        # 验证必要字段
        if not req_data or 'messages' not in req_data:
            return jsonify({'error': 'Missing messages field'}), 400
        
        messages = req_data.get('messages', [])
        model = req_data.get('model', 'gpt-3.5-turbo')
        stream = req_data.get('stream', False)
        max_tokens = req_data.get('max_tokens', 30000)
        
        # 转换消息格式
        messages, chat_info, image_url = convert_messages_to_genai_format(messages)
        
        if not chat_info and not image_url:
            return jsonify({'error': 'No user message found'}), 400

        # 流式响应
        if stream:
            return Response(
                stream_with_context(stream_genai_response(
                    chat_info,
                    messages,
                    model, 
                    max_tokens,
                    image_url
                )),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Content-Type': 'text/event-stream',
                }
            )
        
        # 非流式响应（收集所有内容后返回）
        else:
            complete_content = ""
            for line in stream_genai_response(chat_info, messages, model, max_tokens, image_url):
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                complete_content += content
                    except json.JSONDecodeError:
                        pass
            
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "text_completion",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": complete_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(complete_content),
                    "total_tokens": len(complete_content)
                }
            }
            return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    available_models = [
        "deepseek-v3:671b",
        "deepseek-r1:671b",
        "GPT-5.2",
        "GPT-5-Pro",
        "GPT-5",
        "o4-mini",
        "GPT-4.1",
        "o3",
        "GPT-4.1-mini",
        "qwen-instruct",
        "qwen-think"
    ]

    models = []
    for model_id in available_models:
        models.append({
            "id": model_id,
            "object": "model",
            "owned_by": "genai",
            "permission": []
        })

    return jsonify({"object": "list", "data": models})

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # 运行Flask应用
    app.run(host='0.0.0.0', port=args.port, debug=False)
