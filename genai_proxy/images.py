"""Pure image input parsing helpers used by the proxy."""

import base64
import mimetypes
import os
from urllib.parse import urlparse

import requests


def guess_filename_from_url(image_url):
    """从 URL 推断文件名。"""
    path = urlparse(image_url).path
    filename = os.path.basename(path) or "image"
    if "." not in filename:
        filename += ".jpg"
    return filename


def read_image_from_data_url(data_url):
    """解析 data URL，返回 (bytes, mime_type, filename)。"""
    header, encoded = data_url.split(",", 1)
    mime_type = "image/jpeg"
    if header.startswith("data:"):
        mime_type = header[5:].split(";")[0] or mime_type
    extension = mimetypes.guess_extension(mime_type) or ".jpg"
    image_bytes = base64.b64decode(encoded)
    return image_bytes, mime_type, f"image{extension}"


def fetch_image_bytes(image_url):
    """下载远端图片，返回 (bytes, mime_type, filename)。"""
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    mime_type = response.headers.get("Content-Type", "image/jpeg").split(";")[0].strip() or "image/jpeg"
    filename = guess_filename_from_url(image_url)
    return response.content, mime_type, filename


def parse_image_input_from_message(message):
    """从单条 OpenAI user message 中提取图片输入（URL 或 data URL）。"""
    if not isinstance(message, dict) or message.get("role") != "user":
        return None

    content = message.get("content")
    if not isinstance(content, list):
        return None

    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type not in {"image_url", "input_image"}:
            continue

        if isinstance(part.get("image_url"), dict):
            url_value = part["image_url"].get("url")
            if url_value:
                return url_value
        if isinstance(part.get("image_url"), str):
            return part.get("image_url")
        if isinstance(part.get("url"), str):
            return part.get("url")

    return None
