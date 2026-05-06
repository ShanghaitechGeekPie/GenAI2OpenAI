import base64
import re
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

GENAI_BASE_URL = "https://genai.shanghaitech.edu.cn"
IDS_BASE_URL = "https://ids.shanghaitech.edu.cn"


class LoginError(Exception):
    pass


def encrypt_password(password: str, salt: str) -> str:
    prefix = b"Nu1L" * 16
    combined = prefix + password.encode()
    iv = b"Nu1LNu1LNu1LNu1L"
    key = salt.encode()
    encrypted = AES.new(key, AES.MODE_CBC, iv).encrypt(pad(combined, AES.block_size))
    return base64.b64encode(encrypted).decode()


def _collect_data(html: str, name: str, end_tag: str = "/>" ) -> str:
    start = html.find(f'id="{name}"') if name == "pwdEncryptSalt" else html.find(f'name="{name}"')
    if start == -1:
        return ""
    end = html.find(end_tag, start)
    raw = html[start:end]
    value_start = raw.find('value="') + 7
    value_end = raw.find('"', value_start)
    return raw[value_start:value_end]


def _get_service_url(html: str) -> str | None:
    matched = re.search(r'var service = \["(.*?)"', html)
    return matched.group(1) if matched else None


def login_genai(student_id: str, password: str) -> str:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    })

    resp = session.get(f"{GENAI_BASE_URL}/htk/user/login", allow_redirects=False, timeout=30)
    if 300 <= resp.status_code < 400:
        ids_login_url = urljoin(f"{GENAI_BASE_URL}/htk/user/login", resp.headers["Location"])
    else:
        service_url = _get_service_url(resp.text)
        if not service_url:
            raise LoginError("Failed to determine IDS service URL from GenAI login page")
        ids_login_url = f"{IDS_BASE_URL}/authserver/login?service={service_url}"

    ids_resp = session.get(ids_login_url, timeout=30)
    ids_html = ids_resp.text

    lt = _collect_data(ids_html, "lt")
    execution = _collect_data(ids_html, "execution")
    salt = _collect_data(ids_html, "pwdEncryptSalt")

    if not salt:
        raise LoginError("Failed to get pwdEncryptSalt from IDS login page")
    if not execution:
        raise LoginError("Failed to get execution from IDS login page")

    form_data = {
        "username": student_id,
        "password": encrypt_password(password, salt),
        "lt": lt,
        "dllt": "generalLogin",
        "execution": execution,
        "_eventId": "submit",
        "rmShown": "1",
    }

    post_resp = session.post(ids_resp.url, data=form_data, allow_redirects=False, timeout=30)

    for _ in range(10):
        if not (300 <= post_resp.status_code < 400):
            break
        resolved = urljoin(post_resp.url, post_resp.headers.get("Location", ""))
        if "?token=" in resolved or "&token=" in resolved:
            token = parse_qs(urlparse(resolved).query).get("token", [None])[0]
            if token:
                return token
        post_resp = session.get(resolved, allow_redirects=False, timeout=30)

    if post_resp.status_code == 200 and any(keyword in post_resp.text for keyword in ("authError", "用户名或密码", "incorrectPassword")):
        raise LoginError("Username or password is incorrect")

    if "?token=" in post_resp.url or "&token=" in post_resp.url:
        token = parse_qs(urlparse(post_resp.url).query).get("token", [None])[0]
        if token:
            return token

    raise LoginError("Login flow completed but failed to extract token")
