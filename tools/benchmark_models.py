import argparse
import json
import time

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark all exposed models via OpenAI-compatible streaming API")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key or GenAI token to send as Bearer auth")
    parser.add_argument("--prompt", default="请用中文简要介绍上海科技大学，并尽量输出约300字。", help="Prompt used for each benchmark")
    parser.add_argument("--max-tokens", type=int, default=512, help="max_tokens for each request")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout in seconds")
    parser.add_argument("--models", nargs="*", help="Optional model id list; defaults to /models")
    return parser.parse_args()


def build_headers(api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def fetch_models(base_url, headers, timeout):
    response = requests.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return [model["id"] for model in payload.get("data", []) if isinstance(model, dict) and model.get("id")]


def count_tokens(text):
    # Prefer tiktoken when available; fall back to a rough multilingual estimate.
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def iter_sse_data(response):
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        yield line[5:].strip()


def benchmark_model(base_url, headers, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }

    start_time = time.perf_counter()
    first_token_time = None
    content_parts = []

    with requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for data in iter_sse_data(response):
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])

            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            emitted = content or reasoning
            if emitted and first_token_time is None:
                first_token_time = time.perf_counter()
            if content:
                content_parts.append(content)

    end_time = time.perf_counter()
    content_text = "".join(content_parts)
    output_tokens = count_tokens(content_text)
    first_token_delay = None if first_token_time is None else first_token_time - start_time
    generation_time = None if first_token_time is None else max(end_time - first_token_time, 0.001)
    tokens_per_second = output_tokens / generation_time if generation_time else 0.0

    return {
        "model": model,
        "first_token_delay": first_token_delay,
        "tokens_per_second": tokens_per_second,
        "output_tokens": output_tokens,
        "total_seconds": end_time - start_time,
    }


def print_table(results):
    headers = ["model", "first_token_s", "tokens_per_s", "output_tokens", "total_s", "status"]
    rows = []
    for result in results:
        if result.get("error"):
            rows.append([result["model"], "-", "-", "-", "-", f"ERROR: {result['error']}"])
            continue
        rows.append([
            result["model"],
            f"{result['first_token_delay']:.3f}" if result["first_token_delay"] is not None else "-",
            f"{result['tokens_per_second']:.2f}",
            str(result["output_tokens"]),
            f"{result['total_seconds']:.3f}",
            "ok",
        ])

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def format_result_line(result):
    if result.get("error"):
        return f"{result['model']}: ERROR: {result['error']}"

    first_token = f"{result['first_token_delay']:.3f}s" if result["first_token_delay"] is not None else "-"
    return (
        f"{result['model']}: first_token={first_token}, "
        f"tokens_per_s={result['tokens_per_second']:.2f}, "
        f"output_tokens={result['output_tokens']}, "
        f"total={result['total_seconds']:.3f}s"
    )


def main():
    args = parse_args()
    headers = build_headers(args.api_key)
    models = args.models or fetch_models(args.base_url, headers, args.timeout)
    if not models:
        raise SystemExit("No models found")

    results = []
    for model in models:
        print(f"Benchmarking {model}...", flush=True)
        try:
            result = benchmark_model(args.base_url, headers, model, args.prompt, args.max_tokens, args.timeout)
        except Exception as exc:
            result = {"model": model, "error": str(exc)}
        results.append(result)
        print(format_result_line(result), flush=True)

    print()
    print("Summary")
    print_table(results)


if __name__ == "__main__":
    main()
