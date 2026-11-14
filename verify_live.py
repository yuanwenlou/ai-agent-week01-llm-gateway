"""Exercise both real upstreams through a running local gateway."""

import argparse
import json

import httpx


CHAT_PATH = "/v1/chat/completions"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_model(client: httpx.Client, model: str) -> None:
    payload = {"model": model, "messages": [
        {"role": "user", "content": 'Reply with one JSON object containing a string field called "answer".'},
    ]}
    ordinary = client.post(CHAT_PATH, json=payload)
    ordinary.raise_for_status()
    text = ordinary.json()["choices"][0]["message"]["content"]
    require(bool(text.strip()), f"{model}: empty ordinary reply")
    structured = client.post(CHAT_PATH, json={
        **payload,
        "response_format": {"type": "json_object"},
        "gateway_prompt": {"id": "reviewer", "version": 2, "variables": {"language": "English"}},
    })
    structured.raise_for_status()
    require(isinstance(json.loads(structured.json()["choices"][0]["message"]["content"]), dict),
            f"{model}: expected a JSON object")

    pieces = []
    done = False
    with client.stream("POST", CHAT_PATH, json={**payload, "stream": True}) as response:
        response.raise_for_status()
        require(response.headers["content-type"].startswith("text/event-stream"), f"{model}: expected SSE")
        stream_id = response.headers["x-request-id"]
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(f"{model}: stream failed with {chunk['error']['code']}")
            text = chunk["choices"][0]["delta"].get("content")
            if text:
                pieces.append(text)
    require(bool(pieces) and done, f"{model}: stream had no text or did not finish")

    for kind, request_id in (
        ("ordinary", ordinary.headers["x-request-id"]),
        ("structured", structured.headers["x-request-id"]),
        ("stream", stream_id),
    ):
        response = client.get(f"/v1/usage/{request_id}")
        response.raise_for_status()
        record = response.json()
        require(record["status"] == "success" and record["latency_ms"] >= 0, f"{model}: invalid call record")
        if kind == "stream":
            require(record["first_token_ms"] is not None, f"{model}: missing TTFT")
        if kind == "structured":
            require(record["prompt_id"] == "reviewer" and record["prompt_version"] == 2,
                    f"{model}: incorrect prompt version")
        usage = record["usage"]
        require(all(usage[field] is not None for field in ("input_tokens", "output_tokens")),
                f"{model}: missing upstream token counts")
        print(f"{model} {kind} {json.dumps(record, ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        for model in ("pro", "flash"):
            check_model(client, model)


if __name__ == "__main__":
    main()
