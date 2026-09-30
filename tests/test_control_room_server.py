"""The local dashboard server: serves the page, relays the analyst chat to the LLM and
replays held-out launches for the page's selector.

A stand-in for the LLM server (OpenAI-compatible /v1/models and a streamed
/v1/chat/completions) runs in a thread, and a stand-in replayer answers for the agent, so
no model or GPU is needed. One test replays a real launch when layer-4/model.pkl exists.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import serve

CHUNKS = ["The left ", "booster ", "runs low."]
NOTHING_LISTENS = "http://127.0.0.1:9/v1"          # the discard port: connection refused


class FakeProxy(BaseHTTPRequestHandler):
    """The LLM server: lists its models and streams an answer; records what it was sent."""
    received: list = []
    headers_seen: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        FakeProxy.headers_seen.append(dict(self.headers))
        body = json.dumps({"object": "list", "data": [{"id": "qwen36-27b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        FakeProxy.headers_seen.append(dict(self.headers))
        FakeProxy.received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for text in CHUNKS:
            chunk = {"choices": [{"index": 0, "delta": {"content": text}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


class FakeReplayer:
    """Answers for serve.Replayer: one held-out launch, or the error it is told to raise."""

    def __init__(self, fail: Exception | None = None):
        self.fail, self.asked = fail, []

    def seeds(self):
        return frozenset({202600003})

    def launch(self, seed: int) -> bytes:
        self.asked.append(seed)
        if self.fail:
            raise self.fail
        return json.dumps({"id": f"case-{seed}"}).encode()


class FakeModelServer(BaseHTTPRequestHandler):
    """The model server behind the proxy: lists only the model it has loaded, like llama.cpp."""
    loaded: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list",
                           "data": [{"id": m} for m in FakeModelServer.loaded]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _run(server: ThreadingHTTPServer) -> str:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture()
def page(tmp_path):
    p = tmp_path / "dashboard.html"
    p.write_text("<title>t</title><script>const SERVER = /*__SERVER__*/null;</script>",
                 encoding="utf-8")
    return p


@pytest.fixture()
def proxy():
    FakeProxy.received, FakeProxy.headers_seen = [], []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeProxy)
    yield _run(srv) + "/v1"
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def model_server():
    FakeModelServer.loaded = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeModelServer)
    yield _run(srv) + "/v1"
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def start(page):
    servers = []

    def _start(llm_url: str, model: str = "qwen36-27b", server=None, replayer=None,
               api_key=None, extra_body=None, **kw) -> str:
        cfg = {"url": llm_url, "model": model, "label": model, "temperature": 0.3,
               "max_tokens": 450, "enable_thinking": False, "server": server,
               "api_key": api_key, "extra_body": extra_body or {}}
        srv = serve.make_server(cfg, "127.0.0.1", 0, page=page, replayer=replayer, **kw)
        servers.append(srv)
        return _run(srv)

    yield _start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _post(url: str, body, ctype: str = "application/json"):
    req = urllib.request.Request(url + "/api/analyst", data=json.dumps(body).encode(),
                                 headers={"Content-Type": ctype})
    return urllib.request.urlopen(req, timeout=10)


def _get(url: str):
    return urllib.request.urlopen(url, timeout=10)


def test_page_is_served_with_the_analyst_api(start, proxy):
    url = start(proxy)
    html = _get(url + "/").read().decode()
    assert '"chat": "api/analyst"' in html and "__SERVER__" not in html
    status = json.load(_get(url + "/api/analyst/status"))
    assert status["available"] and status["model"] == "qwen36-27b"


def test_a_held_out_launch_is_replayed_on_request(start):
    rp = FakeReplayer()
    url = start(NOTHING_LISTENS, replayer=rp)
    assert '"launch": "api/launch/"' in _get(url + "/").read().decode()
    assert json.load(_get(url + "/api/launch/202600003")) == {"id": "case-202600003"}
    assert rp.asked == [202600003]


@pytest.mark.parametrize("name, code", [("202600004", 404), ("abc", 400), ("-3", 400),
                                        ("2e8", 400), ("", 400), ("1234567890123", 400)])
def test_only_held_out_launches_are_replayed(start, name, code):
    rp = FakeReplayer()
    url = start(NOTHING_LISTENS, replayer=rp)
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(url + "/api/launch/" + name)
    assert err.value.code == code and rp.asked == []


def test_replays_without_the_model_say_what_is_missing(start):
    url = start(NOTHING_LISTENS, replayer=FakeReplayer(
        FileNotFoundError(2, "No such file or directory", "layer-4/model.pkl")))
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(url + "/api/launch/202600003")
    assert err.value.code == 503 and "model.pkl" in json.load(err.value)["error"]
    url = start(NOTHING_LISTENS)                      # a server without a replayer
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(url + "/api/launch/202600003")
    assert err.value.code == 503


@pytest.mark.skipif(not (serve.DOMAIN_DIR / "layer-4" / "model.pkl").exists(),
                    reason="layer-4/model.pkl is not committed; run_factory.py makes it")
def test_a_replay_equals_the_launch_built_into_the_page():
    html = serve.DASHBOARD.read_text(encoding="utf-8")
    data = json.loads(re.search(r"const DATA = (\{.*?\});\s*\nconst SERVER", html, re.S).group(1))
    built = next(L for L in data["launches"] if L["kind"] == "test")
    rp = serve.Replayer()
    assert built["seed"] in rp.seeds() and len(rp.seeds()) == 500
    assert json.loads(rp.launch(built["seed"])) == built
    assert rp.launch(built["seed"]) is rp.launch(built["seed"])       # kept, not replayed again


def test_chat_is_relayed_with_the_configured_model(start, proxy):
    url = start(proxy)
    messages = [{"role": "system", "content": "SITUATION REPORT"},
                {"role": "user", "content": "Why this call?"}]
    stream = _post(url, {"messages": messages}).read().decode()
    assert "".join(json.loads(line[5:])["choices"][0]["delta"]["content"]
                   for line in stream.splitlines()
                   if line.startswith("data:") and "[DONE]" not in line) == "".join(CHUNKS)
    sent = FakeProxy.received[-1]
    assert sent["messages"] == messages
    assert sent["model"] == "qwen36-27b" and sent["stream"] is True
    assert sent["enable_thinking"] is False and sent["max_tokens"] == 450
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_status_says_what_is_missing(start, proxy):
    status = json.load(_get(start(proxy, model="another-model") + "/api/analyst/status"))
    assert not status["available"] and "does not offer another-model" in status["detail"]
    status = json.load(_get(start(NOTHING_LISTENS) + "/api/analyst/status"))
    assert not status["available"] and "No LLM answers" in status["detail"]


@pytest.mark.parametrize("body, ctype, code", [
    ({"messages": [{"role": "user", "content": "hi"}]}, "text/plain", 415),
    ({"messages": [{"role": "tool", "content": "x"}]}, "application/json", 400),
    ({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]},
     "application/json", 400),
    ({"messages": []}, "application/json", 400),
    ({"prompt": "hi"}, "application/json", 400),
])
def test_bad_requests_never_reach_the_llm(start, proxy, body, ctype, code):
    url = start(proxy)
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(url, body, ctype)
    assert err.value.code == code
    assert FakeProxy.received == []


def test_unreachable_llm_is_a_clear_error(start):
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(start(NOTHING_LISTENS), {"messages": [{"role": "user", "content": "hi"}]})
    assert err.value.code == 502
    assert "no LLM server" in json.load(err.value)["error"]


def test_foreign_host_header_is_refused(start, proxy):
    req = urllib.request.Request(start(proxy) + "/", headers={"Host": "attacker.example"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=10)
    assert err.value.code == 403


def test_an_allowed_public_host_is_answered(start, proxy):
    url = start(proxy, allow_hosts=("Rocket.Example.org",))
    req = urllib.request.Request(url + "/", headers={"Host": "rocket.example.org"})
    assert urllib.request.urlopen(req, timeout=10).status == 200
    req = urllib.request.Request(url + "/", headers={"Host": "attacker.example"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=10)
    assert err.value.code == 403


def test_another_loaded_model_is_refused_not_mislabelled(start, proxy, model_server):
    # llama.cpp answers any model name with the model it has loaded, so the name is not enough
    FakeModelServer.loaded = ["ornith-9b"]
    url = start(proxy, server=model_server)
    status = json.load(_get(url + "/api/analyst/status"))
    assert not status["available"] and "has ornith-9b loaded, not qwen36-27b" in status["detail"]
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(url, {"messages": [{"role": "user", "content": "hi"}]})
    assert err.value.code == 502 and "ornith-9b" in json.load(err.value)["error"]
    assert FakeProxy.received == []
    FakeModelServer.loaded = ["qwen36-27b"]
    assert json.load(_get(url + "/api/analyst/status"))["available"]
    with _post(url, {"messages": [{"role": "user", "content": "hi"}]}) as resp:
        assert resp.status == 200 and b"[DONE]" in resp.read()


def test_an_unreachable_model_server_is_left_to_the_chat_call(start, proxy):
    url = start(proxy, server=NOTHING_LISTENS)
    assert json.load(_get(url + "/api/analyst/status"))["available"]


def test_a_hosted_endpoint_gets_the_key_and_extra_body(start, proxy):
    url = start(proxy, api_key="test-key", extra_body={"reasoning": {"enabled": False}})
    with _post(url, {"messages": [{"role": "user", "content": "hi"}]}) as resp:
        assert resp.status == 200 and b"[DONE]" in resp.read()
    sent = FakeProxy.received[-1]
    assert sent["reasoning"] == {"enabled": False} and sent["model"] == "qwen36-27b"
    assert all(h.get("Authorization") == "Bearer test-key" for h in FakeProxy.headers_seen)


def test_busy_analyst_answers_429(page, proxy):
    cfg = {"url": proxy, "model": "qwen36-27b", "label": "qwen36-27b", "temperature": 0.3,
           "max_tokens": 450, "enable_thinking": False, "server": None, "api_key": None,
           "extra_body": {}}
    srv = serve.make_server(cfg, "127.0.0.1", 0, page=page, max_chats=1)
    url, slots = _run(srv), srv.RequestHandlerClass.chats
    try:
        assert slots.acquire(blocking=False)            # another viewer holds the only slot
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(url, {"messages": [{"role": "user", "content": "hi"}]})
        assert err.value.code == 429 and FakeProxy.received == []
        slots.release()
        with _post(url, {"messages": [{"role": "user", "content": "hi"}]}) as resp:
            assert resp.status == 200 and b"[DONE]" in resp.read()
    finally:
        srv.shutdown()
        srv.server_close()


def test_llm_config_is_a_local_server_without_thinking(monkeypatch):
    for k in ("ANALYST_BASE_URL", "ANALYST_MODEL", "ANALYST_LABEL", "ANALYST_API_KEY",
              "ANALYST_EXTRA_BODY"):
        monkeypatch.delenv(k, raising=False)
    cfg = serve.load_llm_config()
    assert cfg["enable_thinking"] is False and cfg["api_key"] is None
    assert cfg["url"] == "http://127.0.0.1:8001/v1"             # not localhost: see llm_config.toml
    assert cfg["server"] == cfg["url"]                          # a local server is asked its model
    assert cfg["label"] != cfg["model"]


def test_environment_points_the_analyst_at_a_hosted_endpoint(monkeypatch):
    monkeypatch.setenv("ANALYST_BASE_URL", "https://llm.example.org/api/v1/")
    monkeypatch.setenv("ANALYST_MODEL", "vendor/model")
    monkeypatch.setenv("ANALYST_API_KEY", "k")
    monkeypatch.setenv("ANALYST_EXTRA_BODY", '{"provider": {"only": ["x"]}}')
    cfg = serve.load_llm_config()
    assert cfg["url"] == "https://llm.example.org/api/v1" and cfg["model"] == "vendor/model"
    assert cfg["server"] is None                                # a hosted endpoint is not asked
    assert cfg["api_key"] == "k" and cfg["extra_body"] == {"provider": {"only": ["x"]}}
