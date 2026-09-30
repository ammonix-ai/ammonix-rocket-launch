#!/usr/bin/env python
"""Serve the control-room dashboard at a local address, with an LLM as analyst.

    python ControlRoom/serve.py       # http://localhost:8765

GET  /                      the committed dashboard.html
GET  /api/analyst/status    is the analyst's LLM reachable, and which model
POST /api/analyst           {"messages": [...]} -> the LLM's OpenAI-style event stream
GET  /api/launch/<seed>     one held-out launch replayed by the agent, for the page's selector

The analyst talks to any OpenAI-compatible chat server: llama.cpp's llama-server, vLLM,
Ollama, or a hosted endpoint. Where it is and which model it serves come from
../llm_config.toml [analyst]; the environment variables ANALYST_BASE_URL, ANALYST_MODEL,
ANALYST_LABEL, ANALYST_API_KEY and ANALYST_EXTRA_BODY (a JSON object merged into every
request) override it, and --llm-url / --model override those. The page only ever sends the
conversation; the server adds the model and sampling settings.

llama.cpp answers ANY model name with whatever model it has loaded. So when the chat server
runs on this machine, the server asks it which model is loaded and refuses to relay a chat
that another model would answer under the configured name.

The server binds to 127.0.0.1 unless --host says otherwise, and answers only requests
addressed to a loopback name unless --allow-host names the public host it is served under
(behind a reverse proxy). The chat relay uses the standard library only. The page has the
showcase stories and one held-out launch per failure type built in; any other of the 500
held-out launches the viewer picks is replayed here, by the same agent and code that build
the page (build_dashboard.py), in about 0.3 s.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

HERE = Path(__file__).resolve().parent
DOMAIN_DIR = HERE.parent
DASHBOARD = HERE / "dashboard.html"
SERVER_HOOK = "/*__SERVER__*/null"          # replaced in the page so it knows the API exists
SERVER_API = {"status": "api/analyst/status", "chat": "api/analyst", "launch": "api/launch/"}
ROLES = ("system", "user", "assistant")
MAX_MESSAGES = 40
MAX_CHARS = 120_000                         # whole conversation; the situation report is ~10 k
MAX_BODY = 600_000
LOOPBACK = ("127.0.0.1", "localhost", "::1")
KEEP_REPLAYS = 48                           # ~90 kB each


def _is_local(url: str) -> bool:
    return urllib.parse.urlsplit(url).hostname in LOOPBACK


def load_llm_config(path: Path = DOMAIN_DIR / "llm_config.toml") -> Dict[str, Any]:
    role = tomllib.loads(path.read_text(encoding="utf-8"))["analyst"]
    env = os.environ.get
    url = (env("ANALYST_BASE_URL") or role["base_url"]).rstrip("/")
    model = env("ANALYST_MODEL") or role["model"]
    extra = env("ANALYST_EXTRA_BODY")
    return {"url": url, "model": model,
            "label": env("ANALYST_LABEL") or role.get("label") or model,
            "api_key": env("ANALYST_API_KEY") or None,
            "extra_body": json.loads(extra) if extra else {},
            "temperature": role.get("temperature", 0.3), "max_tokens": role.get("max_tokens", 450),
            "enable_thinking": role.get("enable_thinking", False),
            "server": url if _is_local(url) else None}


def _headers(cfg: Dict[str, Any], **extra: str) -> Dict[str, str]:
    h = dict(extra)
    if cfg.get("api_key"):
        h["Authorization"] = f"Bearer {cfg['api_key']}"
    return h


def wrong_model(cfg: Dict[str, Any]) -> Optional[str]:
    """Why the answer would come from a model other than cfg["model"], or None. None also when
    the model server cannot be asked (the chat call then reports that fault) and for a hosted
    endpoint, which serves the model it is asked for."""
    server = cfg.get("server")
    if not server:
        return None
    try:
        with urllib.request.urlopen(urllib.request.Request(server + "/models", headers=_headers(cfg)),
                                    timeout=3) as resp:
            loaded = [str(m.get("id")) for m in json.load(resp).get("data", [])]
    except (OSError, ValueError, AttributeError):
        return None
    if cfg["model"] in loaded:
        return None
    return (f"The model server at {server} has {', '.join(loaded) or 'no model'} loaded, not "
            f"{cfg['model']}, so the answer would carry the wrong model name. Load {cfg['model']} "
            "there, or set the model in llm_config.toml")


def llm_status(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the chat server which models it serves. Listed is not proof that the model answers;
    a chat call that fails reports that separately."""
    out = {"available": False, "model": cfg["model"], "label": cfg.get("label") or cfg["model"],
           "where": "local" if _is_local(cfg["url"]) else "hosted"}
    if out["where"] == "local":
        out["url"] = cfg["url"]
    try:
        req = urllib.request.Request(cfg["url"] + "/models", headers=_headers(cfg))
        with urllib.request.urlopen(req, timeout=5) as resp:
            listed = [m.get("id") for m in json.load(resp).get("data", [])]
    except (OSError, ValueError) as exc:
        out["detail"] = (f"No LLM answers at {cfg['url']} ({exc.__class__.__name__}). Start a "
                         "local model server there (see README), or set base_url in llm_config.toml")
        return out
    if cfg["model"] not in listed:
        out["detail"] = (f"The LLM server does not offer {cfg['model']}; it lists "
                         f"{', '.join(map(str, listed[:8])) or 'no models'}. Set the model in "
                         "llm_config.toml or pass --model.")
        return out
    wrong = wrong_model(cfg)
    if wrong:
        out["detail"] = wrong
        return out
    out["available"] = True
    return out


def validate(body: Any) -> List[Dict[str, str]]:
    msgs = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(msgs, list) or not 1 <= len(msgs) <= MAX_MESSAGES:
        raise ValueError(f"messages must be a list of 1 to {MAX_MESSAGES} turns")
    clean = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ROLES or \
                not isinstance(m.get("content"), str) or not m["content"].strip():
            raise ValueError("every message needs a role (system, user or assistant) and text")
        clean.append({"role": m["role"], "content": m["content"]})
    if sum(len(m["content"]) for m in clean) > MAX_CHARS:
        raise ValueError("the conversation is too long")
    if clean[-1]["role"] != "user":
        raise ValueError("the last message must be the user's")
    return clean


class Replayer:
    """Replays held-out launches for the page's selector with the code that builds the page's
    own launches (build_dashboard.replay_launch and pack_launch), so a replay here equals one
    built into the page. Loads the agent on first use; one replay at a time, recent ones kept."""

    def __init__(self, domain_dir: Path = DOMAIN_DIR, keep: int = KEEP_REPLAYS):
        self.domain_dir, self.keep = domain_dir, keep
        self._lock = threading.Lock()
        self._done: "OrderedDict[int, bytes]" = OrderedDict()
        self._seeds: Optional[FrozenSet[int]] = None
        self._ctx: Optional[tuple] = None

    def seeds(self) -> FrozenSet[int]:
        """The 500 held-out launches (layer-4/split.json); only these may be replayed."""
        if self._seeds is None:
            split = json.loads((self.domain_dir / "layer-4" / "split.json").read_text(encoding="utf-8"))
            self._seeds = frozenset(int(s) for s in split["holdout_seeds"])
        return self._seeds

    def _context(self) -> tuple:
        if self._ctx is None:
            import build_dashboard as bd    # numpy and scikit-learn load only when needed
            agent = bd.ControlRoomAgent(self.domain_dir)
            lo = bd.np.array([c.lo for c in bd.V.CHANNELS])
            hi = bd.np.array([c.hi for c in bd.V.CHANNELS])
            self._ctx = (bd, agent, lo, hi)
        return self._ctx

    def warm(self) -> None:
        """Load the agent ahead of the first request; a failure is reported by that request."""
        try:
            with self._lock:
                self._context()
        except Exception:       # noqa: BLE001
            pass

    def launch(self, seed: int) -> bytes:
        """The launch as the page's JSON: telemetry, the agent's calls, events, snapshots."""
        with self._lock:
            if seed in self._done:
                self._done.move_to_end(seed)
                return self._done[seed]
            bd, agent, lo, hi = self._context()
            res = bd.replay_launch(agent, seed, None, random_anomalies=True)
            packed = bd.pack_launch(res, bd.case_meta(seed), lo, hi, agent)
            body = json.dumps(bd._clean(packed), separators=(",", ":")).encode("utf-8")
            self._done[seed] = body
            while len(self._done) > self.keep:
                self._done.popitem(last=False)
            return body


class Handler(BaseHTTPRequestHandler):
    server_version = "LaunchControl/1.0"
    cfg: Dict[str, Any] = {}
    page: bytes = b""
    replayer: Optional[Replayer] = None
    allowed_hosts: FrozenSet[str] = frozenset()
    chats: Optional[threading.BoundedSemaphore] = None

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.path.startswith("/api/analyst") and self.command == "POST":
            sys.stderr.write(f"  analyst: {fmt % args}\n")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Dict[str, Any]) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _allowed_host_header(self) -> bool:
        """Refuse DNS-rebinding requests: when bound to loopback, only loopback names and the
        hosts named by --allow-host are answered."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if host in self.allowed_hosts:
            return True
        if self.server.server_address[0] not in LOOPBACK:
            return True
        return host in LOOPBACK

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if not self._allowed_host_header():
            self._json(403, {"error": "unexpected Host header"})
        elif path in ("/", "/index.html", "/dashboard.html"):
            self._send(200, self.page, "text/html; charset=utf-8")
        elif path == "/api/analyst/status":
            self._json(200, llm_status(self.cfg))
        elif path.startswith("/api/launch/"):
            self._replay(path[len("/api/launch/"):])
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._json(404, {"error": "not found"})

    def _replay(self, name: str) -> None:
        rp = self.replayer
        if rp is None:
            return self._json(503, {"error": "this server does not replay launches"})
        if not re.fullmatch(r"[0-9]{1,12}", name):
            return self._json(400, {"error": "name a launch by its seed, a whole number"})
        seed = int(name)
        try:
            if seed not in rp.seeds():
                return self._json(404, {"error": f"launch {seed} is not one of the held-out launches"})
            body = rp.launch(seed)
        except FileNotFoundError as exc:
            return self._json(503, {"error": f"cannot replay launches: {exc.filename or exc} is "
                                             "missing (python run_factory.py makes it)"})
        except Exception as exc:        # noqa: BLE001  the page shows the reason
            sys.stderr.write(f"  replay of launch {seed} failed: {exc!r}\n")
            return self._json(500, {"error": f"the replay failed ({exc.__class__.__name__}: {exc})"})
        self._send(200, body, "application/json")

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/analyst":
            return self._json(404, {"error": "not found"})
        if not self._allowed_host_header():
            return self._json(403, {"error": "unexpected Host header"})
        # JSON only: a cross-site form post cannot send it without a CORS preflight.
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            return self._json(415, {"error": "send application/json"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_BODY:
                raise ValueError("empty or oversized request")
            messages = validate(json.loads(self.rfile.read(n)))
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        if self.chats is not None and not self.chats.acquire(blocking=False):
            return self._json(429, {"error": "the analyst is busy with other viewers; ask again in a moment"})
        try:
            self._relay(messages)
        finally:
            if self.chats is not None:
                self.chats.release()

    def _relay(self, messages: List[Dict[str, str]]) -> None:
        cfg = self.cfg
        wrong = wrong_model(cfg)                # the loaded model can change while the page is open
        if wrong:
            return self._json(502, {"error": wrong})
        # enable_thinking is read by some servers, chat_template_kwargs by vLLM and llama.cpp;
        # each ignores the other's field. extra_body carries whatever a hosted endpoint needs.
        payload = {"model": cfg["model"], "messages": messages, "stream": True,
                   "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"],
                   "enable_thinking": cfg["enable_thinking"],
                   "chat_template_kwargs": {"enable_thinking": cfg["enable_thinking"]}}
        payload.update(cfg.get("extra_body") or {})
        req = urllib.request.Request(cfg["url"] + "/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers=_headers(cfg, **{"Content-Type": "application/json",
                                                              "Accept": "text/event-stream"}))
        try:
            upstream = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            return self._json(502, {"error": f"the LLM server answered {exc.code}: {detail}"})
        except OSError as exc:
            return self._json(502, {"error": f"no LLM server at {cfg['url']} ({exc})"})
        with upstream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while chunk := upstream.read1(8192):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except ConnectionError:     # also Windows' ConnectionAbortedError
                pass        # the viewer pressed Stop; closing upstream ends the generation


def make_server(cfg: Dict[str, Any], host: str = "127.0.0.1", port: int = 8765,
                page: Path = DASHBOARD, replayer: Optional[Replayer] = None,
                allow_hosts: tuple = (), max_chats: int = 0) -> ThreadingHTTPServer:
    html = page.read_text(encoding="utf-8").replace(SERVER_HOOK, json.dumps(SERVER_API), 1)
    handler = type("BoundHandler", (Handler,), {
        "cfg": cfg, "page": html.encode("utf-8"), "replayer": replayer,
        "allowed_hosts": frozenset(h.lower() for h in allow_hosts),
        "chats": threading.BoundedSemaphore(max_chats) if max_chats > 0 else None})
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to bind; 0.0.0.0 makes it reachable from your network")
    ap.add_argument("--llm-url", help="OpenAI-compatible base URL (default: llm_config.toml)")
    ap.add_argument("--model", help="model name the LLM server serves (default: llm_config.toml)")
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                    help="also answer requests for this host name (when served behind a reverse proxy)")
    ap.add_argument("--max-chats", type=int, default=0,
                    help="at most this many analyst answers at once; 0 = no limit")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args()
    if not DASHBOARD.exists():
        sys.exit(f"{DASHBOARD} is missing: run ControlRoom/build_dashboard.py first")
    cfg = load_llm_config()
    if args.llm_url:
        cfg["url"] = args.llm_url.rstrip("/")
        cfg["server"] = cfg["url"] if _is_local(cfg["url"]) else None
    if args.model:
        cfg["model"] = cfg["label"] = args.model
    replayer = Replayer()
    httpd = make_server(cfg, args.host, args.port, replayer=replayer,
                        allow_hosts=tuple(args.allow_host), max_chats=args.max_chats)
    shown = "localhost" if args.host in LOOPBACK + ("0.0.0.0",) else args.host
    url = f"http://{shown}:{args.port}/"
    status = llm_status(cfg)
    print(f"Launch Control Agent: {url}")
    print(f"Analyst: {status['label']} via {cfg['url']}: "
          + ("ready" if status["available"] else status["detail"]))
    if (DOMAIN_DIR / "layer-4" / "model.pkl").exists():
        print("Held-out launches: all 500 can be replayed")
        threading.Thread(target=replayer.warm, daemon=True).start()
    else:
        print("Held-out launches: only the page's built-in examples play; layer-4/model.pkl is "
              "missing (python run_factory.py makes it)")
    print("Ctrl+C stops the server.", flush=True)     # also when a launcher reads a pipe
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
