"""The knowledge universe builder: action states, colors, byte layout, reproducibility.

Synthetic scores and a stand-in call policy, so no factory run or model is needed.
"""

from __future__ import annotations

import base64
import gzip
import json
import math
import re
from collections import Counter

import numpy as np
import pytest

import build_dashboard as bd
import build_universe as bu
from launch_sim.anomalies import FAILURE_KEYS

K = len(FAILURE_KEYS)


def _scores(n: int = 12, t: int = 40, seed: int = 0) -> dict:
    """Launch 0 stays nominal; launch i drifts into failure mode i % K in its second half."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 0.05, size=(n, t, K)).astype(np.float32)
    y, manifest = np.zeros((n, K), int), np.full((n, K), np.inf)
    for i in range(1, n):
        k = i % K
        p[i, t // 2:, k] = np.linspace(0.2, 0.99, t - t // 2)
        y[i, k], manifest[i, k] = 1, -60.0 + 0.5 * (t // 2)
    return {"p": p, "ticks": np.round(-60.0 + 0.5 * np.arange(t), 3), "keys": np.array(FAILURE_KEYS),
            "thresholds": np.full(K, 0.9), "seeds": np.arange(n) + 1000,
            "holdout": np.arange(n) % 5 == 0, "y": y, "manifest": manifest}


def _policy(stops: dict):
    """Stand-in for the agent: GO, or from tick s on the given stop call."""
    state = {"i": -1}

    def calls_of(p_launch):
        state["i"] += 1
        call, at = stops.get(state["i"], (None, None))
        return ["GO"] * len(p_launch) if call is None else \
            ["GO"] * at + [call] * (len(p_launch) - at)
    return calls_of


def test_most_probable_action_is_nominal_below_one_half():
    p = np.zeros((4, K))
    p[1, 2], p[2, 0], p[3, 5] = 0.6, 0.5, 0.49
    assert bu.most_probable_action(p).tolist() == [0, 3, 1, 0]


def test_a_launch_ends_where_the_agent_stops_it():
    p = np.zeros((3, 10, K), np.float32)
    probs, calls, start, count, stop = bu.action_states(p, _policy({1: ("HOLD", 3),
                                                                    2: ("ABORT", 6)}))
    assert count.tolist() == [10, 4, 7] and start.tolist() == [0, 10, 14]
    assert stop == ["", "HOLD", "ABORT"] and len(probs) == len(calls) == 21
    names = [bu.CALLS[c] for c in calls]
    assert names[13] == "HOLD" and names[20] == "ABORT" and names.count("HOLD") == 1


def test_landmarks_keep_every_class():
    classes = np.zeros(5000, int)
    classes[:7], classes[7:9] = 3, 11
    lm = bu.pick_landmarks(classes, 60, seed=1)
    assert len(lm) == 60 and {0, 3, 11} <= set(classes[lm].tolist())
    assert (classes[lm] == 3).sum() == 7 and (classes[lm] == 11).sum() == 2


def test_a_landmark_is_placed_exactly_the_rest_among_them():
    rng = np.random.default_rng(3)
    vectors = rng.uniform(0, 1, size=(300, K))
    classes = bu.most_probable_action(vectors)
    y, info = bu.embed(vectors, classes, n_landmarks=120, perplexity=20, neighbours=5)
    lm = bu.pick_landmarks(classes, 120, bu.SEED)
    assert info["landmarks"] == 120 and y.shape == (300, 3) and np.isfinite(y).all()
    lo, hi = y[lm].min(axis=0), y[lm].max(axis=0)
    assert ((y >= lo - 1e-9) & (y <= hi + 1e-9)).all()   # weighted means stay inside


def test_the_universe_round_trips_and_rebuilds_the_same(tmp_path):
    scores = _scores()
    meta, blob = bu.build(scores, _policy({3: ("HOLD", 25)}), log=lambda m: None)
    bu.write(tmp_path, meta, blob)
    first = (tmp_path / "universe.bin.gz").read_bytes()
    assert gzip.decompress(first) == blob

    def arr(name):
        spec = meta["layout"][name]
        a = np.frombuffer(blob, dtype=np.dtype(spec["dtype"]), count=int(np.prod(spec["shape"])),
                          offset=spec["offset"])
        assert spec["offset"] % a.itemsize == 0
        return a.reshape(spec["shape"])

    n, u = meta["n_states"], meta["n_distinct"]
    assert n == 12 * 40 - (40 - 26) and meta["launches"]["stop"][3] == "HOLD"
    assert arr("vec_pos").shape == (u, 3) and np.abs(arr("vec_pos")).max() <= 1.0
    assert arr("state_vec").max() == u - 1 and len(arr("state_call")) == n
    assert arr("vec_prob").max() <= bu.P_SCALE
    assert sum(meta["counts"]["class"]) == n and sum(meta["counts"]["call"]) == n
    assert meta["classes"][0]["mode"] == "NOMINAL" and len(meta["classes"]) == K + 1
    assert meta["launches"]["truth"][0] == [] and meta["launches"]["truth"][1] == [[1, -50.0]]
    json.loads((tmp_path / "universe.json").read_text(encoding="utf-8"),
               parse_constant=lambda c: pytest.fail(f"{c} is not valid JSON for the page"))

    meta2, blob2 = bu.build(scores, _policy({3: ("HOLD", 25)}), log=lambda m: None)
    bu.write(tmp_path, meta2, blob2)
    assert (tmp_path / "universe.bin.gz").read_bytes() == first      # same bytes every build


def test_the_dashboard_embeds_the_universe_once_it_is_built(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "HERE", tmp_path)
    assert bd.universe_payload() is None
    meta, blob = bu.build(_scores(n=4, t=12), _policy({}), log=lambda m: None)
    bu.write(tmp_path, meta, blob)
    payload = bd.universe_payload()
    assert payload["meta"]["n_states"] == meta["n_states"] == 48
    assert gzip.decompress(base64.b64decode(payload["gz"])) == blob


# gitleaks' generic-api-key rule (default config), which the push-run secret scan applies: a
# name containing key, api, auth, token, secret... then a value of 10+ characters with entropy
# of at least 3.5. Values of letters, "_", "." and "-" only are allowed. Generated data tripped
# it twice: a JSON field named key that held a failure-mode name with a digit in it (LH2...).
# Two base64 strings side by side can trip it too. (Quoting the flagged text here would trip it.)
GENERIC_API_KEY = re.compile(
    r"(?i)[\w.-]{0,50}?(?:access|auth|(?-i:[Aa]pi|API)|credential|creds|key|passw(?:or)?d|secret|token)"
    r"(?:[ \t\w.-]{0,20})[\s'\"]{0,3}(?:=|>|:{1,3}=|\|\||:|=>|\?=|,)[\x60'\"\s=]{0,5}"
    r"([\w.=-]{10,150}|[a-z0-9][a-z0-9+/]{11,}={0,3})(?:[\x60'\"\s;]|\\[nr]|$)")


def _entropy(s: str) -> float:
    return -sum(n / len(s) * math.log2(n / len(s)) for n in Counter(s).values())


def test_generated_data_gives_the_secret_scanner_nothing_to_flag():
    page = (bd.HERE / "dashboard.html").read_text(encoding="utf-8")
    data_line = next(line for line in page.splitlines() if line.startswith("const DATA = "))
    for name, text in (("dashboard.html data", data_line),
                       ("universe.json", (bd.HERE / "universe.json").read_text(encoding="utf-8"))):
        hits = [m.group(0)[:80] for m in GENERIC_API_KEY.finditer(text)
                if _entropy(m.group(1)) >= 3.5 and not re.fullmatch(r"[a-zA-Z_.-]+", m.group(1))]
        assert not hits, (name, hits[:5])


def test_the_committed_dashboard_is_the_template_plus_data():
    """Page edits go into dashboard_template.html; build_dashboard.py makes dashboard.html."""
    tpl = (bd.HERE / "dashboard_template.html").read_text(encoding="utf-8")
    page = (bd.HERE / "dashboard.html").read_text(encoding="utf-8")
    pre, post = tpl.split("/*__DATA__*/null")
    body = page[page.index("<body>\n") + len("<body>\n"):page.rindex("\n</body>")]
    assert body.startswith(pre) and body.endswith(post), "rebuild with ControlRoom/build_dashboard.py"
    data = json.loads(body[len(pre):len(body) - len(post)])
    built = json.loads((bd.HERE / "universe.json").read_text(encoding="utf-8"))
    assert data["universe"]["meta"]["n_states"] == built["n_states"]
    assert base64.b64decode(data["universe"]["gz"]) == (bd.HERE / "universe.bin.gz").read_bytes()
