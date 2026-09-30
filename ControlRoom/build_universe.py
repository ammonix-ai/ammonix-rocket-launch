#!/usr/bin/env python
"""Build the knowledge universe: every action state of every launch, as a 3D star map.

An action state is what the agent held at one 0.5 s check of one launch: one probability
per failure mode (its action space), the call it made, and the time. A launch is
the chain of its action states, from H0 - 60 s until the window ends or the agent stops the
launch (HOLD in the countdown, ABORT during engine start).

    python ControlRoom/build_universe.py

Input: layer-4/streaming_scores.npz from run_factory.py: out-of-fold scores for the 2,500
work launches (each scored by fold models that never saw it) and the final models' scores
for the 500 holdout launches. The calls come from ControlRoomAgent.decide, the agent's own
policy (debounce, exclusion rule, latching), so each star carries the call the agent made.

Positions: a 3D t-SNE of the probability vectors, as in the other Ammonix universes: action
space, never feature space. t-SNE does not scale to a million points, so it runs on landmark
vectors, and every state is placed by inverse-distance weighting of its nearest landmarks in
action space. States with the same vector are spread in a small ball that grows with their
number, so a crowd stays visible (the page draws it; the spread carries no meaning).

Color: the most probable action: the failure mode with the highest probability when it is
at least 0.5, else nominal (the taxonomy's twelfth class).

Output, served by serve.py for the dashboard's Knowledge universe tab:
  ControlRoom/universe.json    counts, classes, launches, t-SNE settings, byte layout
  ControlRoom/universe.bin.gz  per distinct vector: float32 x, y, z (within -1..1), uint8 p per mode,
                               uint8 class; per state: its vector's index and uint8 call.
                               The page adds the jitter, so it is not stored a million times.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
DOMAIN_DIR = HERE.parent
sys.path[:0] = [str(HERE), str(DOMAIN_DIR)]

from agent import CALLS  # noqa: E402
from launch_sim.anomalies import BY_KEY  # noqa: E402

P_SCALE = 250            # probabilities are stored as round(p * 250), as in the dashboard
MOST_PROBABLE = 0.5      # a failure mode is the most probable action from this probability on
STOP_CALLS = ("HOLD", "ABORT")
N_LANDMARKS = 24_000
PERPLEXITY = 50.0
NEIGHBOURS = 8
JITTER = {"base": 0.004, "max": 0.06,  # page-side spread of the states that share a vector:
          "rule": "sd = base * cbrt(states on the vector), at most max (of the half-width)"}
SEED = 42


def most_probable_action(p: np.ndarray) -> np.ndarray:
    """Class per row: 0 = nominal, j + 1 = failure mode j (its probability is the highest and
    at least 0.5, i.e. more probable than 'no failure' when that is 1 - max p)."""
    p = np.asarray(p, dtype=np.float64)
    j = np.argmax(p, axis=1)
    top = p[np.arange(len(p)), j]
    return np.where(top >= MOST_PROBABLE, j + 1, 0).astype(np.uint8)


def action_states(p: np.ndarray, calls_of: Callable[[np.ndarray], Sequence[str]]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Cut every launch where the agent stopped it and stack the states.

    p: (launches, ticks, modes). Returns probabilities (N, modes), call indices (N,), and per
    launch the first state's row, the number of states and the stop call ('' if none)."""
    n, n_t, _ = p.shape
    probs, calls, start, count, stop = [], [], np.zeros(n, np.int64), np.zeros(n, np.int64), []
    row = 0
    for i in range(n):
        seq = list(calls_of(p[i]))
        end = next((t + 1 for t, c in enumerate(seq) if c in STOP_CALLS), n_t)
        stop.append(seq[end - 1] if seq[end - 1] in STOP_CALLS else "")
        probs.append(p[i, :end])
        calls.append([CALLS.index(c) for c in seq[:end]])
        start[i], count[i] = row, end
        row += end
    return (np.concatenate(probs).astype(np.float32), np.concatenate(calls).astype(np.uint8),
            start, count, stop)


def pick_landmarks(classes: np.ndarray, n_landmarks: int, seed: int) -> np.ndarray:
    """Indices of landmark vectors: an equal quota per class, the rest drawn at random, so
    the rare failure regions are mapped as finely as the huge nominal one."""
    n = len(classes)
    if n <= n_landmarks:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    labels = np.unique(classes)
    quota = n_landmarks // len(labels)
    chosen = [rng.permutation(np.flatnonzero(classes == c))[:quota] for c in labels]
    taken = np.zeros(n, bool)
    taken[np.concatenate(chosen)] = True
    rest = rng.permutation(np.flatnonzero(~taken))[:n_landmarks - int(taken.sum())]
    taken[rest] = True
    return np.flatnonzero(taken)


def embed(vectors: np.ndarray, classes: np.ndarray, n_landmarks: int = N_LANDMARKS,
          perplexity: float = PERPLEXITY, neighbours: int = NEIGHBOURS,
          seed: int = SEED) -> Tuple[np.ndarray, Dict[str, Any]]:
    """3D coordinates for unique probability vectors: t-SNE on landmarks, then every vector
    at the inverse-distance weighted mean of its nearest landmarks (exact for a landmark)."""
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors

    lm = pick_landmarks(classes, n_landmarks, seed)
    x_lm = vectors[lm]
    perp = float(min(perplexity, max(2.0, (len(lm) - 1) / 3.0 - 1.0)))
    tsne = TSNE(n_components=3, perplexity=perp, init="pca", learning_rate="auto",
                max_iter=1000, random_state=seed, n_jobs=-1)
    y_lm = tsne.fit_transform(x_lm)
    k = min(neighbours, len(lm))
    dist, idx = NearestNeighbors(n_neighbors=k).fit(x_lm).kneighbors(vectors)
    w = 1.0 / np.maximum(dist, 1e-9)
    y = (w[:, :, None] * y_lm[idx]).sum(axis=1) / w.sum(axis=1)[:, None]
    exact = dist[:, 0] < 1e-12
    y[exact] = y_lm[idx[exact, 0]]
    info = {"landmarks": int(len(lm)), "perplexity": perp, "neighbours": int(k),
            "seed": seed, "kl_divergence": float(tsne.kl_divergence_)}
    return y, info


def build(scores: Dict[str, np.ndarray], calls_of: Callable[[np.ndarray], Sequence[str]],
          n_landmarks: int = N_LANDMARKS, log: Callable[[str], None] = print
          ) -> Tuple[Dict[str, Any], bytes]:
    keys = [str(k) for k in scores["keys"]]
    p = np.asarray(scores["p"], dtype=np.float32)
    ticks = np.asarray(scores["ticks"], dtype=np.float64)
    probs, calls, start, count, stop = action_states(p, calls_of)
    n_states = len(probs)
    log(f"{len(p)} launches, {n_states:,} action states")

    q = np.clip(np.rint(probs * P_SCALE), 0, P_SCALE).astype(np.uint8)
    uq, inverse = np.unique(q, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    v_cls = most_probable_action(uq.astype(np.float64) / P_SCALE)
    cls = v_cls[inverse]
    log(f"{len(uq):,} distinct probability vectors; t-SNE on up to {n_landmarks:,} landmarks")
    y_u, tsne_info = embed(uq.astype(np.float64) / P_SCALE, v_cls, n_landmarks)
    y_u -= (y_u.max(axis=0) + y_u.min(axis=0)) / 2.0
    v_pos = (y_u / (float(np.abs(y_u).max()) or 1.0)).astype("<f4")
    s_vec = inverse.astype("<u2" if len(uq) < 65536 else "<u4")

    blobs = [("vec_pos", v_pos), ("vec_prob", uq), ("vec_class", v_cls),
             ("state_vec", s_vec), ("state_call", calls)]
    layout, parts, offset = {}, [], 0
    for name, arr in blobs:
        pad = -offset % arr.dtype.itemsize                 # typed-array views need alignment
        parts.append(b"\0" * pad)
        offset += pad
        raw = np.ascontiguousarray(arr).tobytes()
        layout[name] = {"offset": offset, "bytes": len(raw), "dtype": arr.dtype.str,
                        "shape": list(arr.shape)}
        parts.append(raw)
        offset += len(raw)

    y, manifest = np.asarray(scores["y"]), np.asarray(scores["manifest"], dtype=np.float64)
    truth = [[[int(k), round(float(manifest[i, k]), 1) if np.isfinite(manifest[i, k]) else None]
              for k in np.flatnonzero(y[i])] for i in range(len(p))]
    holdout = np.asarray(scores["holdout"], dtype=bool)
    classes = [{"mode": "NOMINAL", "display": "Nominal", "phase": "all", "call": None}]
    classes += [{"mode": k, "display": BY_KEY[k].display, "phase": BY_KEY[k].phase,
                 "call": BY_KEY[k].call} for k in keys]
    per_launch = np.repeat(np.arange(len(p)), count)
    meta = {
        "version": 1, "created_by": "build_universe.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_launches": int(len(p)), "n_states": int(n_states), "n_distinct": int(len(uq)),
        "n_holdout_states": int(holdout[per_launch].sum()),
        "tick_s": float(ticks[1] - ticks[0]), "t0": float(ticks[0]), "n_ticks": int(len(ticks)),
        "keys": keys, "classes": classes, "calls": list(CALLS),
        "thresholds": [round(float(t), 4) for t in scores["thresholds"]],
        "prob_scale": P_SCALE, "most_probable_from": MOST_PROBABLE,
        "jitter": JITTER, "layout": layout,
        "counts": {"class": np.bincount(cls, minlength=len(classes)).tolist(),
                   "call": np.bincount(calls, minlength=len(CALLS)).tolist(),
                   "stopped": {c: stop.count(c) for c in STOP_CALLS}},
        "tsne": {**tsne_info, "space": f"{len(keys)} failure-mode probabilities"},
        "source": {"scores": "layer-4/streaming_scores.npz",
                   "work": "out-of-fold scores (5 folds)", "holdout": "final models"},
        "launches": {"seed": [int(s) for s in scores["seeds"]],
                     "holdout": holdout.astype(int).tolist(),
                     "start": start.tolist(), "count": count.tolist(), "stop": stop,
                     "truth": truth},
    }
    return meta, b"".join(parts)


def agent_calls(keys: Sequence[str], ticks: np.ndarray) -> Callable[[np.ndarray], Sequence[str]]:
    """The agent's own call policy on one launch's probabilities (needs layer-4/model.pkl)."""
    from agent import ControlRoomAgent

    agent = ControlRoomAgent()
    if list(agent.keys) != list(keys):
        sys.exit("the failure modes in the scores and in layer-4/model.pkl are in a different "
                 "order: rerun run_factory.py")
    return lambda p_launch: agent.decide(ticks, None, np.asarray(p_launch, np.float64),
                                         explain=False).calls


def write(out_dir: Path, meta: Dict[str, Any], blob: bytes) -> None:
    (out_dir / "universe.json").write_text(json.dumps(meta, separators=(",", ":")),
                                           encoding="utf-8")
    with open(out_dir / "universe.bin.gz", "wb") as fh:           # mtime=0: same bytes each build
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, compresslevel=9, mtime=0) as gz:
            gz.write(blob)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", type=Path, default=DOMAIN_DIR / "layer-4" / "streaming_scores.npz")
    ap.add_argument("--landmarks", type=int, default=N_LANDMARKS)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    clock = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - clock:6.1f}s] {msg}", flush=True)

    for need in (args.scores, DOMAIN_DIR / "layer-4" / "model.pkl"):
        if not need.exists():
            sys.exit(f"{need} is missing: run run_factory.py first")
    with np.load(args.scores) as npz:
        scores = {k: npz[k] for k in npz.files}
    keys = [str(k) for k in scores["keys"]]
    meta, blob = build(scores, agent_calls(keys, scores["ticks"]), args.landmarks, log)
    out = args.out_dir
    write(out, meta, blob)
    log(f"wrote universe.json ({(out / 'universe.json').stat().st_size / 1e6:.2f} MB) and "
        f"universe.bin.gz ({(out / 'universe.bin.gz').stat().st_size / 1e6:.2f} MB, "
        f"{len(blob) / 1e6:.1f} MB unpacked): {meta['n_states']:,} action states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
