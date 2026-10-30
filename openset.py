#!/usr/bin/env python3
"""
openset.py — open-set "teach & recognize" loop on top of Atlas Vector Search.

The classifier has fixed classes; Atlas is a *growing* memory. This module adds
the two things a trained model structurally can't do:

  • NOVELTY DETECTION  — if a recording's nearest stored neighbour is below a
    similarity threshold, we say "I don't recognise this" instead of forcing a
    wrong label.
  • ONE-SHOT TEACH     — label that novel item once and it is INSTANTLY
    recognisable (a single Atlas insert; no retrain, no redeploy).

Embeddings use the teammate's 10-D `feature_extractor` (what the Atlas
`cup_dynamics_index` was built on), so new items live in the same space as the
existing `signatures`.

Usage:
    python openset.py recognize <recording.csv>            # known? novel? ambiguous?
    python openset.py teach <recording.csv> --label oat_milk
    python openset.py demo <recording.csv> [--label X]      # recognize -> teach-if-novel -> re-recognize
    python openset.py stats                                 # what the memory currently knows

Note (honest): on the current proprioception-only embeddings, items collapse to
~0.97 cosine, so novelty/fine-grained separation is weak — the LOOP is correct and
sharpens the moment a richer embedding (vision/audio) is added. See models/README.md.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import atlas
import feature_extractor as fx

MOTION_TYPE = "swirl_macro"


# ── embedding ───────────────────────────────────────────────────────────────
def embed(csv_path: str):
    """Recording CSV -> (10-D normalized embedding list, metadata dict)."""
    df = pd.read_csv(csv_path)
    out = fx.extract_features(df)
    meta = {
        "has_swirl": bool((df["phase"] == fx.SWIRL_PHASE).any()),
        "true_label": str(df["object_label"].iloc[0]) if "object_label" in df else None,
    }
    return out["e"], out["z"], meta


# ── nearest neighbours (raw, for novelty) ─────────────────────────────────────
def nearest(embedding, top_k=5, session=None):
    """Raw nearest signatures with cosine scores (highest first)."""
    db = atlas.get_db()
    stage = {
        "$vectorSearch": {
            "index": "cup_dynamics_index",
            "path": "feature_vector",
            "queryVector": embedding,
            "numCandidates": max(50, top_k * 10),
            "limit": top_k,
            **({"filter": {"session": session}} if session else {}),
        }
    }
    proj = {"$project": {"label": 1, "session": 1, "_id": 0,
                         "score": {"$meta": "vectorSearchScore"}}}
    return list(db.signatures.aggregate([stage, proj]))


# ── recognize (novelty-aware) ─────────────────────────────────────────────────
def recognize(embedding, novelty_threshold=0.90, ambiguity_margin=0.02,
              top_k=5, session=None) -> dict:
    """
    Returns {status, label, nn_score, margin, matches}.
      status = "known"    nearest neighbour is close AND one label clearly wins
             = "novel"    nearest neighbour below novelty_threshold -> unseen item
             = "ambiguous" close to memory but no label separates (teach to disambiguate)
             = "empty_memory"
    """
    hits = nearest(embedding, top_k=top_k, session=session)
    if not hits:
        return {"status": "empty_memory", "label": None, "nn_score": 0.0,
                "margin": 0.0, "matches": []}

    nn_score = hits[0]["score"]                              # single best neighbour
    # per-label mean score
    from collections import defaultdict
    by = defaultdict(list)
    for h in hits:
        by[h["label"]].append(h["score"])
    ranked = sorted(((l, sum(s) / len(s)) for l, s in by.items()),
                    key=lambda t: t[1], reverse=True)
    best_label, best_conf = ranked[0]
    margin = best_conf - ranked[1][1] if len(ranked) > 1 else 1.0
    matches = [{"label": l, "score": round(s, 4)} for l, s in ranked]

    if nn_score < novelty_threshold:
        status = "novel"
    elif margin < ambiguity_margin:
        status = "ambiguous"
    else:
        status = "known"
    return {"status": status, "label": best_label if status == "known" else None,
            "candidate": best_label, "confidence": round(best_conf, 4),
            "nn_score": round(nn_score, 4), "margin": round(margin, 4),
            "matches": matches}


# ── one-shot teach ────────────────────────────────────────────────────────────
def teach_new(label, embedding, z=None, meta=None, session="live") -> dict:
    props = {"source": "openset", **({} if not meta else {
        "has_swirl": meta.get("has_swirl")})}
    ids = atlas.teach(label=label, feature_vector=embedding,
                      raw_data=None, motion_type=MOTION_TYPE,
                      properties=props, session=session)
    print(f"[teach] '{label}' stored — now recognisable (sig_id={ids['sig_id']})")
    return ids


# ── CLI ───────────────────────────────────────────────────────────────────────
def _print_reco(r):
    print(f"\n  status     : {r['status'].upper()}")
    print(f"  nn_score   : {r['nn_score']:.3f}   margin={r['margin']:.3f}")
    if r["matches"]:
        print("  neighbours : " + ", ".join(f"{m['label']}={m['score']:.3f}" for m in r["matches"][:4]))
    if r["status"] == "known":
        print(f"  => recognised as '{r['label']}' (conf {r['confidence']:.3f})")
    elif r["status"] == "novel":
        print("  => NOVEL item — not in memory. Teach it with: openset.py teach <csv> --label <name>")
    elif r["status"] == "ambiguous":
        print(f"  => close to memory but ambiguous (best guess '{r['candidate']}'); teach to disambiguate")
    else:
        print("  => memory is empty — teach the first item")


def cmd_recognize(a):
    e, z, meta = embed(a.csv)
    print(f"file: {os.path.basename(a.csv)}  (true: {meta['true_label']}, swirl={meta['has_swirl']})")
    _print_reco(recognize(e, a.novelty, a.margin, a.top_k, a.session))


def cmd_teach(a):
    e, z, meta = embed(a.csv)
    teach_new(a.label, e, z, meta, session=a.session)


def cmd_demo(a):
    e, z, meta = embed(a.csv)
    label = a.label or meta["true_label"] or "unknown_item"
    print(f"=== OPEN-SET DEMO on {os.path.basename(a.csv)} (will teach as '{label}' if novel) ===")
    print("\n[1] First encounter — recognize:")
    r = recognize(e, a.novelty, a.margin, a.top_k, a.session); _print_reco(r)
    if r["status"] in ("novel", "empty_memory"):
        print(f"\n[2] One-shot teach as '{label}':")
        teach_new(label, e, z, meta, session=a.session)
        print("\n[3] Re-encounter the same item — recognize again:")
        _print_reco(recognize(e, a.novelty, a.margin, a.top_k, a.session))
    else:
        print("\n(already known — skipping teach)")


def cmd_stats(a):
    db = atlas.get_db()
    from collections import Counter
    sigs = list(db.signatures.find({}, {"label": 1, "_id": 0}))
    c = Counter(s["label"] for s in sigs)
    print(f"memory: {len(sigs)} signatures across {len(c)} labels")
    for lbl, n in sorted(c.items()):
        print(f"  {lbl:24} {n}")


def main():
    p = argparse.ArgumentParser(description="Open-set teach & recognize over Atlas")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("recognize", "teach", "demo"):
        sp = sub.add_parser(name)
        sp.add_argument("csv")
        sp.add_argument("--session", default=None)
        sp.add_argument("--novelty", type=float, default=0.90, help="cosine below this = novel")
        sp.add_argument("--margin", type=float, default=0.02, help="label margin below this = ambiguous")
        sp.add_argument("--top-k", type=int, default=5)
        if name in ("teach", "demo"):
            sp.add_argument("--label", default=None)
    sub.add_parser("stats")
    a = p.parse_args()
    {"recognize": cmd_recognize, "teach": cmd_teach, "demo": cmd_demo, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
