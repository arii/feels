"""
Atlas module — MongoDB Atlas connection, collection setup, teach/identify pipeline.

Collections:
  runs        — raw telemetry arrays, one document per motion run
  signatures  — extracted feature vector + embedding, indexed for vector search
  items       — ground-truth item catalogue (label, description, photo_url)
"""

import argparse
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

import certifi
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid, OperationFailure
from pymongo.operations import SearchIndexModel

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DATABASE", "physical_ai_db")

# Vector search indexes on signatures.feature_vector, keyed by embedding dim.
# 10-D serves hand/pca; the chronos time-series FM needs its own 2304-D index.
VECTOR_INDEXES = {
    10: "cup_dynamics_index",
    2304: "cup_dynamics_chronos_index",
}


def _index_for_dim(dim: int) -> str:
    if dim not in VECTOR_INDEXES:
        raise ValueError(
            f"no vector index for {dim}-D embeddings (have: {sorted(VECTOR_INDEXES)})")
    return VECTOR_INDEXES[dim]

_client: MongoClient | None = None


def get_db():
    """Return the MongoDB database handle, connecting on first call.

    Returns:
        The feels database object.
    Raises:
        RuntimeError: when MONGODB_URI is not set.
    """
    global _client
    if _client is None:
        if not MONGODB_URI:
            raise RuntimeError(
                "MONGODB_URI not set. Add it to your .env file.\n"
                "Format: mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/"
            )
        _client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    return _client[DB_NAME]


# ---------------------------------------------------------------------------
# One-time setup: create collections + basic indexes
# ---------------------------------------------------------------------------

def setup_collections():
    """
    Create the three collections and their scalar indexes.
    The vector search index on signatures.feature_vector must be created
    in the Atlas UI (see instructions below).
    Run this once at the start of the hackathon.
    """
    db = get_db()
    for name in ("runs", "signatures", "items"):
        try:
            db.create_collection(name)
            print(f"  created collection: {name}")
        except CollectionInvalid:
            print(f"  collection already exists: {name}")

    # Scalar indexes for fast filtering
    db.signatures.create_index([("label", 1)])
    db.signatures.create_index([("session", 1)])
    db.signatures.create_index([("label", 1), ("session", 1)])
    db.items.create_index([("label", 1)], unique=True)
    print("  scalar indexes created")

    # Vector Search indexes (requires Atlas M0+ cluster)
    _create_vector_index(db, "cup_dynamics_index", 10)


# ---------------------------------------------------------------------------
# Vector Search index creation
# ---------------------------------------------------------------------------

def _create_vector_index(db, name, dims):
    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "feature_vector",
                    "numDimensions": dims,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "label"},
                {"type": "filter", "path": "session"},
            ]
        },
        name=name,
        type="vectorSearch",
    )
    try:
        db.signatures.create_search_index(model)
        print(f"  vector search index '{name}' creation requested — takes ~1 min to build")
        print("  check status: Atlas UI → your cluster → Atlas Search tab")
    except OperationFailure as e:
        if "already exists" in str(e):
            print(f"  vector search index '{name}' already exists — skipping")
        else:
            print(f"  vector search index error: {e}")


def ensure_vector_indexes(which="all"):
    """Create Atlas vector search indexes. Idempotent — safe to re-run.

    Run once before first use of an embedder whose index doesn't exist yet:
        python -c "import atlas; atlas.ensure_vector_indexes('chronos')"

    which: "all" | "index_10d" | "chronos"
    """
    db = get_db()
    targets = {"index_10d": ("cup_dynamics_index", 10),
               "chronos": ("cup_dynamics_chronos_index", 2304)}
    if which != "all":
        targets = {which: targets[which]}
    for _, (name, dims) in targets.items():
        _create_vector_index(db, name, dims)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _run_doc(label: str, motion_type: str, raw_data: list[float], session: str = "thu") -> dict:
    return {
        "label": label,
        "motion_type": motion_type,
        "raw_data": raw_data,
        "session": session,
        "created_at": datetime.now(timezone.utc),
    }


def _signature_doc(
    label: str,
    feature_vector: list[float],
    motion_type: str,
    properties: dict | None = None,
    session: str = "thu",
    run_id=None,
) -> dict:
    return {
        "label": label,
        "motion_type": motion_type,
        "feature_vector": feature_vector,
        "properties": properties or {},
        "session": session,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# teach() — store a labeled run + its signature
# ---------------------------------------------------------------------------

def teach(
    label: str,
    feature_vector: list[float],
    raw_data: list[float] | None = None,
    motion_type: str = "3in1_macro",
    properties: dict | None = None,
    session: str = "thu",
) -> dict:
    """
    Insert a labeled run into Atlas.

    Args:
        label:          e.g. "dish_soap_half_full"
        feature_vector: normalized embedding list[float] (10-D hand/pca or 2304-D chronos)
        raw_data:       optional raw joint torque time-series
        motion_type:    "3in1_macro" | "sine_shake" | "tilt" | "fig8"
        properties:     optional dict {"estimated_mass_g": 185, "fill_ratio": 0.5}
        session:        "thu" or "fri" — used to filter vector search by day

    Returns:
        dict with inserted run_id and sig_id
    """
    db = get_db()

    run_id = None
    if raw_data is not None:
        result = db.runs.insert_one(_run_doc(label, motion_type, raw_data, session))
        run_id = result.inserted_id

    sig_result = db.signatures.insert_one(
        _signature_doc(label, feature_vector, motion_type, properties, session, run_id)
    )

    print(f"[teach] stored '{label}'  sig_id={sig_result.inserted_id}")
    return {"run_id": str(run_id), "sig_id": str(sig_result.inserted_id)}


# ---------------------------------------------------------------------------
# identify() — k-NN vector search, returns label + confidence
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Material taxonomy: what matters is WHAT is inside, not how full the cup is.
# Taught labels carry capacity suffixes (water_025pct, solid_150g,
# dish_soap_half); material_of() strips them to the coarse material class.
# New labels should follow the same <material>_<capacity> convention, or be
# added to MATERIAL_STEMS.
# ---------------------------------------------------------------------------

MATERIAL_STEMS = {
    "water": "water",
    "soap": "soap",
    "dish_soap": "soap",
    "candy": "candy",
    "solid": "solid",
    "screws": "screws",
    "screw": "screws",
    "empty": "empty",
    "empty_cup": "empty_cup",
}

_CAPACITY_SUFFIX = re.compile(r"_(\d+(pct|g|ml|oz)|half|full|mixed|empty)$")


def material_of(label: str) -> str:
    """Coarse material class for a taught label.

    water_025pct / water_150g / water_half -> water; solid_020g -> solid;
    dish_soap_half -> soap; candy_mixed -> candy.
    empty and empty_cup are distinct classes (gripper empty vs holding empty cup).
    Unknown labels map to themselves (their own material class).

    Args:
        label: taught signature label, e.g. "water_075pct".
    Returns:
        Material stem, e.g. "water".
    """
    stem = _CAPACITY_SUFFIX.sub("", label.strip().lower())
    return MATERIAL_STEMS.get(stem, stem)


# Coarser still: the demo headline. Liquids slosh (variance + phase lag),
# solids stay rigidly coupled or rattle, empty is empty. Fine material type
# (water vs soap) is a harder claim than the 3.5 Hz signal reliably supports;
# the category is the robust one.
CATEGORY_OF_MATERIAL = {
    "water": "water",
    "empty": "empty",
    "empty_cup": "empty_cup",
    "screws": "screws",
    "candy": "candy",
}


def category_of(label: str) -> str:
    """Class for a taught label: water / empty / empty_cup / screws / candy.

    Unknown materials map to themselves (their own category).

    Args:
        label: taught signature label, e.g. "water".
    Returns:
        One of the five classes, or the material itself when unknown.
    """
    mat = material_of(label)
    return CATEGORY_OF_MATERIAL.get(mat, mat)


def identify(
    feature_vector: list[float],
    top_k: int = 5,
    session: str | None = None,
    confidence_threshold: float = 0.82,
) -> dict:
    """
    Query Atlas Vector Search to identify what's in the cup.

    Args:
        feature_vector:        normalized query embedding — 10-D (hand/pca) or
                               2304-D (chronos); the vector index is picked by dim
        top_k:                 number of neighbours to retrieve
        session:               if set, filter to this session tag ("thu"/"fri")
        confidence_threshold:  below this score, flag as low-confidence

    Returns:
        {
          "label":      predicted label string (finest taught granularity),
          "confidence": float 0–1,
          "matches":    list of top-k {label, score} dicts,
          "low_confidence": bool,
          "material":   predicted material class (capacity suffixes stripped),
          "material_confidence": float 0–1,
          "material_matches": list of {material, score} dicts,
          "material_low_confidence": bool,
          "category":   predicted category (liquid / solid / empty),
          "category_confidence": float 0–1,
          "category_matches": list of {category, score} dicts,
          "category_low_confidence": bool
        }
    """
    db = get_db()

    # Build the $vectorSearch stage (index picked by embedding dimension)
    vector_search_stage = {
        "$vectorSearch": {
            "index": _index_for_dim(len(feature_vector)),
            "path": "feature_vector",
            "queryVector": feature_vector,
            "numCandidates": top_k * 4,
            "limit": top_k,
            **({"filter": {"session": session}} if session else {}),
        }
    }

    project_stage = {
        "$project": {
            "label": 1,
            "session": 1,
            "score": {"$meta": "vectorSearchScore"},
            "_id": 0,
        }
    }

    results = list(db.signatures.aggregate([vector_search_stage, project_stage]))

    if not results:
        # Empty library: same dict shape as a real result so the dashboard
        # renders "teach items first" instead of 500ing on a KeyError.
        return {"label": None, "confidence": 0.0, "matches": [],
                "margin": 1.0, "ambiguous": False, "low_confidence": True,
                "material": None, "material_confidence": 0.0, "material_matches": [],
                "material_margin": 1.0, "material_ambiguous": False,
                "material_low_confidence": True,
                "category": None, "category_confidence": 0.0, "category_matches": [],
                "category_margin": 1.0, "category_ambiguous": False,
                "category_low_confidence": True}

    # Group by label: mean similarity of each label's hits
    label_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        label_scores[r["label"]].append(r["score"])

    ranked = sorted(
        {lbl: sum(scores) / len(scores) for lbl, scores in label_scores.items()}.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    best_label, best_confidence = ranked[0]
    margin = best_confidence - ranked[1][1] if len(ranked) > 1 else 1.0
    # A tiny lead across labels means the space can't separate this query
    # (e.g. stale feature scales collapsing every embedding to ~0.97 cosine).
    # Report ambiguity instead of a confident-looking number.
    ambiguous = margin < 0.02
    matches = [{"label": lbl, "score": round(score, 4)} for lbl, score in ranked]

    # Material-level aggregation: capacity variants of one material vote
    # together. water_025pct vs water_075pct is a fill question; water vs
    # soap is the material question the demo actually asks.
    mat_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        mat_scores[material_of(r["label"])].append(r["score"])
    mat_ranked = sorted(
        {m: sum(s) / len(s) for m, s in mat_scores.items()}.items(),
        key=lambda x: x[1], reverse=True,
    )
    best_mat, best_mat_conf = mat_ranked[0]
    mat_margin = best_mat_conf - mat_ranked[1][1] if len(mat_ranked) > 1 else 1.0
    mat_ambiguous = mat_margin < 0.02
    mat_matches = [{"material": m, "score": round(s, 4)} for m, s in mat_ranked]

    print(f"[identify] → '{best_label}'  confidence={best_confidence:.3f}  margin={margin:.3f}"
          + ("  AMBIGUOUS" if ambiguous else ""))
    print(f"           material '{best_mat}'  confidence={best_mat_conf:.3f}  margin={mat_margin:.3f}"
          + ("  AMBIGUOUS" if mat_ambiguous else ""))

    # Category-level aggregation: liquid / solid / empty. The coarsest claim
    # and the most robust — the level the demo headlines.
    cat_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        cat_scores[category_of(r["label"])].append(r["score"])
    cat_ranked = sorted(
        {c: sum(s) / len(s) for c, s in cat_scores.items()}.items(),
        key=lambda x: x[1], reverse=True,
    )
    best_cat, best_cat_conf = cat_ranked[0]
    cat_margin = best_cat_conf - cat_ranked[1][1] if len(cat_ranked) > 1 else 1.0
    cat_ambiguous = cat_margin < 0.02
    cat_matches = [{"category": c, "score": round(s, 4)} for c, s in cat_ranked]
    print(f"           category '{best_cat}'  confidence={best_cat_conf:.3f}  margin={cat_margin:.3f}"
          + ("  AMBIGUOUS" if cat_ambiguous else ""))
    for m in matches:
        print(f"           {m['label']:30s}  {m['score']:.4f}")

    return {
        "label": best_label,
        "confidence": best_confidence,
        "matches": matches,
        "margin": round(margin, 4),
        "ambiguous": ambiguous,
        "low_confidence": (best_confidence < confidence_threshold) or ambiguous,
        "material": best_mat,
        "material_confidence": best_mat_conf,
        "material_matches": mat_matches,
        "material_margin": round(mat_margin, 4),
        "material_ambiguous": mat_ambiguous,
        "material_low_confidence": (best_mat_conf < confidence_threshold) or mat_ambiguous,
        "category": best_cat,
        "category_confidence": best_cat_conf,
        "category_matches": cat_matches,
        "category_margin": round(cat_margin, 4),
        "category_ambiguous": cat_ambiguous,
        "category_low_confidence": (best_cat_conf < confidence_threshold) or cat_ambiguous,
    }


# ---------------------------------------------------------------------------
# Supervised decision head (LDA)
# ---------------------------------------------------------------------------

# Measured held-out accuracy of the LDA head (manifest 84/21 split, Jan-2026
# bake-off). Displayed as the trust signal because LDA's own probabilities
# are uncalibrated (correct vs wrong predictions are indistinguishable at
# ~0.57 vs ~0.55 max-proba). Honest beats precise.
EMPIRICAL_ACCURACY = {
    "label": 0.43,
    "material": 0.43,
    "category": 0.52,
    "binary": 0.86,
}


def _lda_library(db):
    """Fetch (label, z) for every signature with a stored raw feature vector."""
    docs = list(db.signatures.find({"properties.z": {"$exists": True}},
                                   {"label": 1, "properties.z": 1, "_id": 0}))
    labels, Z = [], []
    for d in docs:
        z = (d.get("properties") or {}).get("z")
        if d.get("label") and isinstance(z, (list, tuple)) and len(z) == 10:
            try:
                labels.append(str(d["label"]))
                Z.append([float(v) for v in z])
            except (TypeError, ValueError):
                continue
    return labels, np.array(Z, dtype=float) if Z else np.zeros((0, 10))


def predict_lda(query_z, confidence_threshold=0.82):
    """Supervised decision head: shrinkage-LDA on standardized z.

    Fetches the whole labeled library, fits StandardScaler + LDA in
    milliseconds (no training step, no deployment — refit on every predict,
    so a teach is effective immediately), and returns the same dict shape
    as identify() so the dashboard works unchanged.

    Measured accuracy (held-out): category 52%, material 43%, empty/nonempty
    86% — roughly 2x the vector-search retrieval rule on the same features.
    Raises ValueError when the library can't support it; callers fall back
    to identify().

    Args:
        query_z: 10-D raw feature vector of the query run.
        confidence_threshold: below this score the result is flagged low-confidence.
    Returns:
        Dict with label, material, category, matches, margins, and
        decider="lda" — the same shape as identify().
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    db = get_db()
    labels, Z = _lda_library(db)
    n_classes = len(set(labels))
    if len(Z) < 10 or n_classes < 2:
        raise ValueError(
            f"LDA needs >=10 signatures across >=2 classes, have {len(Z)}/{n_classes}")
    qz = np.asarray(query_z, dtype=float).reshape(1, -1)
    if qz.shape[1] != Z.shape[1]:
        raise ValueError(f"query z dim {qz.shape[1]} != library {Z.shape[1]}")

    sc = StandardScaler().fit(Z)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(sc.transform(Z), labels)
    proba = clf.predict_proba(sc.transform(qz))[0]
    classes = [str(c) for c in clf.classes_]
    order = np.argsort(proba)[::-1]

    def aggregate(fn):
        """Sum class probabilities into groups (material / category) and rank."""
        groups = {}
        for i in order:
            g = fn(classes[i])
            groups[g] = groups.get(g, 0.0) + float(proba[i])
        return sorted(groups.items(), key=lambda kv: kv[1], reverse=True)

    matches = [{"label": classes[i], "score": round(float(proba[i]), 4)}
               for i in order[:5]]
    best_label = classes[order[0]]
    mat_ranked = aggregate(material_of)
    cat_ranked = aggregate(category_of)
    best_mat, best_mat_p = mat_ranked[0]
    best_cat, best_cat_p = cat_ranked[0]
    # close call = top two within 0.15 probability (weak signal, but honest)
    close = (float(proba[order[0]]) - float(proba[order[1]])) < 0.15 if len(order) > 1 else False
    mat_close = (mat_ranked[0][1] - mat_ranked[1][1]) < 0.15 if len(mat_ranked) > 1 else False
    cat_close = (cat_ranked[0][1] - cat_ranked[1][1]) < 0.15 if len(cat_ranked) > 1 else False

    print(f"[predict_lda] → '{best_label}'  material '{best_mat}'  category '{best_cat}'"
          + ("  CLOSE-CALL" if close else ""))

    return {
        "decider": "lda",
        "label": best_label,
        "confidence": EMPIRICAL_ACCURACY["material"],
        "matches": matches,
        "margin": round(float(proba[order[0]] - proba[order[1]]) if len(order) > 1 else 1.0, 4),
        "ambiguous": close,
        "low_confidence": close,
        "material": best_mat,
        "material_confidence": EMPIRICAL_ACCURACY["material"],
        "material_matches": [{"material": m, "score": round(float(p), 4)} for m, p in mat_ranked],
        "material_margin": round(float(mat_ranked[0][1] - mat_ranked[1][1]) if len(mat_ranked) > 1 else 1.0, 4),
        "material_ambiguous": mat_close,
        "material_low_confidence": mat_close,
        "category": best_cat,
        "category_confidence": EMPIRICAL_ACCURACY["category"],
        "category_matches": [{"category": c, "score": round(float(p), 4)} for c, p in cat_ranked],
        "category_margin": round(float(cat_ranked[0][1] - cat_ranked[1][1]) if len(cat_ranked) > 1 else 1.0, 4),
        "category_ambiguous": cat_close,
        "category_low_confidence": cat_close,
        "n_signatures": len(Z),
    }


# ---------------------------------------------------------------------------
# Seed the item catalogue
# ---------------------------------------------------------------------------

BASELINE_ITEMS = [
    {
        "label": "empty_cup",
        "description": "Empty lidded coffee cup — baseline tare reference",
        "category": "empty",
        "expected_signature": "Low mass, near-zero shake resistance, no rattle",
    },
    {
        "label": "water_half",
        "description": "Half-full cup of water",
        "category": "newtonian_liquid",
        "expected_signature": "Medium mass, dynamic phase lag, low high-frequency rattle",
    },
    {
        "label": "dish_soap_half",
        "description": "Half-full cup of dish soap",
        "category": "viscous_liquid",
        "expected_signature": "Medium mass, heavy viscous damping, delayed peak torque",
    },
    {
        "label": "candy_loose",
        "description": "Halloween fun-size candy (loose pieces)",
        "category": "loose_solid",
        "expected_signature": "Sharp high-frequency impact chatter (5–25 Hz rattle peaks)",
    },
    {
        "label": "solid_weight",
        "description": "Small metal tool or solid roll of tape",
        "category": "rigid_solid",
        "expected_signature": "High mass, rigid coupling, zero rattle, zero phase lag",
    },
]


def seed_items(overwrite: bool = False):
    """Insert baseline items into the items catalogue."""
    db = get_db()
    for item in BASELINE_ITEMS:
        if overwrite:
            db.items.replace_one({"label": item["label"]}, item, upsert=True)
        else:
            db.items.update_one({"label": item["label"]}, {"$setOnInsert": item}, upsert=True)
        print(f"  seeded item: {item['label']}")


# ---------------------------------------------------------------------------
# Quick smoke test — run directly: python atlas.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up Atlas collections/vector index for arii/feels.",
        epilog="""Run once per fresh database. Needs MONGODB_URI in .env, and the Atlas
IP allowlist must include this machine. Plain `python atlas.py` only creates
collections + the vector index and seeds the item catalogue — it does NOT
touch the signature library. `--smoke` additionally inserts one RANDOM
vector; only use it on an empty database.""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true",
                        help="Also run a smoke teach() with a RANDOM vector + identify(). "
                             "DO NOT use after real items are taught — it inserts a junk "
                             "'water_half' row that will pollute $vectorSearch results.")
    args = parser.parse_args()

    print("=== Atlas setup ===\n")
    print(f"Database: {DB_NAME}")

    db = get_db()
    print(f"Connected. Collections: {db.list_collection_names()}\n")

    print("--- Setting up collections ---")
    setup_collections()

    print("\n--- Seeding item catalogue ---")
    seed_items()

    if args.smoke:
        print("\n--- Inserting a fake teach() run (smoke test, random vector) ---")
        fake_vector = [0.1, 0.9, -0.3, 0.5, 0.2, -0.1, 0.4, 0.0, 0.3, 0.6]
        teach(
            label="water_half",
            feature_vector=fake_vector,
            motion_type="3in1_macro",
            properties={"estimated_mass_g": 150, "fill_ratio": 0.5},
            session="thu",
        )

        print("\n--- Running identify() ---")
        result = identify(fake_vector, top_k=5, session="thu")
        print(f"\n  Prediction : {result['label']}")
        print(f"  Confidence : {result['confidence']:.3f}")
        print(f"  Low conf?  : {result['low_confidence']}")
        print("\nDone.")
