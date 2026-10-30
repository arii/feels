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
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid, OperationFailure
from pymongo.operations import SearchIndexModel

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DATABASE", "physical_ai_db")

_client: MongoClient | None = None


def get_db():
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

    # Vector Search index (requires Atlas M0+ cluster)
    _create_vector_index(db)


# ---------------------------------------------------------------------------
# Vector Search index creation
# ---------------------------------------------------------------------------

def _create_vector_index(db):
    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "feature_vector",
                    "numDimensions": 10,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "label"},
                {"type": "filter", "path": "session"},
            ]
        },
        name="cup_dynamics_index",
        type="vectorSearch",
    )
    try:
        db.signatures.create_search_index(model)
        print("  vector search index creation requested — takes ~1 min to build")
        print("  check status: Atlas UI → your cluster → Atlas Search tab")
    except OperationFailure as e:
        if "already exists" in str(e):
            print("  vector search index already exists — skipping")
        else:
            print(f"  vector search index error: {e}")


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
        feature_vector: 10-dim normalized embedding list[float]
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
    "empty_cup": "empty",
}

_CAPACITY_SUFFIX = re.compile(r"_(\d+(pct|g|ml|oz)|half|full|mixed|empty)$")


def material_of(label: str) -> str:
    """Coarse material class for a taught label.

    water_025pct / water_150g / water_half -> water; solid_020g -> solid;
    dish_soap_half -> soap; candy_mixed -> candy; empty_cup -> empty.
    Unknown labels map to themselves (their own material class).
    """
    stem = _CAPACITY_SUFFIX.sub("", label.strip().lower())
    return MATERIAL_STEMS.get(stem, stem)


def identify(
    feature_vector: list[float],
    top_k: int = 5,
    session: str | None = None,
    confidence_threshold: float = 0.82,
) -> dict:
    """
    Query Atlas Vector Search to identify what's in the cup.

    Args:
        feature_vector:        10-dim normalized query embedding
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
          "material_low_confidence": bool
        }
    """
    db = get_db()

    # Build the $vectorSearch stage
    vector_search_stage = {
        "$vectorSearch": {
            "index": "cup_dynamics_index",
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
        return {"label": None, "confidence": 0.0, "matches": [], "low_confidence": True}

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
    parser = argparse.ArgumentParser(description="Set up Atlas collections/vector index for arii/feels.")
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
