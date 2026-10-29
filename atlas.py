"""
Atlas module — MongoDB Atlas connection, collection setup, teach/identify pipeline.

Collections:
  runs        — raw telemetry arrays, one document per motion run
  signatures  — extracted feature vector + embedding, indexed for vector search
  items       — ground-truth item catalogue (label, description, photo_url)
"""

import os
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
          "label":      predicted label string,
          "confidence": float 0–1,
          "matches":    list of top-k {label, score} dicts,
          "low_confidence": bool
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
    from collections import defaultdict
    label_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        label_scores[r["label"]].append(r["score"])

    ranked = sorted(
        {lbl: sum(scores) / len(scores) for lbl, scores in label_scores.items()}.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    best_label, best_confidence = ranked[0]
    matches = [{"label": lbl, "score": round(score, 4)} for lbl, score in ranked]

    print(f"[identify] → '{best_label}'  confidence={best_confidence:.3f}")
    for m in matches:
        print(f"           {m['label']:30s}  {m['score']:.4f}")

    return {
        "label": best_label,
        "confidence": best_confidence,
        "matches": matches,
        "low_confidence": best_confidence < confidence_threshold,
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
    print("=== Atlas setup ===\n")
    print(f"Database: {DB_NAME}")

    db = get_db()
    print(f"Connected. Collections: {db.list_collection_names()}\n")

    print("--- Setting up collections ---")
    setup_collections()

    print("\n--- Seeding item catalogue ---")
    seed_items()

    print("\n--- Inserting a fake teach() run ---")
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
