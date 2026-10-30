#!/usr/bin/env python3
"""
clear_old_sessions.py — Remove old-trajectory signatures from Atlas.

Keeps only signatures taught via the dashboard (session="thu").
Deletes HF imports (session="hf_5class") and anything else.

    python tools/clear_old_sessions.py              # delete old, keep dashboard teaches
    python tools/clear_old_sessions.py --dry-run    # show what would be deleted
    python tools/clear_old_sessions.py --all        # clear entire library

Requires MONGODB_URI set.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", action="store_true", help="clear entire library")
    args = ap.parse_args()

    from realtime_plot import _atlas
    at = _atlas()
    coll = at.get_db().signatures

    if args.all:
        filt = {}
        desc = "ALL signatures"
    else:
        # Keep dashboard teaches (session="thu"), delete everything else
        filt = {"session": {"$ne": "thu"}}
        desc = 'signatures where session != "thu"'

    n = coll.count_documents(filt)
    print(f"Found {n} {desc}")

    if args.dry_run:
        print("Dry run — not deleting.")
        # Show breakdown by session
        pipe = [{"$group": {"_id": "$session", "count": {"$sum": 1}}}]
        for d in coll.aggregate(pipe):
            print(f"  session={d['_id']!r}: {d['count']}")
        return

    if n == 0:
        print("Nothing to delete.")
        return

    result = coll.delete_many(filt)
    print(f"Deleted {result.deleted_count} signatures.")

    # Refresh the vspace cache
    from realtime_plot import _vspace_cache_refresh_from_atlas
    _vspace_cache_refresh_from_atlas()
    print("Vspace cache refreshed.")


if __name__ == "__main__":
    main()
