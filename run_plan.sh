#!/usr/bin/env bash
#
# run_plan.sh — guided data-collection run sheet for the feels cup-content study.
#
# TWO phases:
#   1) MAIN pass  — experiments in order (normal + swirl paths).
#                   Resumes at RESUME_FROM (default E9, since E1-E8 normal are done).
#   2) SWIRL CATCH-UP — swirl-only trials for the rigid solids (E2-E7) and the
#                   early swirls that were skipped (E1 empty, E8 water_025).
#                   Runs AFTER the main pass. Skip with SKIP_CATCHUP=1.
#
# Each trial runs ONE command:
#   repeatable_grasp.py --object <label> --record --cycles 1 [--swirl-smooth]
# driving the grasp cycle and writing a phase-annotated CSV to recordings/.
# (NORMAL and SWIRL share the object_label; SWIRL rows are tagged phase "SWIRL".)
#
# Pauses let you change contents (before each experiment), switch motion path,
# and re-grasp (before each trial).
#   [Enter] continue   s = skip rest of this experiment   q = quit
#
# Env knobs:
#   RESUME_FROM=E9   main pass starts at this experiment ID (default E9)
#   START_FROM=E12   hard override of the start ID
#   P0_ONLY=1        only priority-0 experiments (also skips the catch-up)
#   SCALE=0.25       multiply every trial count (min 1 per active cell)
#   SKIP_CATCHUP=1   don't run the swirl catch-up pass
#   DRY_RUN=1        print commands instead of running them
#
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
[ -d .venv ] && source .venv/bin/activate 2>/dev/null || true

P0_ONLY="${P0_ONLY:-0}"
SCALE="${SCALE:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CATCHUP="${SKIP_CATCHUP:-0}"
RESUME_FROM="${RESUME_FROM:-E9}"
START_FROM="${START_FROM:-$RESUME_FROM}"
STARTED=0; [ -z "$START_FROM" ] && STARTED=1

# ── MAIN experiment table ─────────────────────────────────────────────────
# ID | LABEL | NORMAL_N | SWIRL_N | P0(1/0) | SETUP INSTRUCTION
# (solid ladder E2-E7 carry swirl=0 here — their swirls run in the catch-up pass.)
EXPERIMENTS=(
  "E1|empty_cup|10|5|1|Empty cup, lid ON, nothing inside (baseline/tare)"
  "E2|solid_020g|10|0|0|Rigid 20 g weight FIXED (taped, can't shift), lid ON"
  "E3|solid_050g|10|0|1|Rigid 50 g weight FIXED, lid ON"
  "E4|solid_100g|10|0|1|Rigid 100 g weight FIXED, lid ON"
  "E5|solid_200g|10|0|1|Rigid 200 g weight FIXED, lid ON"
  "E6|solid_350g|10|0|0|Rigid 350 g weight FIXED, lid ON"
  "E7|solid_500g|10|0|0|Rigid 500 g weight FIXED, lid ON"
  "E8|water_025pct|15|15|0|Water at 25% fill, lid ON"
  "E9|water_050pct|15|15|1|Water at 50% fill, lid ON"
  "E10|water_075pct|15|15|0|Water at 75% fill, lid ON"
  "E11|water_100pct|15|15|0|Water at 100% fill, lid ON"
  "E12|soap_050pct|15|15|1|Dish soap at 50% fill, lid ON"
  "E13|oil_050pct|15|15|0|Cooking oil at 50% fill, lid ON"
  "E14|honey_050pct|12|12|0|Honey at 50% fill, lid ON"
  "E15|rice_100g|12|5|0|100 g rice (fine granular), loose, lid ON"
  "E16|beads_100g|12|5|0|100 g beads/pellets (medium), loose, lid ON"
  "E17|candy_100g|12|5|1|100 g fun-size candy, loose, lid ON"
  "E18|screws_100g|12|5|0|100 g metal screws (hard), loose, lid ON"
  "E19|candy_050g|12|0|0|50 g candy, loose, lid ON"
  "E20|candy_200g|12|0|0|200 g candy, loose, lid ON"
  "E21|solid_150g|15|15|1|CONFOUND: 150 g rigid weight FIXED, lid ON"
  "E22|water_150g|15|15|1|CONFOUND: ~150 mL water (~150 g), lid ON"
  "E23|candy_150g|15|15|1|CONFOUND: 150 g candy, loose, lid ON"
  "E24|soap_150g|15|15|1|CONFOUND: 150 g dish soap, lid ON"
  "E25|candy_horizontal_100g|15|15|0|100 g candy spread LOW/WIDE (horizontal), lid ON"
  "E26|water_vertical_100g|15|15|0|~100 g water in a TALL column, lid ON"
)

# ── SWIRL CATCH-UP (swirl-only, runs after main) ──────────────────────────
# LABEL | SWIRL_N | SETUP INSTRUCTION
SWIRL_CATCHUP=(
  "solid_020g|5|Rigid 20 g weight FIXED, lid ON"
  "solid_050g|5|Rigid 50 g weight FIXED, lid ON"
  "solid_100g|5|Rigid 100 g weight FIXED, lid ON"
  "solid_200g|5|Rigid 200 g weight FIXED, lid ON"
  "solid_350g|5|Rigid 350 g weight FIXED, lid ON"
  "solid_500g|5|Rigid 500 g weight FIXED, lid ON"
  "empty_cup|5|Empty cup, lid ON (skipped-early swirl)"
  "water_025pct|15|Water at 25% fill, lid ON (skipped-early swirl)"
)

# ── helpers ───────────────────────────────────────────────────────────────
scaled() { awk -v n="$1" -v s="$SCALE" 'BEGIN{ if(n<=0){print 0; exit} v=int(n*s+0.999); if(v<1)v=1; print v }'; }

pause() {
  printf "\n\033[36m%s\033[0m\n" "$1"
  read -r -p "   [Enter]=go  s=skip  q=quit > " ans
  case "$ans" in s|S) return 1 ;; q|Q) echo "— quitting —"; exit 0 ;; *) return 0 ;; esac
}

run_trial() { # label, i, total, path
  local label="$1" i="$2" tot="$3" path="$4" extra=""
  [ "$path" = "swirl" ] && extra="--swirl-smooth"
  pause "   ▶ $label  [$path]  trial $i/$tot — RE-GRASP: release & reposition the cup." || return 1
  local cmd="$PY repeatable_grasp.py --object \"$label\" --record --cycles 1 $extra"
  if [ "$DRY_RUN" = "1" ]; then echo "     DRY: $cmd"; else eval "$cmd"; fi
}

run_block() { # label, count, path
  local label="$1" n; n="$(scaled "$2")"; local path="$3"
  [ "$n" -eq 0 ] && return 0
  printf "\n\033[33m--- %s : %s path : %d trials ---\033[0m\n" "$label" "$path" "$n"
  [ "$path" = "swirl" ] && { pause "   SWIRL path for '$label' — confirm cup seated for orbital motion." || return 0; }
  local i
  for i in $(seq 1 "$n"); do
    run_trial "$label" "$i" "$n" "$path" || { echo "   (skipping rest of $label/$path)"; return 0; }
  done
}

# ── banner ────────────────────────────────────────────────────────────────
cat <<'BANNER'
============================================================
  feels — GUIDED DATA COLLECTION
  ⚠  PHYSICAL ARM. Clear workspace. Keep e-stop in reach.
============================================================
BANNER
echo "start=$START_FROM  P0_ONLY=$P0_ONLY  SCALE=$SCALE  SKIP_CATCHUP=$SKIP_CATCHUP  DRY_RUN=$DRY_RUN"
pause "PHASE 1 — MAIN pass (resumes at $START_FROM). Ready?"

# ── PHASE 1: main pass ────────────────────────────────────────────────────
for row in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r ID LABEL NORMAL_N SWIRL_N P0 SETUP <<< "$row"
  [ "$STARTED" -eq 0 ] && { [ "$ID" = "$START_FROM" ] && STARTED=1 || continue; }
  [ "$P0_ONLY" = "1" ] && [ "$P0" != "1" ] && continue

  printf "\n\033[1;32m==================  %s : %s  ==================\033[0m\n" "$ID" "$LABEL"
  echo "Normal=$(scaled "$NORMAL_N")  Swirl=$(scaled "$SWIRL_N")"
  pause "SET UP CONTENTS → $SETUP" || { echo "(skipped $ID)"; continue; }
  run_block "$LABEL" "$NORMAL_N" "normal"
  run_block "$LABEL" "$SWIRL_N"  "swirl"
done

# ── PHASE 2: swirl catch-up for solids + skipped-early swirls ──────────────
if [ "$SKIP_CATCHUP" != "1" ] && [ "$P0_ONLY" != "1" ]; then
  printf "\n\033[1;35m################  PHASE 2 — SWIRL CATCH-UP  ################\033[0m\n"
  pause "Swirl-only pass for the rigid solids + the early swirls you skipped. Ready?"
  for row in "${SWIRL_CATCHUP[@]}"; do
    IFS='|' read -r LABEL SWIRL_N SETUP <<< "$row"
    printf "\n\033[1;35m-----  SWIRL CATCH-UP : %s  -----\033[0m\n" "$LABEL"
    pause "SET UP CONTENTS → $SETUP" || { echo "(skipped $LABEL)"; continue; }
    run_block "$LABEL" "$SWIRL_N" "swirl"
  done
fi

echo -e "\n\033[1;32m✓ Run plan complete.\033[0m  CSVs in recordings/."
