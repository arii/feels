#!/usr/bin/env bash
#
# run_plan.sh — guided data-collection run sheet for the feels cup-content study.
#
# Walks the full experiment list one trial at a time. Pauses for you to:
#   • set up the contents before each experiment (change weight / fill / item)
#   • re-grasp / reposition the cup before each individual trial
#   • switch motion path (NORMAL vs SWIRL) between sub-blocks
#
# Each trial runs ONE command:
#   repeatable_grasp.py --object <label> --record --cycles 1 [--swirl-smooth]
# which drives the grasp cycle AND writes a phase-annotated CSV to recordings/.
# (NORMAL and SWIRL share the same object_label; they differ by the phase/
#  motion_type inside the CSV — SWIRL rows are tagged "SWIRL".)
#
# NOTE: this uses repeatable_grasp.py (motion + phase-tagged recording). It does
# NOT capture the extra auto-discovered channels that record_dataset.py adds
# (currents/velocities/depth) — unifying motion + rich capture needs the planned
# --macro mode. Ask if you want that before a big run.
#
# Controls at every pause:  [Enter] continue   s = skip rest of this experiment
#                           q = quit the plan
#
# Env knobs:
#   P0_ONLY=1      run only the priority-0 (minimum-viable) experiments
#   SCALE=0.25     multiply every trial count (quick pass); min 1 per active cell
#   START_FROM=E9  skip experiments until this ID
#   DRY_RUN=1      print the commands instead of running them
#
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
[ -d .venv ] && source .venv/bin/activate 2>/dev/null || true

P0_ONLY="${P0_ONLY:-0}"
SCALE="${SCALE:-1}"
START_FROM="${START_FROM:-}"
DRY_RUN="${DRY_RUN:-0}"
STARTED=0; [ -z "$START_FROM" ] && STARTED=1

# ── experiment table ──────────────────────────────────────────────────────
# ID | LABEL | NORMAL_N | SWIRL_N | P0(1/0) | SETUP INSTRUCTION
EXPERIMENTS=(
  "E1|empty_cup|30|15|1|Empty cup, lid ON, nothing inside (baseline/tare)"
  "E2|solid_020g|20|0|0|Rigid 20 g weight FIXED (taped so it can't shift), lid ON"
  "E3|solid_050g|20|0|1|Rigid 50 g weight FIXED, lid ON"
  "E4|solid_100g|20|0|1|Rigid 100 g weight FIXED, lid ON"
  "E5|solid_200g|20|0|1|Rigid 200 g weight FIXED, lid ON"
  "E6|solid_350g|20|0|0|Rigid 350 g weight FIXED, lid ON"
  "E7|solid_500g|20|0|0|Rigid 500 g weight FIXED, lid ON"
  "E8|water_025pct|20|20|0|Water at 25% fill, lid ON"
  "E9|water_050pct|20|20|1|Water at 50% fill, lid ON"
  "E10|water_075pct|20|20|0|Water at 75% fill, lid ON"
  "E11|water_100pct|20|20|0|Water at 100% fill, lid ON"
  "E12|soap_050pct|20|20|1|Dish soap at 50% fill, lid ON"
  "E13|oil_050pct|20|20|0|Cooking oil at 50% fill, lid ON"
  "E14|honey_050pct|15|15|0|Honey at 50% fill, lid ON"
  "E15|rice_100g|20|0|0|100 g rice (fine granular), loose, lid ON"
  "E16|beads_100g|20|0|0|100 g beads/pellets (medium), loose, lid ON"
  "E17|candy_100g|20|15|1|100 g fun-size candy, loose, lid ON"
  "E18|screws_100g|20|0|0|100 g metal screws (hard), loose, lid ON"
  "E19|candy_050g|20|0|0|50 g candy, loose, lid ON"
  "E20|candy_200g|20|0|0|200 g candy, loose, lid ON"
  "E21|solid_150g|20|15|1|CONFOUND: 150 g rigid weight FIXED, lid ON"
  "E22|water_150g|20|20|1|CONFOUND: ~150 mL water (~150 g), lid ON"
  "E23|candy_150g|20|20|1|CONFOUND: 150 g candy, loose, lid ON"
  "E24|soap_150g|20|20|1|CONFOUND: 150 g dish soap, lid ON"
  "E25|candy_horizontal_100g|15|15|0|100 g candy spread LOW/WIDE (horizontal), lid ON"
  "E26|water_vertical_100g|15|15|0|~100 g water in a TALL column, lid ON"
)

# ── helpers ───────────────────────────────────────────────────────────────
scaled() { awk -v n="$1" -v s="$SCALE" 'BEGIN{ if(n<=0){print 0; exit} v=int(n*s+0.999); if(v<1)v=1; print v }'; }

pause() { # $1 = message ; sets REPLY; returns 1 to skip, 2 to quit
  local msg="$1"
  printf "\n\033[36m%s\033[0m\n" "$msg"
  read -r -p "   [Enter]=go  s=skip experiment  q=quit > " ans
  case "$ans" in
    s|S) return 1 ;;
    q|Q) echo "— quitting —"; exit 0 ;;
    *)   return 0 ;;
  esac
}

run_trial() { # $1 label, $2 trial#, $3 total, $4 path(normal|swirl)
  local label="$1" i="$2" tot="$3" path="$4" extra=""
  [ "$path" = "swirl" ] && extra="--swirl-smooth"
  pause "   ▶ $label  [$path]  trial $i/$tot — RE-GRASP: release & reposition the cup in the gripper zone." || return 1
  local cmd="$PY repeatable_grasp.py --object \"$label\" --record --cycles 1 $extra"
  if [ "$DRY_RUN" = "1" ]; then echo "     DRY: $cmd"; else eval "$cmd"; fi
  return 0
}

run_block() { # $1 label, $2 count, $3 path
  local label="$1" n; n="$(scaled "$2")"; local path="$3"
  [ "$n" -eq 0 ] && return 0
  printf "\n\033[33m--- %s : %s path : %d trials ---\033[0m\n" "$label" "$path" "$n"
  if [ "$path" = "swirl" ]; then
    pause "   SWIRL path for '$label' — contents unchanged; confirm cup seated for orbital motion." || return 0
  fi
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
  Each trial drives a grasp cycle and records to recordings/.
============================================================
BANNER
echo "P0_ONLY=$P0_ONLY  SCALE=$SCALE  START_FROM=${START_FROM:-none}  DRY_RUN=$DRY_RUN"
pause "Ready to begin?"

# ── main loop ─────────────────────────────────────────────────────────────
for row in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r ID LABEL NORMAL_N SWIRL_N P0 SETUP <<< "$row"
  [ "$STARTED" -eq 0 ] && { [ "$ID" = "$START_FROM" ] && STARTED=1 || continue; }
  [ "$P0_ONLY" = "1" ] && [ "$P0" != "1" ] && continue

  printf "\n\033[1;32m==================  %s : %s  ==================\033[0m\n" "$ID" "$LABEL"
  echo "Normal=$(scaled "$NORMAL_N")  Swirl=$(scaled "$SWIRL_N")  ${P0:+(P0=$P0)}"
  pause "SET UP CONTENTS → $SETUP" || { echo "(skipped $ID)"; continue; }

  run_block "$LABEL" "$NORMAL_N" "normal"
  run_block "$LABEL" "$SWIRL_N"  "swirl"
done

echo -e "\n\033[1;32m✓ Run plan complete.\033[0m  CSVs in recordings/."
