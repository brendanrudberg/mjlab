#!/usr/bin/env bash
# Standard C1 evaluation of an unpenalized and a penalized run at one iteration.
#
# Usage (from the repo root):
#   src/mjlab/scripts/standard_eval.sh <unpenalized_run_dir> <penalized_run_dir> \
#     [iteration=3200] [out_dir]
#
# For each run: a 20 s video at every command, and load_probe across all
# commands. Then compare_policies on both runs side by side, and eval_report for
# tables (tables.md + CSV) and figures. Everything lands in out_dir, by default
# logs/rsl_rl/c1_velocity/eval_iter<iteration>_<date-time>/.
set -euo pipefail

UNPEN=${1:?unpenalized run dir}
PEN=${2:?penalized run dir}
ITER=${3:-3200}
OUT=${4:-logs/rsl_rl/c1_velocity/eval_iter${ITER}_$(date +%Y-%m-%d_%H-%M)}
CMDS="-2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0"
LABELS=(unpenalized penalized)

mkdir -p "$OUT/videos" "$OUT/logs"
RUNS=("$UNPEN" "$PEN")
for k in 0 1; do
  RUN=${RUNS[$k]%/}
  CK="$RUN/model_${ITER}.pt"
  # Wait for a run that is still training (checkpoints are written atomically
  # enough that a 30 s settle after the file appears is plenty).
  if [[ ! -f "$CK" ]]; then
    echo "waiting for $CK"
    until [[ -f "$CK" ]]; do sleep 60; done
    sleep 30
  fi
  label=${LABELS[$k]}
  echo "== $label: $CK"
  echo "$label: $CK" >> "$OUT/runs.txt"
  head -c 200 "$RUN/git/mjlab.diff" | grep -A1 "git commit" | tail -1 \
    | sed "s/^/  commit /" >> "$OUT/runs.txt" || true

  for c in $CMDS; do
    tag=$(printf '%+.1f' "$c")
    echo "   video at $tag m/s"
    MUJOCO_GL=egl uv run python src/mjlab/scripts/record_video.py "$CK" 1000 \
      --cmd "$c" > "$OUT/logs/video_${label}_${tag}.txt" 2>&1
    cp "$RUN/videos/model_${ITER}/cmd_${tag}/rl-video-step-0.mp4" \
      "$OUT/videos/${label}_${tag}.mp4"
    sed -n '/^commanded vx/,$p' "$OUT/logs/video_${label}_${tag}.txt" \
      > "$OUT/videos/${label}_${tag}.txt"
  done

  echo "   load_probe"
  # shellcheck disable=SC2086
  uv run src/mjlab/scripts/load_probe.py "$CK" $CMDS --envs 64 --steps 500 \
    > "$OUT/logs/load_probe_${label}.txt" 2>&1
  cp "$RUN/loads_model_${ITER}.csv" "$OUT/loads_${label}.csv"
  sed -n '/robot mass/,$p' "$OUT/logs/load_probe_${label}.txt" \
    > "$OUT/load_probe_${label}.txt"
done

echo "== compare_policies"
# shellcheck disable=SC2086
uv run src/mjlab/scripts/compare_policies.py "${UNPEN%/}/model_${ITER}.pt" \
  "${PEN%/}/model_${ITER}.pt" --cmds $CMDS --envs 64 --steps 500 \
  --csv "$OUT/compare.csv" --samples "$OUT/samples.npz" \
  > "$OUT/logs/compare.txt" 2>&1
sed -n '/^\[0\]/,$p' "$OUT/logs/compare.txt" > "$OUT/compare.txt"

echo "== eval_report"
uv run src/mjlab/scripts/eval_report.py "$OUT" --labels "${LABELS[@]}"

echo "done: $OUT"
