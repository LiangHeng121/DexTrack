#!/bin/bash
# Watchdog: poll the 7 training logs every 10 min. If any training's epoch is
# stuck for 2 consecutive checks (=20 min) AND not finished, exit reporting it
# (so the agent gets notified and resumes it). Exit after MAXITER with "ok".
declare -A LOG=(
  [G1_wuji_cube_multi]=/tmp/resume_wuji_cube.log
  [G2_wuji_flute_multi]=/tmp/resume_wuji_flute.log
  [G3_allegro_cube_multi]=/tmp/resume_allegro_cube.log
  [G4_allegro_flute_multi]=/tmp/resume_allegro_flute.log
  [G5_allegro_comb_multi]=/tmp/resume_allegro_combined.log
  [G6_wuji_flute_offset]=/tmp/wuji_flute_offset_train.log
  [G7_wuji_comb_multi]=/tmp/resume_wuji_combined.log
)
declare -A MAXEP=( [G6_wuji_flute_offset]=1000 )  # others default 10000
declare -A PREV STALL
MAXITER=36   # 36 * 10min = 6h then exit for restart
ep_of(){ grep -oE "epoch: [0-9]+/" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+"; }
for ((i=1;i<=MAXITER;i++)); do
  sleep 600
  for k in "${!LOG[@]}"; do
    cur=$(ep_of "${LOG[$k]}"); max=${MAXEP[$k]:-10000}
    [ -z "$cur" ] && cur=NONE
    [ "$cur" = "$max" ] && continue        # finished
    if [ "$cur" = "${PREV[$k]}" ]; then STALL[$k]=$(( ${STALL[$k]:-0} + 1 )); else STALL[$k]=0; fi
    PREV[$k]=$cur
    if [ "${STALL[$k]}" -ge 2 ]; then
      echo "DEAD: $k stuck at epoch $cur (log ${LOG[$k]}) after $((i*10))min"; exit 0
    fi
  done
done
echo "WATCHDOG_OK: ${MAXITER}0min elapsed, all 7 advancing, restart me"
