#!/bin/bash
# Background monitor for the #1/#2/#3/#4 wuji cubesmall reward experiments + baseline.
# Polls every 5 min; appends a status table to $OUT; exits (-> notifies) when any single
# run finishes (ep>=1000), any run stalls (epoch frozen 3 polls = 15min), or 3h elapse.
cd /home/liangh/DexTrack/isaacgymenvs
OUT=/tmp/exp_monitor.log
declare -A LOG=(
  [base_multi]="logs/grab_multiple_wuji_cubesmall/run/20260604_185559/screen.log"
  [s1_single]="logs/grab_single_wuji_fp1/ori_grab_s2_cubesmall_inspect_1/20260606_101143/screen.log"
  [s1_multi]="logs/grab_multiple_wuji_fp1/wuji_cubesmall_fp1/20260606_101143/screen.log"
  [s23_single]="logs/grab_single_wuji_relax23/ori_grab_s2_cubesmall_inspect_1/20260606_095846/screen.log"
  [s23_multi]="logs/grab_multiple_wuji_relax23/wuji_cubesmall_relax23/20260606_095846/screen.log"
  [s4_single]="logs/grab_single_wuji_fp4/ori_grab_s2_cubesmall_inspect_1/20260606_104821/screen.log"
  [s4_multi]="logs/grab_multiple_wuji_fp4/wuji_cubesmall_fp4/20260606_104821/screen.log"
)
ORDER=(base_multi s1_single s1_multi s23_single s23_multi s4_single s4_multi)
declare -A PREV STALL
MAX_CYCLES=36          # 36 * 300s = 3h fallback
ep_of(){ grep -oE "epoch: [0-9]+/[0-9]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1; }
best_of(){ grep -oE "ep_[0-9]+_rew_[-0-9.]+" "$1" 2>/dev/null | grep -oE "rew_[-0-9.]+" | sed 's/rew_//' | sort -g | tail -1; }

reason=""
for ((c=1;c<=MAX_CYCLES;c++)); do
  ts=$(date +%H:%M:%S)
  line="[$ts] "
  for k in "${ORDER[@]}"; do
    L=${LOG[$k]}; ep=$(ep_of "$L"); best=$(best_of "$L"); ep=${ep:-?}; best=${best:-?}
    line+="$k=ep$ep/best$best  "
    # stall detection (only for live runs that had progress)
    if [ "$ep" != "?" ]; then
      if [ "${PREV[$k]}" == "$ep" ]; then STALL[$k]=$(( ${STALL[$k]:-0} + 1 )); else STALL[$k]=0; fi
      PREV[$k]=$ep
      if [ "${STALL[$k]:-0}" -ge 3 ]; then reason="STALL $k frozen at ep$ep (15min)"; fi
      # single runs finish at 1000
      if [[ "$k" == *single* ]] && [ "$ep" -ge 1000 ]; then reason="DONE $k reached ep$ep"; fi
    fi
  done
  echo "$line" >> "$OUT"
  [ -n "$reason" ] && break
  sleep 300
done
[ -z "$reason" ] && reason="3h heartbeat"
echo "MONITOR_EXIT: $reason"
echo "=== final status ==="
for k in "${ORDER[@]}"; do
  L=${LOG[$k]}; printf "%-12s ep=%s best=%s\n" "$k" "$(ep_of "$L")" "$(best_of "$L")"
done
