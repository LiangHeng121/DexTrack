#!/bin/bash
# Monitor only the 3 multi-task reward experiments + baseline (singles already done).
# Polls every 5 min -> /tmp/multi_monitor.log; exits (notifies) on stall (15min frozen),
# any multi reaching ep5000 (halfway milestone), or 2h heartbeat.
cd /home/liangh/DexTrack/isaacgymenvs
OUT=/tmp/multi_monitor.log
declare -A LOG=(
  [base]="logs/grab_multiple_wuji_cubesmall/run/20260604_185559/screen.log"
  [s1]="logs/grab_multiple_wuji_fp1/wuji_cubesmall_fp1/20260606_101143/screen.log"
  [s23]="logs/grab_multiple_wuji_relax23/wuji_cubesmall_relax23/20260606_095846/screen.log"
  [s4]="logs/grab_multiple_wuji_fp4/wuji_cubesmall_fp4/20260606_104821/screen.log"
)
ORDER=(base s1 s23 s4)
declare -A PREV STALL
ep_of(){ grep -oE "epoch: [0-9]+/[0-9]+" "$1" 2>/dev/null | tail -1 | grep -oE "^epoch: [0-9]+" | grep -oE "[0-9]+"; }
best_of(){ grep -oE "ep_[0-9]+_rew_[-0-9.]+" "$1" 2>/dev/null | grep -oE "rew_[-0-9.]+" | sed 's/rew_//' | sort -g | tail -1; }
reason=""
for ((c=1;c<=24;c++)); do
  ts=$(date +%H:%M:%S); line="[$ts] "
  for k in "${ORDER[@]}"; do
    ep=$(ep_of "${LOG[$k]}"); best=$(best_of "${LOG[$k]}"); ep=${ep:-?}; best=${best:-?}
    line+="$k=ep$ep/b$best  "
    if [ "$ep" != "?" ]; then
      if [ "${PREV[$k]}" == "$ep" ]; then STALL[$k]=$(( ${STALL[$k]:-0} + 1 )); else STALL[$k]=0; fi
      PREV[$k]=$ep
      [ "${STALL[$k]:-0}" -ge 3 ] && reason="STALL $k frozen ep$ep (15min)"
      [ "$k" != "base" ] && [ "$ep" -ge 5000 ] && reason="MILESTONE $k reached ep$ep"
    fi
  done
  echo "$line" >> "$OUT"
  [ -n "$reason" ] && break
  sleep 300
done
[ -z "$reason" ] && reason="2h heartbeat"
echo "MULTI_MONITOR_EXIT: $reason"
for k in "${ORDER[@]}"; do printf "%-6s ep=%s best=%s\n" "$k" "$(ep_of "${LOG[$k]}")" "$(best_of "${LOG[$k]}")"; done
