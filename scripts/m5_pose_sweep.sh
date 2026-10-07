#!/usr/bin/env bash
# 冻结位姿扫描（E5 用；可复用于任何"多姿态、无人值守、一次一实例"的驾驶测量）：
# 端口等待 + 端口轮换 + 失败退避 + 残留清理。
#
# 为什么（2026-10-07 实测）：
#  * 连续快速启动会让握手失败（`Connecting to the simulator failed`）——上一个实例的端口没释放；
#  * 外部 timeout 会让驱动 finally 不执行 → 孤儿实例（已改用驱动内 --max-wall-s）；
#  * 因此每次启动前：① 确认没有我们的游戏进程残留；② 确认端口空闲（否则等）；
#    ③ 每次尝试换一个端口（64257 + attempt），避开 TIME_WAIT；④ 失败后按 30/60/90 s 退避。
# 预注册：docs/T16_E5_POSE_SWEEP_PREREG_20261007.md（含 150 s 墙钟上限与"未完成"分类）
set -u
cd "$(dirname "$0")/.." || exit 1

OUT=logs/experiments/e5_pose_sweep_20261007
mkdir -p "$OUT"
PY=.venv/Scripts/python.exe
MODEL=logs/m5_seg/seg_model_hand/best_task.pt
AX=779.7; AY=735.6; AYAW=-13.0
BASE_PORT=64257

read -r NX NY <<<"$($PY -c "
import math
y = math.radians($AYAW)
print(-math.sin(y), math.cos(y))
")"

port_free() {   # $1=port -> 0 空闲
  $PY -c "
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('127.0.0.1', int(sys.argv[1]))); print('free')
except OSError:
    print('busy')
finally:
    s.close()
" "$1" | grep -q free
}

cleanup_games() {  # 只清我们自己的游戏进程（所有权差集语义）
  $PY -c "
from beamng_autopilot.experiments.collection import game_pids
import subprocess, time
for rnd in range(4):
    pids = sorted(game_pids() or [])
    if not pids: break
    for p in pids: subprocess.run(['taskkill','/F','/PID',str(p)], capture_output=True)
    time.sleep(4)
print('games:', len(game_pids() or []))
"
}

wait_port() {   # $1=port, 最多等 150 s
  for _ in $(seq 1 30); do
    port_free "$1" && return 0
    sleep 5
  done
  return 1
}

run_pose() {    # x y yaw out -> 0 有遥测 / 3 未获得放置 / 1 失败
  local X="$1" Y="$2" YAW="$3" OUTJ="$4"
  local attempt=1
  while [ $attempt -le 3 ]; do
    local port=$((BASE_PORT + attempt))
    cleanup_games >/dev/null
    if ! wait_port "$port"; then
      echo "    [attempt $attempt] 端口 $port 150 s 内未释放"; sleep 30; attempt=$((attempt+1)); continue
    fi
    BEAMNG_TECH_PORT="$port" $PY scripts/m5_fsd_drive.py --runtime tech --map italy \
      --teleport "$X" "$Y" "$YAW" --goal 868.3 744.9 \
      --lane-mode sensor --strict --seconds 25 --speed 6.0 --max-wall-s 150 \
      --seg-model "$MODEL" --out "$OUTJ" >/tmp/e5_pose_run.log 2>&1
    local rc=$?
    if [ -f "$OUTJ" ]; then tail -2 /tmp/e5_pose_run.log; return 0; fi
    if [ "$rc" = "3" ]; then
      echo "    unplaceable（rc=3：未在上限内获得放置）" | tee -a "$OUT/unplaceable.txt"
      return 3
    fi
    echo "    [attempt $attempt] rc=$rc：$(tail -1 /tmp/e5_pose_run.log)"
    cleanup_games >/dev/null
    sleep $((30 * attempt))
    attempt=$((attempt+1))
  done
  return 1
}

ok=0; unplace=0; fail=0
for d in 0.0 0.5 1.0; do
  for dps in 0 5 10; do
    tag="d${d}_y${dps}"
    X=$($PY -c "print(f'{$AX + $d * $NX:.3f}')")
    Y=$($PY -c "print(f'{$AY + $d * $NY:.3f}')")
    YAW=$($PY -c "print(f'{$AYAW + $dps:.2f}')")
    if [ -f "$OUT/$tag.json" ]; then echo "=== [skip] $tag 已有遥测 ==="; ok=$((ok+1)); continue; fi
    echo "=== pose $tag  X=$X Y=$Y YAW=$YAW  ($(date +%H:%M:%S)) ==="
    run_pose "$X" "$Y" "$YAW" "$OUT/$tag.json"
    case $? in
      0) ok=$((ok+1));;
      3) unplace=$((unplace+1));;
      *) fail=$((fail+1)); echo "    FAILED $tag";;
    esac
  done
done

echo "=== 收尾 $(date +%H:%M:%S)：完成 $ok / 未获得放置 $unplace / 失败 $fail ==="
cleanup_games
ls "$OUT"/*.json 2>/dev/null | wc -l
