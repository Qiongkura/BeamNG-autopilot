"""reCAPTCHA-style grid labeler: click the frames that contain lane paint.

Opens a local browser page with a 3x3 grid of frames; the human clicks every
tile whose image contains painted lane markings (车道线/边线/箭头/斑马线),
presses 提交, and the next batch loads.  "全部无标线" marks the whole batch
negative in one click (negatives matter as much as positives).  Each submit
appends one JSONL record per image:

    {"ts": "...", "image": "rel/name.jpg", "path": "abs path",
     "has_line": 0|1, "strategy": "random"|"score"}

Re-running with the same --src/--out resumes: labeled frames are skipped.
Feedback loop: --strategy score --scores <json {frame: score}> prioritizes
the highest-score pending frames (with epsilon exploration), so a model can
decide which frames the human sees next - active learning / reward-label
loop for the lane-line presence signal.

Usage:
    .venv\\Scripts\\python.exe scripts\\m5_line_grid_labeler.py \\
        --src logs/m5_seg/manual_capture -r
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.labeling.grid_labeler import (
    DEFAULT_TASK, GridLabelApp, LabelStore, scan_images, serve)


def main() -> int:
    ap = argparse.ArgumentParser(description="reCAPTCHA 式标线宫格标注")
    ap.add_argument("--src", type=str, required=True,
                    help="图片目录（相对名 = 目录内相对路径）")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="递归扫描子目录")
    ap.add_argument("--out", type=str, default=None,
                    help="标签 JSONL 输出路径（默认 logs/labeling/，可续标）")
    ap.add_argument("--batch-size", type=int, default=9)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--task", type=str, default=DEFAULT_TASK)
    ap.add_argument("--strategy", type=str, default="random",
                    choices=("random", "score"),
                    help="random 均匀抽样；score 按分数优先（需 --scores）")
    ap.add_argument("--scores", type=str, default=None,
                    help="JSON {帧名: 分数}，score 策略的排序依据")
    ap.add_argument("--include-labeled", action="store_true",
                    help="不跳过已标注帧（复核模式）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    paths = scan_images(src, recursive=args.recursive)
    if not paths:
        print(f"[grid-label] no images under {src}")
        return 1

    out = (Path(args.out) if args.out
           else config.LOGS_DIR / "labeling" / "line_grid_labels.jsonl")
    store = LabelStore(out)

    scores = None
    strategy = args.strategy
    if args.scores:
        scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
        if not isinstance(scores, dict):
            print("[grid-label] --scores must be a JSON object {name: score}")
            return 1
    elif strategy == "score":
        print("[grid-label] --strategy score without --scores, "
              "falling back to random")
        strategy = "random"

    app = GridLabelApp(src, paths, store, task=args.task,
                       batch_size=args.batch_size, cols=args.cols,
                       strategy=strategy, scores=scores,
                       include_labeled=args.include_labeled, seed=args.seed)
    prog = app.progress()
    print(f"[grid-label] {prog['total']} frames | already labeled "
          f"{prog['labeled']} | pending {prog['total'] - prog['labeled']}")
    print(f"[grid-label] labels -> {out}")

    httpd = serve(app, host=args.host, port=args.port)
    url = f"http://{args.host}:{httpd.server_address[1]}/"
    print(f"[grid-label] serving on {url}  (Ctrl+C 退出，标签实时落盘)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    prog = app.progress()
    print(f"[grid-label] done: labeled {prog['labeled']}/{prog['total']} "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
