#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BeamNG.tech 构建多线程下载器（纯标准库，支持断点续传）

用法:
    python download_beamng_tech.py --url <链接> [--out <保存路径>] [--threads N]

--url 必填：下载链接由构建提供方给出，本脚本不内置任何下载地址。
默认保存到 ~\\Downloads\\BeamNG.tech.v0.38.5.0.zip
中断后重新运行同一命令即可断点续传（已下完的分段会跳过）。
"""

import argparse
import base64
import hashlib
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URL = ""
DEFAULT_DEST = str(Path.home() / "Downloads" / "BeamNG.tech.v0.38.5.0.zip")
DEFAULT_MD5 = ""
SEGMENT_SIZE = 16 * 1024 * 1024  # 每个分段 16MB
MAX_THREADS = 16
MAX_RETRIES = 8
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class Progress:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.lock = threading.Lock()
        self.start = time.time()
        self.last = 0.0

    def add(self, n: int):
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last >= 0.5:
                self.render(now)
                self.last = now

    def render(self, now: float | None = None):
        now = now or time.time()
        elapsed = now - self.start
        speed = self.done / elapsed / 1024 / 1024 if elapsed > 0 else 0.0
        pct = self.done * 100.0 / self.total if self.total else 0.0
        eta_min = (
            (self.total - self.done) / (self.done / elapsed) / 60
            if self.done > 0 and elapsed > 0
            else 0.0
        )
        line = (
            "\r[%5.1f%%] %6.1f / %6.1f MB | %6.2f MB/s | ETA %5.1f min"
            % (pct, self.done / 1048576, self.total / 1048576, speed, eta_min)
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self):
        self.render()
        sys.stdout.write("\n")
        sys.stdout.flush()


def build_request(url: str, headers: dict | None = None) -> Request:
    merged = {"User-Agent": UA}
    if headers:
        merged.update(headers)
    return Request(url, headers=merged)


def fetch_size(url: str) -> tuple[int, bool]:
    """返回 (总大小, 是否支持 Range)。优先 HEAD，失败则回退到 Range GET。"""
    try:
        with urlopen(build_request(url, {"Range": "bytes=0-0"}), timeout=30) as r:
            cr = r.headers.get("Content-Range", "")
            total = int(cr.split("/")[-1]) if cr else int(r.headers.get("Content-Length", 0))
            return total, bool(cr)
    except HTTPError as e:
        if e.code == 416:  # 服务器不支持 Range 但文件已存在
            return 0, False
        raise


def download_segment(url: str, part_path: str, start: int, end: int, progress: Progress) -> int:
    """下载 [start, end] 区间到 part_path。返回实际写入字节数。"""
    written = 0
    offset = start
    for attempt in range(MAX_RETRIES):
        try:
            with open(part_path, "r+b") as f:
                f.seek(offset)
                with urlopen(
                    build_request(url, {"Range": "bytes=%d-%d" % (offset, end)}),
                    timeout=60,
                ) as r:
                    remaining = end - offset + 1
                    while remaining > 0:
                        chunk = r.read(min(256 * 1024, remaining))
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)
                        offset += len(chunk)
                        remaining -= len(chunk)
                        progress.add(len(chunk))
                return written
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError("segment %d-%d failed: %s" % (start, end, exc))
            time.sleep(2 * (attempt + 1))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="BeamNG.tech 多线程下载器")
    parser.add_argument("--url", required=True,
                        help="下载链接（由构建提供方给出，必填）")
    parser.add_argument("--out", default=DEFAULT_DEST, help="保存路径（含文件名）")
    parser.add_argument("--threads", type=int, default=MAX_THREADS, help="并发线程数（默认 16）")
    parser.add_argument("--md5", default=DEFAULT_MD5, help="期望的 MD5（base64/hex 均可，下载后校验）")
    args = parser.parse_args()

    # 重定向输出时保证 UTF-8，避免日志乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    dest = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    free_gb = shutil.disk_usage(os.path.dirname(dest)).free / 1073741824
    print("[*] 目标盘剩余空间: %.1f GB" % free_gb)
    if free_gb < 30:
        print("[!] 警告: BeamNG.tech 解压后需要 20GB+ 空间，剩余空间可能不足。")

    print("[*] 连接服务器获取文件信息 ...")
    try:
        total, ranges_ok = fetch_size(args.url)
    except Exception as exc:
        print("[!] 无法获取文件信息: %s" % exc)
        print("    请确认链接仍有效（邮件里的下载链接有时效）。")
        return 1

    if not total:
        print("[!] 服务器未返回文件大小，链接可能已失效。")
        return 1

    print("[*] 文件大小: %.2f GB, Range 支持: %s" % (total / 1073741824, ranges_ok))
    if not ranges_ok:
        print("[!] 服务器不支持分段下载，退回单线程（可能仍比 Edge 快）。")

    # 分段
    seg_count = max(1, (total + SEGMENT_SIZE - 1) // SEGMENT_SIZE)
    segs = []
    for i in range(seg_count):
        start = i * SEGMENT_SIZE
        end = min(start + SEGMENT_SIZE - 1, total - 1)
        segs.append((i, start, end))

    # 跳过已下载完成的分段（断点续传）
    pending = []
    for i, start, end in segs:
        part = "%s.part.%05d" % (dest, i)
        done_len = os.path.getsize(part) if os.path.exists(part) else 0
        need = end - start + 1
        if done_len >= need:
            continue  # 该分段已完成
        if done_len > 0:
            start += done_len  # 从断点继续
        pending.append((part, start, end))

    if not pending:
        print("[*] 所有分段都已下载完成，直接合并。")
    else:
        print("[*] 待下载分段: %d / %d，线程数: %d" % (len(pending), seg_count, args.threads))
        progress = Progress(total)
        progress.render()

        def worker(item):
            part, start, end = item
            if not os.path.exists(part):
                with open(part, "wb"):
                    pass
            download_segment(args.url, part, start, end, progress)

        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = [pool.submit(worker, item) for item in pending]
            failed = 0
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    failed += 1
                    print("\n[!] 分段下载失败: %s" % exc)
        progress.finish()
        if failed:
            print("[!] 有 %d 个分段失败，重新运行同一命令可续传。" % failed)
            return 1

    # 合并
    print("[*] 合并分段 -> %s ..." % dest)
    with open(dest, "wb") as out:
        for i, _, _ in segs:
            part = "%s.part.%05d" % (dest, i)
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out, 1024 * 1024)
    for i, _, _ in segs:
        part = "%s.part.%05d" % (dest, i)
        try:
            os.remove(part)
        except OSError:
            pass

    print("[✓] 下载完成: %s (%.2f GB)" % (dest, os.path.getsize(dest) / 1073741824))

    # MD5 校验
    if args.md5:
        expected = args.md5.strip()
        print("[*] 校验 MD5 ...")
        h = hashlib.md5()
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        actual_hex = h.hexdigest()
        candidates = {expected.lower(), expected}
        try:
            candidates.add(base64.b64decode(expected + "==").hex())
        except Exception:
            pass
        if actual_hex in candidates:
            print("[✓] MD5 校验通过")
        else:
            print("[!] MD5 不匹配: 期望 %s, 实际 %s" % (expected, actual_hex))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
