"""Per-run provenance manifest (plan P0-1 / P0-2).

Every closed-loop run must be reproducible from its own record.  Until
now a run left a telemetry JSON and a scorecard, and nothing said WHICH
code, WHICH switch values, WHICH model and WHICH process produced them -
so "same configuration" was an assumption, and a difference between two
runs could not be attributed to the variable under test.

The manifest is that record:

* **code** - commit, branch, and the dirty working-tree paths (a run from
  a dirty tree is not reproducible from the commit alone).
* **switches** - every ``BEAMNG_*`` variable's RESOLVED value, including
  the ones declared in the code but not set (reported as ``None``, i.e.
  "the module default is in force", which is not the same as an explicit
  ``0``).
* **artifacts** - model/weight files by size, mtime and a short digest.
* **process** - pid, creation time and command line of the controller.
* **exclusivity** - any OTHER process that looks like a controller.  The
  game API accepts one client per port, so two controllers silently
  interleave two teleports on one vehicle; that contamination was found
  once already by comparing log mtimes, which only works when the runs
  happen to overlap to the second.  This asks the OS instead.

Pure logic, no game: every function takes its inputs (root, env, process
list) as arguments so it can be tested without a running BeamNG.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

# Every switch the code declares, so one that is merely UNSET still
# appears in the manifest instead of vanishing.  Grouped for readability;
# both groups are recorded identically.
BEHAVIOUR_SWITCHES = (
    "BEAMNG_ASYNC_HEADS",
    "BEAMNG_BODY_COVERAGE_GATE",
    "BEAMNG_CENTRE_HOLD",
    "BEAMNG_CONTROL_SUBSTEP",
    "BEAMNG_DASHED_RECOVERY",
    "BEAMNG_DRIVE_MODES",
    "BEAMNG_LANE_GEOM",
    "BEAMNG_LANE_HOLD_NONE_FRAMES",
    "BEAMNG_LANE_REF_SLEW",
    "BEAMNG_LATERAL_RL",
    "BEAMNG_LONG_PLAN",
    "BEAMNG_MARK_CLASS",
    "BEAMNG_NEARFIELD_CAM",
    "BEAMNG_NEARFIELD_EVERY_N",
    "BEAMNG_PAVED_LANE",
    "BEAMNG_REF_STABILITY",
    "BEAMNG_ROAD_SURFACE_GATE",
    "BEAMNG_SCHED_KEEPALIVE",
    "BEAMNG_SEG_PROB_GATE",
    "BEAMNG_SEG_ZONES",
    "BEAMNG_STEER_BLEND",
    "BEAMNG_YELLOW_FUSION",
)
ENVIRONMENT_VARS = (
    "BEAMNG_HOME",
    "BEAMNG_PORT",
    "BEAMNG_PROCESS_NAMES",
    "BEAMNG_PROCS",
    "BEAMNG_RUNTIME",
    "BEAMNG_TECH_HOME",
    "BEAMNG_TECH_PORT",
    "BEAMNG_TECH_USER",
    "BEAMNG_USER",
)
DECLARED_SWITCHES = BEHAVIOUR_SWITCHES + ENVIRONMENT_VARS

# Entry points that mean "a controller is driving a vehicle".
CONTROLLER_MARKERS = (
    "m5_fsd_drive",
    "m5_fsd_benchmark",
    "m5_drive_test",
    "m5_e2e_test",
    "fsd_drive",
)

# Files whose identity must be part of the record.  Missing entries are
# reported as absent, not skipped silently.
DEFAULT_ARTIFACTS = (
    "weights/tech_smallgrid.pt",
    "logs/m4_dqn/dqn_decision.zip",
)

# Digesting a multi-hundred-MB checkpoint on every run start is not worth
# it; above this the manifest records size + mtime and says the digest was
# skipped rather than pretending it was verified.
HASH_MAX_BYTES = 32 * 1024 * 1024


def switch_state(env: dict | None = None) -> dict:
    """Resolved value of every switch: ``None`` means "not set".

    ``None`` is the honest reading of "the module default is in force",
    and it is deliberately distinct from an explicit ``"0"``.
    """
    e = os.environ if env is None else env
    seen = {k: v for k, v in e.items() if str(k).startswith("BEAMNG_")}
    names = sorted(set(DECLARED_SWITCHES) | set(seen))
    return {name: seen.get(name) for name in names}


def _git(root: Path, *args: str) -> str | None:
    try:
        p = subprocess.run(("git",) + args, cwd=str(root),
                           capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    return p.stdout if p.returncode == 0 else None


def git_state(root) -> dict:
    """Commit / branch / dirty paths for the working tree at ``root``.

    ``dirty`` lists the paths git reports as modified or untracked: a run
    whose code does not match its commit cannot be reproduced from that
    commit, so the paths are part of the record.  An untracked
    ``.workbuddy-ai/`` is expected and harmless, but it is still listed
    rather than filtered - a filter would need a judgement this layer
    should not make.
    """
    root = Path(root)
    out: dict = {"root": str(root)}
    commit = _git(root, "rev-parse", "HEAD")
    out["commit"] = commit.strip() if commit else None
    out["commit_short"] = out["commit"][:7] if out["commit"] else None
    subject = _git(root, "log", "-1", "--pretty=%s")
    out["head_subject"] = subject.strip() if subject else None
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    out["branch"] = branch.strip() if branch else None
    status = _git(root, "status", "--porcelain")
    dirty: list[str] = []
    if status:
        for line in status.splitlines():
            if len(line) > 3:
                dirty.append(line[3:].strip())
    out["dirty"] = dirty
    out["dirty_count"] = len(dirty)
    # Ahead/behind needs an upstream; a local-only branch has none.
    counts = _git(root, "rev-list", "--left-right", "--count",
                  "@{u}...HEAD")
    if counts:
        parts = counts.split()
        if len(parts) == 2:
            out["behind"] = int(parts[0])
            out["ahead"] = int(parts[1])
    return out


def artifact_identity(paths=DEFAULT_ARTIFACTS, root=None) -> dict:
    """Size / mtime / short digest for each artifact that exists."""
    base = Path(root) if root is not None else Path(".")
    out: dict = {}
    for rel in paths:
        p = base / rel
        rec: dict = {"path": str(rel), "present": p.is_file()}
        if not rec["present"]:
            out[str(rel)] = rec
            continue
        try:
            st = p.stat()
            rec["size"] = int(st.st_size)
            rec["mtime"] = round(float(st.st_mtime), 3)
            if st.st_size <= HASH_MAX_BYTES:
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                rec["sha256"] = h.hexdigest()[:16]
            else:
                rec["sha256"] = None
                rec["hash_skipped"] = f"size > {HASH_MAX_BYTES} bytes"
        except Exception as exc:
            rec["error"] = str(exc)
        out[str(rel)] = rec
    return out


def process_identity(pid: int | None = None) -> dict:
    """pid / creation time / command line of this controller."""
    pid = os.getpid() if pid is None else int(pid)
    rec: dict = {"pid": pid, "wall_time": round(time.time(), 3)}
    try:
        import psutil
    except Exception:
        rec["error"] = "psutil unavailable"
        return rec
    try:
        p = psutil.Process(pid)
        rec["create_time"] = round(float(p.create_time()), 3)
        rec["name"] = p.name()
        rec["cmdline"] = list(p.cmdline())
    except Exception as exc:
        rec["error"] = str(exc)
    return rec


def own_process_lineage(pid: int | None = None) -> set:
    r"""This process and its ancestors.

    On Windows ``.venv\Scripts\python.exe`` is a shim: it spawns the real
    interpreter as a CHILD, so the shim is this process's parent and
    carries the same command line.  Excluding only ``os.getpid()``
    therefore reports your own launcher as a second controller and every
    run refuses to start - which is what happened on the first live
    attempt (pid 36168 was this process's own parent).

    Only the ancestors are excluded.  A second controller started from the
    same shell is a SIBLING and must still be reported.
    """
    me = os.getpid() if pid is None else int(pid)
    kin = {me}
    try:
        import psutil
        for parent in psutil.Process(me).parents():
            kin.add(parent.pid)
    except Exception:
        pass                       # no psutil, or an injected pid
    return kin


def other_controller_processes(pid: int | None = None,
                               procs=None, kin: set | None = None
                               ) -> list[dict]:
    """Other processes that look like a controller on this machine.

    The exclusivity evidence P0-2 asks for.  Two controllers on one port
    drive the same vehicle through two teleports, and the resulting run
    belongs to neither experiment - the earlier detection method compared
    log mtimes, which only fires when the two runs overlap to the second.
    ``procs`` lets a test inject a fake process list, ``kin`` the set of
    pids to treat as self.
    """
    me = os.getpid() if pid is None else int(pid)
    kin = own_process_lineage(pid) if kin is None else set(kin)
    found: list[dict] = []
    if procs is None:
        try:
            import psutil
        except Exception:
            return found
        try:
            procs = list(psutil.process_iter(
                attrs=["pid", "name", "cmdline", "create_time"]))
        except Exception:
            return found
    for p in procs or ():
        try:
            info = p.info if hasattr(p, "info") else dict(p)
        except Exception:
            continue
        try:
            ppid = int(info.get("pid"))
        except Exception:
            continue
        if ppid in kin:
            continue
        cmd = info.get("cmdline") or []
        joined = " ".join(str(c) for c in cmd)
        if not any(m in joined for m in CONTROLLER_MARKERS):
            continue
        rec = {"pid": ppid, "cmdline": [str(c) for c in cmd]}
        ct = info.get("create_time")
        if isinstance(ct, (int, float)):
            rec["create_time"] = round(float(ct), 3)
        found.append(rec)
    return sorted(found, key=lambda r: r["pid"])


def exclusivity(pid: int | None = None, procs=None) -> dict:
    """``{"ok": bool, "others": [...]}`` - is this the only controller?"""
    others = other_controller_processes(pid=pid, procs=procs)
    return {"ok": not others, "others": others}


def tech_version(home) -> str | None:
    """BeamNG.tech version, read off the install directory name.

    There is no version API on the connector, so the install path is the only
    place the version is written down - the default is
    ``G:\\BeamNG.tech.v0.38.5.0``.  A run from a different Tech build is a
    different experiment, and nothing in the telemetry said which build it
    was.  Returns ``None`` rather than guessing when the path carries no
    version.
    """
    if home is None:
        return None
    text = str(home)
    m = re.search(r"[vV](\d+(?:\.\d+)+)", text)
    return f"v{m.group(1)}" if m else None


def build_manifest(root=None, *, run: dict | None = None,
                   artifacts=DEFAULT_ARTIFACTS,
                   env: dict | None = None,
                   procs=None, pid: int | None = None,
                   tech_home=None) -> dict:
    """The whole record for one run.

    ``run`` carries what only the caller knows: scenario, spawn pose,
    goal, traffic/seed, warmup, vehicle id, session state.
    """
    base = Path(root) if root is not None else Path(".")
    home = tech_home
    if home is None and env:
        home = env.get("BEAMNG_TECH_HOME")
    return {
        "manifest_version": 1,
        "created_wall": round(time.time(), 3),
        "git": git_state(base),
        "switches": switch_state(env),
        "artifacts": artifact_identity(artifacts, root=base),
        "process": process_identity(pid),
        "exclusivity": exclusivity(pid=pid, procs=procs),
        "tech": {"home": str(home) if home is not None else None,
                 "version": tech_version(home)},
        "run": dict(run or {}),
    }
