#!/usr/bin/env python3
"""Symmetric-memory rendezvous leak probe (pytorch/pytorch#199896).

Checks whether a *rejected* CUDA SymmetricMemory rendezvous leaks the exported
allocation. ``make_peer_alloc_info()`` in
``torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu`` exports the
block with ``cuMemExportToShareableHandle`` before it validates the gathered
rendezvous requests, and only closes the exported fd after a successful import
loop and barrier. Any throw in between leaves the fd open, and an open exported
fd keeps the physical allocation alive after the tensor is freed.

Each scenario allocates, rendezvouses (catching the error), frees, and tracks
free device memory (``cudaMemGetInfo``) and this process's open fds:

    ok             control: own GPU per rank, same size -> rendezvous succeeds
    size           odd ranks allocate 2x -> rejected by the block_size check
    overlap        every rank on cuda:0 -> rejected by the overlapping-devices
                   check (the issue's repro)
    overlap-close  as overlap, but close the /dev/nvidia* fds each failure left
                   open; if the leak disappears, those fds are what pin it

``--handle posix`` forces the POSIX-fd path on fabric-capable machines (H100/
GB200 with an IMEX channel), which otherwise use fabric handles and do not leak
on these checks. torch has no knob for this, so the rank re-execs itself with a
tiny LD_PRELOAD shim (built once with ``cc``) that makes
``nvmlDeviceGetGpuFabricInfoV`` report NOT_SUPPORTED.

    torchrun --nproc_per_node=2 recipes/symm_mem_rendezvous_leak.py
    torchrun --nproc_per_node=2 recipes/symm_mem_rendezvous_leak.py --handle posix

Rank 0 writes report.md / report.json plus per-rank torch logs to --outdir.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass

MiB = 1 << 20
ISSUE = "https://github.com/pytorch/pytorch/issues/199896"
SHIM_ENV = "SYMM_MEM_LEAK_PROBE_SHIM"

SHIM_SRC = r"""
// Hide GPU fabric support from torch so CUDASymmetricMemory uses POSIX fds.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
static int no_fabric(void* dev, void* info) { (void)dev; (void)info; return 3; }
typedef void* (*dlsym_t)(void*, const char*);
void* dlsym(void* handle, const char* name) {
  static dlsym_t real;
  static const char* versions[] = {"GLIBC_2.34", "GLIBC_2.17", "GLIBC_2.2.5"};
  for (unsigned i = 0; !real && i < sizeof(versions) / sizeof(*versions); ++i)
    real = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", versions[i]);
  if (name && strcmp(name, "nvmlDeviceGetGpuFabricInfoV") == 0) return (void*)no_fabric;
  return real(handle, name);
}
"""


@dataclass(frozen=True)
class Scenario:
    name: str
    overlap: bool
    size_mismatch: bool
    close_leaked_fds: bool
    what: str


SCENARIOS = {
    s.name: s
    for s in (
        Scenario("ok", False, False, False, "control: own GPU per rank, same size"),
        Scenario("size", False, True, False, "odd ranks allocate 2x (block_size check)"),
        Scenario("overlap", True, False, False, "all ranks on cuda:0 (overlapping devices check)"),
        Scenario(
            "overlap-close", True, False, True, "overlap, then close the fds each failure left open"
        ),
    )
}


def reexec_with_posix_shim() -> None:
    """Restart this rank under the no-fabric shim. Must run before torch is imported."""
    if os.environ.get(SHIM_ENV) == "1":
        return
    cache_root = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache = os.path.join(cache_root, "nv-recipes")
    os.makedirs(cache, exist_ok=True)
    digest = hashlib.sha1(SHIM_SRC.encode()).hexdigest()[:10]
    so = os.path.join(cache, f"no_fabric_shim-{platform.machine()}-{digest}.so")
    if not os.path.exists(so):
        cc = shutil.which("cc") or shutil.which("gcc")
        if cc is None:
            raise SystemExit("--handle posix needs cc or gcc to build the LD_PRELOAD shim")
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            src, out = os.path.join(tmp, "shim.c"), os.path.join(tmp, "shim.so")
            with open(src, "w") as fh:
                fh.write(SHIM_SRC)
            subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", out, src, "-ldl"], check=True)
            os.replace(out, so)  # atomic: every local rank races to build it
    env = dict(os.environ, **{SHIM_ENV: "1"})
    env["LD_PRELOAD"] = ":".join(p for p in (so, os.environ.get("LD_PRELOAD")) if p)
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)


def fd_table() -> dict[int, str]:
    out = {}
    for name in os.listdir("/proc/self/fd"):
        with contextlib.suppress(OSError):
            out[int(name)] = os.readlink(f"/proc/self/fd/{name}")
    return out


def slope(values: list[float]) -> float:
    """Growth per iteration, ignoring whatever the first iteration paid once."""
    return (values[-1] - values[0]) / (len(values) - 1) if len(values) > 1 else 0.0


def run_scenario(
    sc: Scenario, args: argparse.Namespace, rank: int, local_rank: int, dist, torch, symm_mem
) -> dict:
    device = torch.device("cuda", 0 if sc.overlap else local_rank)
    torch.cuda.set_device(device)
    # Absorb one-time costs (context on this device, allocator fds) before the baseline.
    symm_mem.empty(1, device=device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dist.barrier()

    free0, _ = torch.cuda.mem_get_info(device)
    nfd0 = len(fd_table())
    mult = 2 if sc.size_mismatch and rank % 2 else 1
    numel = args.mib * MiB // 4 * mult
    errors: list[str] = []
    leaked_targets: dict[str, int] = {}
    closed = 0
    drops, fds = [], []
    for _ in range(args.iters):
        t = symm_mem.empty(numel, dtype=torch.float32, device=device)
        before = fd_table()
        try:
            symm_mem.rendezvous(t, args.group)
        except RuntimeError as exc:
            errors.append(str(exc).strip().splitlines()[0].split("  (Could this error")[0][:200])
            for fd, target in fd_table().items():
                if fd in before:
                    continue
                leaked_targets[target] = leaked_targets.get(target, 0) + 1
                if sc.close_leaked_fds and target.startswith("/dev/nvidia"):
                    os.close(fd)
                    closed += 1
        del t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier()
        free, _ = torch.cuda.mem_get_info(device)
        drops.append((free0 - free) / MiB)
        fds.append(len(fd_table()) - nfd0)

    return {
        "scenario": sc.name,
        "rank": rank,
        "device": device.index,
        "alloc_mib": args.mib * mult,
        "errors": len(errors),
        "first_error": errors[0] if errors else None,
        "free_drop_mib": drops,
        "open_fds_delta": fds,
        "leak_mib_per_iter": slope(drops),
        "leak_fds_per_iter": slope(fds),
        "new_fd_targets": leaked_targets,
        "fds_closed_by_probe": closed,
    }


def verdict(r: dict, world_size: int, overlap: bool) -> str:
    # Overlapping ranks share cuda:0, so its free memory reflects every rank's leak.
    expected = r["alloc_mib"] * (world_size if overlap else 1)
    per_iter = r["leak_mib_per_iter"]
    if per_iter >= 0.5 * r["alloc_mib"]:
        return f"LEAK (~{per_iter:.0f} MiB/iter of ~{expected} MiB allocated)"
    if per_iter <= 0.05 * r["alloc_mib"] and r["leak_fds_per_iter"] < 0.5:
        return "no leak"
    return f"unclear ({per_iter:.0f} MiB/iter, {r['leak_fds_per_iter']:.1f} fds/iter)"


def handle_path_from_log(text: str) -> str:
    paths = set(re.findall(r"using (posix fd|fabric handle) to import", text))
    fabric = set(re.findall(r"has_fabric_support (\d)", text))
    if not paths:
        return "unknown (no torch INFO log seen)"
    return f"{'/'.join(sorted(paths))} (has_fabric_support={'/'.join(sorted(fabric)) or '?'})"


def driver_version() -> str:
    with contextlib.suppress(OSError):
        with open("/proc/driver/nvidia/version") as fh:
            m = re.search(r"Kernel Module\s+(?:for \S+\s+)?([\d.]+)", fh.readline())
            if m:
                return m.group(1)
    return "unknown"


def write_report(outdir: str, meta: dict, results: list[dict], scenarios: list[Scenario]) -> str:
    world_size = meta["world_size"]
    lines = [
        "# Symmetric-memory rendezvous leak probe",
        "",
        f"Issue: {ISSUE}",
        "",
        f"- date: {meta['date']}",
        f"- host: {meta['host']} ({meta['gpu']}, driver {meta['driver']})",
        f"- torch: {meta['torch']} (git {meta['git']}, CUDA {meta['cuda']})",
        f"- world size: {world_size}, alloc: {meta['mib']} MiB/rank, iters: {meta['iters']}",
        f"- handle path: {meta['handle_path']}"
        + (" -- forced with the no-fabric shim" if meta["forced_posix"] else ""),
        "",
        "Leak rates are growth per iteration after the first, so one-time costs drop out.",
        "",
        "| scenario | what | rejected | rank 0 verdict | leaked fds/iter (rank 0) |",
        "|---|---|---|---|---|",
    ]
    by = {(r["scenario"], r["rank"]): r for r in results}
    leaks = {}
    for sc in scenarios:
        r0 = by[(sc.name, 0)]
        rejected = sum(by[(sc.name, r)]["errors"] for r in range(world_size))
        v = verdict(r0, world_size, sc.overlap)
        leaks[sc.name] = v.startswith("LEAK")
        lines.append(
            f"| {sc.name} | {sc.what} | {rejected}/{meta['iters'] * world_size} | {v} "
            f"| {r0['leak_fds_per_iter']:.1f} |"
        )

    lines += ["", "## Bottom line", ""]
    rejected_names = [s.name for s in scenarios if s.name in ("size", "overlap")]
    if "ok" in leaks and leaks["ok"]:
        lines.append("- The control leaked too, so the numbers below are not trustworthy.")
    if any(leaks.get(n) for n in rejected_names):
        hit = ", ".join(n for n in rejected_names if leaks.get(n))
        lines.append(f"- REPRODUCED: rejected rendezvous leaks the allocation ({hit}).")
    elif rejected_names:
        lines.append(f"- No leak on rejected rendezvous ({', '.join(rejected_names)}).")
    if "overlap-close" in leaks and leaks.get("overlap"):
        lines.append(
            "- Closing the fds a failure leaves open stops the leak: the exported fd pins it."
            if not leaks["overlap-close"]
            else "- Closing leftover fds did NOT stop the leak: something besides the fd pins it."
        )
    if not meta["forced_posix"] and "fabric" in meta["handle_path"]:
        lines.append(
            "- This box used fabric handles. Rerun with `--handle posix` to test the fd path."
        )

    header = (
        "| scenario | rank | device | MiB/iter | fds/iter | free drop by iter (MiB) | new fds |"
    )
    lines += ["", "## Per rank", "", header + " first error |", "|---|---|---|---|---|---|---|---|"]
    for sc in scenarios:
        for rank in range(world_size):
            r = by[(sc.name, rank)]
            drops = " ".join(f"{d:.0f}" for d in r["free_drop_mib"])
            targets = ", ".join(f"{k} x{v}" for k, v in r["new_fd_targets"].items()) or "-"
            err = (r["first_error"] or "-").replace("|", "/")
            lines.append(
                f"| {sc.name} | {rank} | {r['device']} | {r['leak_mib_per_iter']:.0f} "
                f"| {r['leak_fds_per_iter']:.1f} | {drops} | {targets} | {err} |"
            )
    report = "\n".join(lines) + "\n"
    with open(os.path.join(outdir, "report.md"), "w") as fh:
        fh.write(report)
    with open(os.path.join(outdir, "report.json"), "w") as fh:
        json.dump({"meta": meta, "results": results}, fh, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--handle",
        choices=("auto", "posix"),
        default="auto",
        help="auto: whatever torch picks; posix: hide fabric support from torch",
    )
    p.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help=f"comma-separated subset of {','.join(SCENARIOS)}",
    )
    p.add_argument("--iters", type=int, default=6, help="rendezvous attempts per scenario (>= 2)")
    p.add_argument("--mib", type=int, default=256, help="allocation size per rank")
    p.add_argument("--pg-timeout", type=float, default=120.0)
    p.add_argument("--outdir", default=None, help="default: ./reports/symm_mem_leak-<time>-<host>")
    args = p.parse_args()
    unknown = set(args.scenarios.split(",")) - set(SCENARIOS)
    if unknown:
        p.error(f"unknown scenarios: {sorted(unknown)}")
    if args.iters < 2:
        p.error("--iters must be >= 2 to separate per-iteration leaks from one-time costs")
    return args


def main() -> int:
    args = parse_args()
    if "RANK" not in os.environ:
        print("run under torchrun: torchrun --nproc_per_node=2 " + sys.argv[0], file=sys.stderr)
        return 2
    if args.handle == "posix":
        reexec_with_posix_shim()

    if os.environ.setdefault("TORCH_SYMMMEM", "CUDA") != "CUDA":
        print("this probe targets the CUDA symm_mem backend; unset TORCH_SYMMMEM", file=sys.stderr)
        return 2
    # torch reports which handle type it picked only at INFO. Keep that (and the rest of the
    # C++ chatter) out of the terminal: each rank's stderr goes to a log that lands in outdir.
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    rank = int(os.environ["RANK"])
    log = tempfile.NamedTemporaryFile(
        prefix=f"symm_mem_leak_rank{rank}_", suffix=".log", delete=False
    )
    saved_stderr = os.dup(2)
    os.dup2(log.fileno(), 2)
    try:
        return run(args, rank, log.name)
    except BaseException:
        os.dup2(saved_stderr, 2)
        print(f"[rank {rank}] failed; torch log: {log.name}", file=sys.stderr)
        raise


def run(args: argparse.Namespace, rank: int, log_path: str) -> int:
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    scenarios = [SCENARIOS[n] for n in args.scenarios.split(",")]
    needs_own_gpu = any(not s.overlap for s in scenarios)
    if world_size < 2 or (needs_own_gpu and torch.cuda.device_count() <= local_rank):
        raise SystemExit("needs >= 2 ranks, and one GPU per local rank for ok/size")

    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=args.pg_timeout))
    args.group = dist.group.WORLD.group_name
    stamp = [datetime.datetime.now().strftime("%Y%m%d-%H%M%S")]
    dist.broadcast_object_list(stamp, src=0)
    args.outdir = args.outdir or os.path.join(
        "reports", f"symm_mem_leak-{stamp[0]}-{socket.gethostname().split('.')[0]}-{args.handle}"
    )
    os.makedirs(args.outdir, exist_ok=True)
    say = (lambda msg: print(msg, flush=True)) if rank == 0 else (lambda msg: None)
    say(
        f"symm_mem rendezvous leak probe: torch {torch.__version__}, world_size={world_size}, "
        f"handle={args.handle}, {args.mib} MiB x {args.iters} iters -> {args.outdir}"
    )

    results = []
    for sc in scenarios:
        say(f"  {sc.name:<14} {sc.what}")
        results.append(run_scenario(sc, args, rank, local_rank, dist, torch, symm_mem))

    sys.stderr.flush()
    with open(log_path) as fh:
        handle_path = handle_path_from_log(fh.read())
    shutil.copy(log_path, os.path.join(args.outdir, f"rank{rank}.log"))
    os.unlink(log_path)

    gathered: list[list[dict] | None] = [None] * world_size
    dist.all_gather_object(gathered, results)
    paths: list[str | None] = [None] * world_size
    dist.all_gather_object(paths, handle_path)
    if rank == 0:
        meta = {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0),
            "driver": driver_version(),
            "torch": torch.__version__,
            "git": (torch.version.git_version or "")[:12],
            "cuda": torch.version.cuda,
            "world_size": world_size,
            "mib": args.mib,
            "iters": args.iters,
            "handle_path": " | ".join(sorted(set(paths))),
            "forced_posix": args.handle == "posix",
            "scenarios": [asdict(s) for s in scenarios],
        }
        flat = [r for per_rank in gathered for r in per_rank]
        report = write_report(args.outdir, meta, flat, scenarios)
        print(report, flush=True)
        print(f"report: {os.path.join(args.outdir, 'report.md')}", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
