#!/usr/bin/env python3
"""Do memory snapshots see a pluggable allocator? Checked against its own log.

A ``CUDAPluggableAllocator`` can be plugged in two ways, and they are not the
same thing:

``pool``    ``MemPool(allocator)``. The caching allocator stays in charge; it
            only calls our alloc/free where it would have called cudaMalloc /
            cudaFree for a whole segment. Splitting, rounding, caching and the
            history recorder are all still the caching allocator's.
``global``  ``change_current_allocator(allocator)``. The caching allocator is
            gone: every tensor is one call into our alloc. Nothing is left to
            record history or take a snapshot unless the pluggable path
            implements it.

The allocator here is two Python functions over ``cuda.bindings.runtime``,
handed to torch as ctypes function pointers -- no .so, same entry point
(``torch._C._cuda_customAllocator``) that ``CUDAPluggableAllocator`` uses after
it dlopens one. Every call is logged, and the snapshot is checked against that
log: each segment in the pool should be exactly one of our cudaMallocs, same
address, same size, and nothing we allocated should be missing.

    python recipes/pluggable_allocator_snapshot.py            # both modes
    python recipes/pluggable_allocator_snapshot.py --mode pool --out /tmp/snaps

Each mode runs in its own subprocess: ``change_current_allocator`` only works
before the CUDA allocator is first used, and allocator state is process-global.
``--out`` keeps the ``.pickle`` from ``torch.cuda.memory._dump_snapshot``, which
https://www.kapilsharma.dev/oomscope/ (or pytorch.org/memory_viz) can open.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile

import cuda.bindings.runtime as rt

OK = rt.cudaError_t.cudaSuccess

# What torch calls, from c10/cuda/CUDAAllocatorConfig / CUDAPluggableAllocator:
#   void* alloc(size_t size, int device, cudaStream_t stream)
#   void  free(void* ptr, size_t size, int device, cudaStream_t stream)
ALLOC_T = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)
FREE_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)

MiB = 1 << 20


class LoggingAllocator:
    """cudaMalloc / cudaFree through cuda.bindings, recording every call."""

    def __init__(self):
        self.calls: list[dict] = []
        self.live: dict[int, int] = {}
        # Keep the ctypes thunks alive for as long as torch may call them.
        self._alloc = ALLOC_T(self.alloc)
        self._free = FREE_T(self.free)

    def alloc(self, size, device, stream):
        err, ptr = rt.cudaMalloc(size)
        if err != OK:
            self.calls.append({"op": "alloc", "size": size, "error": err.name})
            return None  # torch turns a null return into an OOM
        ptr = int(ptr)
        self.live[ptr] = size
        self.calls.append(
            {"op": "alloc", "addr": ptr, "size": size, "device": device, "stream": stream or 0}
        )
        return ptr

    def free(self, ptr, size, device, stream):
        (err,) = rt.cudaFree(ptr)
        self.live.pop(ptr, None)
        self.calls.append(
            {"op": "free", "addr": ptr, "size": size, "error": None if err == OK else err.name}
        )

    def torch_allocator(self):
        import torch
        from torch.cuda.memory import _CUDAAllocator

        fn = lambda f: ctypes.cast(f, ctypes.c_void_p).value  # noqa: E731
        return _CUDAAllocator(torch._C._cuda_customAllocator(fn(self._alloc), fn(self._free)))


def workload(torch, pool=None):
    """A few shapes that exercise both size classes and the cache."""
    kept = []
    ctx = torch.cuda.use_mem_pool(pool) if pool is not None else _null()
    with ctx:
        kept.append(torch.empty(30 * MiB, dtype=torch.uint8, device="cuda"))  # own large segment
        kept += [
            torch.empty(3 * MiB, dtype=torch.uint8, device="cuda") for _ in range(4)
        ]  # split one 20 MiB
        kept += [
            torch.empty(4096, dtype=torch.uint8, device="cuda") for _ in range(8)
        ]  # small pool
        tmp = torch.empty(6 * MiB, dtype=torch.uint8, device="cuda")
        del tmp  # back to the cache, not to us
    return kept


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def try_(label, fn, report):
    try:
        out = fn()
        report["steps"][label] = "ok"
        return out
    except Exception as e:  # noqa: BLE001 -- the error text is the finding
        report["steps"][label] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        return None


def run_mode(mode: str, out_dir: str | None) -> dict:
    alloc = LoggingAllocator()
    report: dict = {"mode": mode, "steps": {}}

    import torch

    report["torch"] = torch.__version__
    if mode == "global":
        # Must precede any CUDA allocation in this process.
        try_(
            "change_current_allocator",
            lambda: torch.cuda.memory.change_current_allocator(alloc.torch_allocator()),
            report,
        )

    try_(
        "_record_memory_history",
        lambda: torch.cuda.memory._record_memory_history(max_entries=100_000),
        report,
    )

    pool = torch.cuda.MemPool(alloc.torch_allocator().allocator()) if mode == "pool" else None
    if pool is not None:
        report["pool_id"] = list(pool.id)
    _default_side = (
        torch.empty(8 * MiB, dtype=torch.uint8, device="cuda") if mode == "pool" else None
    )  # noqa: F841
    kept = workload(torch, pool)  # noqa: F841
    torch.cuda.synchronize()

    snap = try_("_snapshot", torch.cuda.memory._snapshot, report)
    if out_dir and snap is not None:
        path = os.path.join(out_dir, f"pluggable-{mode}.pickle")
        try_("_dump_snapshot", lambda: torch.cuda.memory._dump_snapshot(path), report)
        report["pickle"] = path
    try_(
        "memory_allocated",
        lambda: report.__setitem__("memory_allocated", torch.cuda.memory_allocated()),
        report,
    )

    allocs = [c for c in alloc.calls if c["op"] == "alloc" and "addr" in c]
    report["allocator_calls"] = {
        "alloc": len(allocs),
        "free": sum(c["op"] == "free" for c in alloc.calls),
        "live": len(alloc.live),
        "live_bytes": sum(alloc.live.values()),
        "sizes_mib": sorted(round(c["size"] / MiB, 3) for c in allocs),
    }

    if snap is not None:
        report["snapshot"] = compare(snap, alloc, pool)
    return report


def compare(snap: dict, alloc: LoggingAllocator, pool) -> dict:
    """Line up snapshot segments with what our allocator actually handed out."""
    pid = tuple(pool.id) if pool is not None else None
    segs = snap["segments"]
    ours = [s for s in segs if s["address"] in alloc.live]
    in_pool = [
        s for s in segs if pid is not None and tuple(s.get("segment_pool_id", (0, 0))) == pid
    ]
    traces = snap.get("device_traces") or [[]]
    trace = traces[0] if traces else []

    mismatched = [
        {
            "addr": hex(s["address"]),
            "snapshot": s["total_size"],
            "allocator": alloc.live[s["address"]],
        }
        for s in ours
        if s["total_size"] != alloc.live[s["address"]]
    ]
    seen = {s["address"] for s in segs}
    missing = [{"addr": hex(a), "size": n} for a, n in alloc.live.items() if a not in seen]
    seg_allocs = {e["addr"] for e in trace if e["action"] == "segment_alloc"}

    return {
        "segments_total": len(segs),
        "segments_from_our_allocator": len(ours),
        "segments_tagged_with_pool": len(in_pool),
        "our_segments_not_tagged_with_pool": sum(s not in in_pool for s in ours) if pid else None,
        "pool_segment_types": sorted({s["segment_type"] for s in in_pool}),
        "blocks_in_our_segments": sum(len(s["blocks"]) for s in ours),
        "live_blocks_in_our_segments": sum(
            b["state"] == "active_allocated" for s in ours for b in s["blocks"]
        ),
        "size_mismatches": mismatched,
        "allocated_but_missing_from_snapshot": missing,
        "trace_events": len(trace),
        "trace_segment_allocs_for_our_segments": sum(a in seg_allocs for a in alloc.live),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["pool", "global", "both"], default="both")
    ap.add_argument("--out", help="directory to keep the snapshot pickles in")
    ap.add_argument("--json", action="store_true", help="print raw JSON (used by the parent run)")
    args = ap.parse_args()

    if args.mode != "both":
        report = run_mode(args.mode, args.out)
        print(json.dumps(report) if args.json else json.dumps(report, indent=2))
        return

    out = args.out or tempfile.mkdtemp(prefix="pluggable-snap-")
    os.makedirs(out, exist_ok=True)
    for mode in ("pool", "global"):
        p = subprocess.run(
            [sys.executable, __file__, "--mode", mode, "--out", out, "--json"],
            capture_output=True,
            text=True,
        )
        if p.returncode != 0 or not p.stdout.strip():
            print(f"== {mode}: process failed (exit {p.returncode})\n{p.stderr[-2000:]}")
            continue
        summarise(json.loads(p.stdout.strip().splitlines()[-1]))


def summarise(r: dict):
    print(f"\n== {r['mode']}  (torch {r['torch']})")
    for k, v in r["steps"].items():
        print(f"  {k:26s} {v}")
    a = r["allocator_calls"]
    print(
        f"  our allocator              {a['alloc']} alloc / {a['free']} free, "
        f"{a['live']} live = {a['live_bytes'] / MiB:.1f} MiB, sizes {a['sizes_mib']}"
    )
    if "memory_allocated" in r:
        print(f"  torch memory_allocated     {r['memory_allocated'] / MiB:.1f} MiB")
    s = r.get("snapshot")
    if not s:
        print("  snapshot                   none")
        return
    if "pool_id" in r:
        print(f"  pool id                    {tuple(r['pool_id'])}")
    for k, v in s.items():
        print(f"  {k:36s} {v}")
    if r.get("pickle"):
        print(f"  pickle                     {r['pickle']}")


if __name__ == "__main__":
    main()
