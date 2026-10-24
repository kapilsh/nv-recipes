#!/usr/bin/env python3
"""Symmetric-memory allocation rollback probe (pytorch/pytorch#201284).

Replays the CUDA VMM setup in
``torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory{.cu,Utils.cpp}``
over ``cuda.bindings``: ``CUDASymmetricMemoryAllocator::alloc()``,
``map_block()``, one iteration of the peer-import loop in
``make_peer_alloc_info()`` and ``MulticastSetup::bind_and_map()``. Any one
step can be made to fail, and the run then reports what is still held.

The PR adds several guards. Each one is a separate switch, so the probe runs:

    base          the code before the PR
    fix           the PR as written
    fix-<guard>   the PR with that one guard removed

A guard that can be dropped without making any injected failure worse is not
protecting anything. One that only matters for failures that cannot happen
without a sticky context error or std::bad_alloc is defensive only.

What is still held is read from three places: a ledger of every driver call the
code under test made; free device memory (``cuMemGetInfo``) before and after,
which shows physical memory that is pinned; and a reaper that frees whatever the
ledger says is left. A reaper call that returns CUDA_SUCCESS is the driver
confirming the resource really was still live.

    python recipes/symm_mem_alloc_rollback.py                  # all paths, all arms
    python recipes/symm_mem_alloc_rollback.py --paths alloc --trace

Single process, no torch. ``peer`` and ``mc`` need two GPUs (``mc`` also needs
multicast); one process drives both devices, which is enough to bind a
multicast object.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import cuda.bindings.driver as drv

PR = "https://github.com/pytorch/pytorch/pull/201284"
MiB = 1 << 20
OK = drv.CUresult.CUDA_SUCCESS
POSIX_FD = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
INJECTED = "INJECTED"

# step -> (what fails, reachable without a sticky context error or bad_alloc)
STEPS: dict[str, dict[str, tuple[str, bool]]] = {
    "alloc": {
        "none": ("control: alloc succeeds, free() drops the AllocationRef", True),
        "create": ("cuMemCreate; nothing acquired yet", True),
        "reserve": ("cuMemAddressReserve in map_block", True),
        "map": ("cuMemMap in map_block", True),
        "set_access": ("cuMemSetAccess in map_block", True),
        "set_access+unmap": ("set_access fails, then the rollback's cuMemUnmap fails too", False),
        "owner": ("make_intrusive<AllocationRef> throws (std::bad_alloc)", False),
        "memset": ("cudaMemsetAsync of the signal pad", False),
        "sync": ("cudaStreamSynchronize after the memset", False),
    },
    "peer": {
        "none": ("control: import and map one peer, then tear down", True),
        "import": ("cuMemImportFromShareableHandle", True),
        "reserve": ("cuMemAddressReserve in map_block", True),
        "map": ("cuMemMap in map_block", True),
        "set_access": ("cuMemSetAccess in map_block (peer access)", True),
        "set_access+unmap": ("set_access fails, then the rollback's cuMemUnmap fails too", False),
    },
    "mc": {
        "none": ("control: bind_and_map succeeds, commit(), AllocationRef freed", True),
        "reserve": ("cuMemAddressReserve in map_block", True),
        "map": ("cuMemMap of the multicast handle", True),
        "set_access": ("cuMemSetAccess of the multicast mapping", True),
        "set_access+unmap": ("set_access fails, then the rollback's cuMemUnmap fails too", False),
    },
}

GUARDS = {
    "unwind": "map_block undoes its own reserve/map when a later step throws",
    "late_publish": "map_block writes *ptr only once the mapping is complete",
    "unmap_gate": "a rollback skips cuMemAddressFree when its cuMemUnmap failed",
    "alloc_release": "alloc() scope guard releases the handle until AllocationRef owns it",
    "alloc_unmap": "that guard also unmaps/frees alloc_base if it was published",
    "early_owner": "AllocationRef is built before the memset/sync instead of after",
}
ARMS: dict[str, frozenset[str]] = {
    "base": frozenset(),
    "fix": frozenset(GUARDS),
    **{f"fix-{g}": frozenset(GUARDS) - {g} for g in GUARDS},
}


def call(result: Any) -> tuple[drv.CUresult, Any]:
    """Unpack a cuda.bindings ``(CUresult, *out)`` tuple without raising."""
    if not isinstance(result, tuple):
        result = (result,)
    err, *out = result
    if not out:
        return err, None
    return err, out[0] if len(out) == 1 else tuple(out)


def as_int(handle: Any) -> int:
    return 0 if handle is None else int(handle)


def err_name(err: Any) -> str:
    return err if isinstance(err, str) else err.name


def access_desc(dev: int) -> drv.CUmemAccessDesc:
    desc = drv.CUmemAccessDesc()
    desc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = dev
    desc.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return desc


def alloc_prop(dev: int) -> drv.CUmemAllocationProp:
    prop = drv.CUmemAllocationProp()
    prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev
    prop.requestedHandleTypes = POSIX_FD
    return prop


class DriverError(RuntimeError):
    """What C10_CUDA_DRIVER_CHECK throws."""

    def __init__(self, step: str, err: str) -> None:
        super().__init__(f"{step}: {err}")
        self.step, self.err = step, err


class Ledger:
    """Every driver call the code under test makes, and what it still holds.

    ``check`` is C10_CUDA_DRIVER_CHECK (throws), ``warn`` is
    C10_CUDA_DRIVER_CHECK_WARN (records and carries on). A step named in
    ``fail`` is not sent to the driver at all; it returns INJECTED.
    """

    def __init__(self, fail: set[str]) -> None:
        self.fail = fail
        self.handles: dict[int, str] = {}
        self.vas: dict[int, int] = {}
        self.maps: dict[int, int] = {}
        self.fds: set[int] = set()
        self.freed: set[int] = set()
        self.warnings: list[str] = []
        self.hazards: list[str] = []
        self.trace: list[str] = []

    def injected(self, step: str) -> bool:
        return step in self.fail

    def _run(self, step: str, what: str, fn: Callable[[], Any]) -> tuple[Any, Any]:
        if self.injected(step):
            self.trace.append(f"{what} -> {INJECTED}")
            return INJECTED, None
        err, out = call(fn())
        self.trace.append(f"{what} -> {err.name}")
        return err, out

    def check(self, step: str, what: str, fn: Callable[[], Any]) -> Any:
        err, out = self._run(step, what, fn)
        if err != OK:
            raise DriverError(step, err_name(err))
        return out

    def warn(self, step: str, what: str, fn: Callable[[], Any]) -> Any:
        err, _ = self._run(step, what, fn)
        if err != OK:
            self.warnings.append(f"{what}={err_name(err)}")
        return err

    def _stale(self, va: int, what: str) -> None:
        if va in self.freed and va not in self.vas:
            self.hazards.append(f"{what} on a VA this code already freed")

    def create(self, size: int, prop: drv.CUmemAllocationProp) -> int:
        h = as_int(self.check("create", "cuMemCreate", lambda: drv.cuMemCreate(size, prop, 0)))
        self.handles[h] = "created"
        return h

    def import_fd(self, fd: int) -> int:
        h = as_int(
            self.check(
                "import",
                "cuMemImportFromShareableHandle",
                lambda: drv.cuMemImportFromShareableHandle(fd, POSIX_FD),
            )
        )
        self.handles[h] = "imported"
        return h

    def reserve(self, size: int, align: int) -> int:
        va = as_int(
            self.check(
                "reserve", "cuMemAddressReserve", lambda: drv.cuMemAddressReserve(size, align, 0, 0)
            )
        )
        self.vas[va] = size
        self.freed.discard(va)
        return va

    def map(self, va: int, size: int, handle: int) -> None:
        self.check("map", "cuMemMap", lambda: drv.cuMemMap(va, size, 0, handle, 0))
        self.maps[va] = size

    def set_access(self, va: int, size: int, dev: int) -> None:
        self.check(
            "set_access",
            "cuMemSetAccess",
            lambda: drv.cuMemSetAccess(va, size, [access_desc(dev)], 1),
        )

    def unmap(self, va: int, size: int, step: str = "unmap") -> Any:
        self._stale(va, "cuMemUnmap")
        err = self.warn(step, "cuMemUnmap", lambda: drv.cuMemUnmap(va, size))
        if err == OK:
            self.maps.pop(va, None)
        return err

    def addr_free(self, va: int, size: int, step: str = "free") -> Any:
        self._stale(va, "cuMemAddressFree")
        err = self.warn(step, "cuMemAddressFree", lambda: drv.cuMemAddressFree(va, size))
        if err == OK:
            if va in self.maps:
                self.hazards.append("cuMemAddressFree succeeded on a still-mapped VA")
            self.vas.pop(va, None)
            self.freed.add(va)
        return err

    def release(self, handle: int, step: str = "release") -> Any:
        err = self.warn(step, "cuMemRelease", lambda: drv.cuMemRelease(handle))
        if err == OK:
            self.handles.pop(handle, None)
        return err

    def close_fd(self, fd: int) -> None:
        os.close(fd)
        self.fds.discard(fd)

    def reap(self) -> list[str]:
        """Free what is left. SUCCESS means the driver agrees it was live."""
        out = []
        for va, size in list(self.maps.items()):
            out.append(f"unmap:{call(drv.cuMemUnmap(va, size))[0].name}")
        for va, size in list(self.vas.items()):
            out.append(f"free:{call(drv.cuMemAddressFree(va, size))[0].name}")
        for h in list(self.handles):
            out.append(f"release:{call(drv.cuMemRelease(h))[0].name}")
        for fd in list(self.fds):
            os.close(fd)
            out.append("close:ok")
        self.maps.clear(), self.vas.clear(), self.handles.clear(), self.fds.clear()
        return out


class Out:
    """The ``void** ptr`` out-parameter."""

    v = 0


@dataclass
class AllocationRef:
    """~AllocationRef: unmap, then release. No cuMemAddressFree -- that
    normal-path VA leak is pytorch/pytorch#201091's separate follow-up."""

    L: Ledger
    ptr: int
    handle: int
    size: int

    def destroy(self) -> None:
        self.L.unmap(self.ptr, self.size)
        if self.handle:
            self.L.release(self.handle)


def new_owner(L: Ledger, ptr: int, handle: int, size: int) -> AllocationRef:
    if L.injected("owner"):
        L.trace.append(f"make_intrusive<AllocationRef> -> {INJECTED}")
        raise DriverError("owner", "std::bad_alloc")
    return AllocationRef(L, ptr, handle, size)


def map_block(
    L: Ledger, g: frozenset[str], out: Out, handle: int, size: int, dev: int, align: int
) -> None:
    """map_block(): reserve, map, set access."""
    if "late_publish" in g:
        out.v = 0
    va, reserved, mapped = 0, False, False
    try:
        va = L.reserve(size, align)
        reserved = True
        if "late_publish" not in g:
            out.v = va  # pre-PR: cuMemAddressReserve writes straight into *ptr
        L.map(va, size, handle)
        mapped = True
        L.set_access(va, size, dev)
    except DriverError:
        if "unwind" in g:
            unmapped = L.unmap(va, size, "rollback_unmap") if mapped else OK
            if reserved and (unmapped == OK or "unmap_gate" not in g):
                L.addr_free(va, size, "rollback_free")
        raise
    if "late_publish" in g:
        out.v = va


@dataclass
class Env:
    dev: int
    peer: int
    size: int
    mc_align: int
    iters: int
    ctx: dict[int, Any]
    cudev: dict[int, Any]
    mc_prop: Any = None
    harness: Ledger = field(default_factory=lambda: Ledger(set()))


def alloc(L: Ledger, g: frozenset[str], env: Env) -> AllocationRef:
    """CUDASymmetricMemoryAllocator::alloc(), CUDA branch."""
    size = env.size
    h = L.create(size, alloc_prop(env.dev))
    out, owner = Out(), None
    armed = "alloc_release" in g  # c10::make_scope_exit right after cuMemCreate
    try:
        map_block(L, g, out, h, size, env.dev, 0)
        if "early_owner" in g:
            owner = new_owner(L, out.v, h, size)
            armed = False  # rollback.release()
        L.check("memset", "cudaMemsetAsync", lambda: drv.cuMemsetD8(out.v, 0, min(size, MiB)))
        L.check("sync", "cudaStreamSynchronize", drv.cuCtxSynchronize)
        if owner is None:
            owner = new_owner(L, out.v, h, size)
            armed = False
        return owner
    except DriverError:
        if owner is not None:
            owner.destroy()  # the intrusive_ptr unwinds
        elif armed:
            if "alloc_unmap" in g and out.v:
                unmapped = L.unmap(out.v, size, "rollback_unmap")
                if unmapped == OK or "unmap_gate" not in g:
                    L.addr_free(out.v, size, "rollback_free")
            L.release(h, "rollback_release")
        raise


def run_alloc(L: Ledger, g: frozenset[str], env: Env) -> str:
    try:
        owner = alloc(L, g, env)
    except DriverError as exc:
        return f"threw:{exc.step}"
    owner.destroy()  # free(): erase the Block -> ~AllocationRef
    return "ok"


def run_peer(L: Ledger, g: frozenset[str], env: Env) -> str:
    """One peer of the import loop in make_peer_alloc_info(). The loop has no
    cleanup: a throw leaves handles[r] and the received fd behind, as before."""
    H, size = env.harness, env.size
    peer_handle = H.create(size, alloc_prop(env.peer))
    fd = as_int(
        H.check(
            "export",
            "cuMemExportToShareableHandle",
            lambda: drv.cuMemExportToShareableHandle(peer_handle, POSIX_FD, 0),
        )
    )
    received = os.dup(fd)  # what SCM_RIGHTS would have delivered
    L.fds.add(received)
    os.close(fd)  # the exporter closes block_handle after the barrier
    ref, result = None, "ok"
    try:
        handle = L.import_fd(received)
        out = Out()
        map_block(L, g, out, handle, size, env.dev, 0)
        L.close_fd(received)
        ref = AllocationRef(L, out.v, handle, size)
    except DriverError as exc:
        result = f"threw:{exc.step}"
    if ref is not None:
        ref.destroy()
    H.release(peer_handle)  # the peer frees its own tensor
    return result


def run_mc(L: Ledger, g: frozenset[str], env: Env) -> str:
    """MulticastSetup::bind_and_map() on env.dev, then release() or commit()."""
    H, size, devs = env.harness, env.size, (env.dev, env.peer)
    mc = as_int(H.check("mc", "cuMulticastCreate", lambda: drv.cuMulticastCreate(env.mc_prop)))
    H.handles[mc] = "mc"
    for d in devs:
        H.check(
            "mc", "cuMulticastAddDevice", lambda d=d: drv.cuMulticastAddDevice(mc, env.cudev[d])
        )
    phys = {d: H.create(size, alloc_prop(d)) for d in devs}
    for d in devs:
        H.check(
            "mc",
            "cuMulticastBindMem",
            lambda d=d: drv.cuMulticastBindMem(mc, 0, phys[d], 0, size, 0),
        )

    addr, ref, result = Out(), None, "ok"
    try:
        map_block(L, g, addr, mc, size, env.dev, env.mc_align)
        ref = AllocationRef(L, addr.v, 0, size)  # commit(); the harness owns mc
        addr.v = 0
    except DriverError as exc:
        result = f"threw:{exc.step}"  # bind_and_map() returns false
    if addr.v:  # MulticastSetup::release(): unmap addr_, never cuMemAddressFree
        L.unmap(addr.v, size, "release_unmap")
        addr.v = 0
    if ref is not None:
        ref.destroy()
    for d in devs:
        H.warn(
            "mc",
            "cuMulticastUnbind",
            lambda d=d: drv.cuMulticastUnbind(mc, int(env.cudev[d]), 0, size),
        )
    H.release(mc)
    for d in devs:
        H.release(phys[d])
    return result


RUNNERS = {"alloc": run_alloc, "peer": run_peer, "mc": run_mc}


def free_bytes(env: Env, dev: int) -> int:
    call(drv.cuCtxSetCurrent(env.ctx[dev]))
    _, (free, _total) = call(drv.cuMemGetInfo())
    call(drv.cuCtxSetCurrent(env.ctx[env.dev]))
    return int(free)


@dataclass
class Cell:
    path: str
    arm: str
    step: str
    outcome: str
    handles: int
    vas: int
    maps: int
    fds: int
    phys_mib: int
    pinned: bool
    warnings: list[str]
    hazards: list[str]
    reaped: list[str]
    dirty: bool
    trace: list[str]

    def score(self) -> tuple[int, ...]:
        return (
            self.handles,
            self.vas,
            self.maps,
            self.fds,
            int(self.pinned),
            len(self.real_warnings()),
            len(self.hazards),
        )

    def real_warnings(self) -> list[str]:
        """Cleanup errors from the driver, not the fault this case injected."""
        return [w for w in self.warnings if not w.endswith(INJECTED)]

    def short(self) -> str:
        if self.outcome.startswith("harness:"):
            return "HARNESS"
        bits = [
            f"{tag}{n if n > 1 else ''}"
            for tag, n in (("H", self.handles), ("V", self.vas), ("M", self.maps), ("F", self.fds))
            if n
        ]
        if self.pinned:
            bits.append(f"P{self.phys_mib}")
        if self.real_warnings():
            bits.append(f"w{len(self.real_warnings())}")
        if self.hazards:
            bits.append("!")
        if any(not r.endswith(("CUDA_SUCCESS", "ok")) for r in self.reaped):
            bits.append("?")
        if self.dirty:
            bits.append("D")
        return " ".join(bits) or "clean"


def run_cell(env: Env, path: str, arm: str, step: str) -> Cell:
    fail = {"rollback_unmap" if s == "unmap" else s for s in step.split("+")} - {"none"}
    L, env.harness = Ledger(fail), Ledger(set())
    measured = env.peer if path == "peer" else env.dev
    before = free_bytes(env, measured)
    outcomes = []
    try:
        for _ in range(env.iters):
            outcomes.append(RUNNERS[path](L, ARMS[arm], env))
    except DriverError as exc:
        outcomes.append(f"harness:{exc}")
    after = free_bytes(env, measured)
    phys = (before - after) // MiB
    snapshot = dict(handles=len(L.handles), vas=len(L.vas), maps=len(L.maps), fds=len(L.fds))
    reaped = L.reap()
    env.harness.reap()
    dirty = abs(before - free_bytes(env, measured)) >= env.size // 2
    return Cell(
        path=path,
        arm=arm,
        step=step,
        outcome=outcomes[-1],
        phys_mib=int(phys),
        pinned=phys * MiB >= env.size * env.iters // 2,
        warnings=L.warnings,
        hazards=sorted(set(L.hazards)),
        reaped=reaped,
        dirty=dirty,
        trace=L.trace,
        **snapshot,
    )


def facts(env: Env) -> list[tuple[str, str, str]]:
    """Driver behaviour the guards rely on, measured directly."""
    size, dev, rows = env.size, env.dev, []

    def fresh(mapped: bool) -> tuple[int, int]:
        _, h = call(drv.cuMemCreate(size, alloc_prop(dev), 0))
        _, va = call(drv.cuMemAddressReserve(size, 0, 0, 0))
        h, va = as_int(h), as_int(va)
        if mapped:
            call(drv.cuMemMap(va, size, 0, h, 0))
            call(drv.cuMemSetAccess(va, size, [access_desc(dev)], 1))
        return h, va

    h, va = fresh(mapped=False)
    err = call(drv.cuMemUnmap(va, size))[0].name
    rows.append(("cuMemUnmap on a reserved, never-mapped VA", err, "late_publish / mc release()"))
    call(drv.cuMemAddressFree(va, size)), call(drv.cuMemRelease(h))

    base = free_bytes(env, dev)
    h, va = fresh(mapped=True)
    held = (base - free_bytes(env, dev)) // MiB
    call(drv.cuMemRelease(h))
    after_release = (base - free_bytes(env, dev)) // MiB
    call(drv.cuMemUnmap(va, size))
    after_unmap = (base - free_bytes(env, dev)) // MiB
    call(drv.cuMemAddressFree(va, size))
    rows.append(
        (
            "MiB held: mapped / after cuMemRelease / after cuMemUnmap",
            f"{held} / {after_release} / {after_unmap}",
            "unwind (a leaked mapping pins memory)",
        )
    )

    h, va = fresh(mapped=True)
    err = call(drv.cuMemAddressFree(va, size))[0].name
    still = call(drv.cuMemUnmap(va, size))[0].name
    again = call(drv.cuMemAddressFree(va, size))[0].name
    rows.append(
        (
            "cuMemAddressFree on a still-mapped VA (then unmap, then free)",
            f"{err} ({still}, {again})",
            "unmap_gate",
        )
    )
    call(drv.cuMemRelease(h))

    _, va = call(drv.cuMemAddressReserve(size, 0, 0, 0))
    va = as_int(va)
    call(drv.cuMemAddressFree(va, size))
    twice = call(drv.cuMemAddressFree(va, size))[0].name
    unmap_freed = call(drv.cuMemUnmap(va, size))[0].name
    _, reuse = call(drv.cuMemAddressReserve(size, 0, 0, 0))
    reused = as_int(reuse) == va
    call(drv.cuMemAddressFree(as_int(reuse), size))
    rows.append(
        (
            "freed VA: free again / unmap / next reserve returns it",
            f"{twice} / {unmap_freed} / {reused}",
            "late_publish (caller acting on a freed VA)",
        )
    )
    return rows


def verdicts(cells: dict[tuple[str, str, str], Cell], paths: list[str], arms: list[str]):
    """For each guard: where dropping it from the PR changes the outcome."""
    out = {}
    for g in GUARDS:
        arm = f"fix-{g}"
        if arm not in arms or "fix" not in arms:
            continue
        worse, better = [], []
        for path in paths:
            for step in STEPS[path]:
                a, b = cells[(path, "fix", step)], cells[(path, arm, step)]
                if a.score() == b.score():
                    continue
                tag = f"{path}/{step}: {a.short()} -> {b.short()}"
                if any(y > x for x, y in zip(a.score(), b.score(), strict=True)):
                    worse.append((tag, STEPS[path][step][1]))
                else:
                    better.append(tag)
        if not worse:
            verdict = "REDUNDANT: no injected failure gets worse without it"
        elif not any(real for _, real in worse):
            verdict = "DEFENSIVE ONLY: matters only for sticky-error / bad_alloc / double faults"
        else:
            verdict = "LOAD-BEARING"
        out[g] = {"verdict": verdict, "worse": [t for t, _ in worse], "better": better}
    return out


def print_matrix(cells, path: str, arms: list[str]) -> None:
    w = 2 + max(len(a) for a in arms)
    w = max(w, *(len(cells[(path, a, s)].short()) + 2 for a in arms for s in STEPS[path]))
    print(f"\n[{path}]".ljust(21) + "".join(a.ljust(w) for a in arms))
    for step, (_, real) in STEPS[path].items():
        label = step if real else f"{step}*"
        print(f"  {label:<19}" + "".join(cells[(path, a, step)].short().ljust(w) for a in arms))


def setup(args: argparse.Namespace) -> tuple[Env, list[str]]:
    call(drv.cuInit(0))
    _, count = call(drv.cuDeviceGetCount())
    devs = sorted({args.dev, args.peer}) if count > 1 else [args.dev]
    ctx, cudev = {}, {}
    for d in devs:
        _, cudev[d] = call(drv.cuDeviceGet(d))
        _, ctx[d] = call(drv.cuDevicePrimaryCtxRetain(cudev[d]))
    call(drv.cuCtxSetCurrent(ctx[args.dev]))

    _, gran = call(
        drv.cuMemGetAllocationGranularity(
            alloc_prop(args.dev),
            drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    )
    size = max(args.mib * MiB, gran)
    paths = list(args.paths)
    mc_prop, mc_align = None, 0
    if len(devs) < 2:
        paths = [p for p in paths if p == "alloc"]
        print("one GPU: only the alloc path runs")
    elif "mc" in paths:
        attr = drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
        _, supported = call(drv.cuDeviceGetAttribute(attr, cudev[args.dev]))
        if not supported:
            paths.remove("mc")
            print("no multicast support: skipping the mc path")
        else:
            mc_prop = drv.CUmulticastObjectProp()
            mc_prop.numDevices, mc_prop.handleTypes, mc_prop.flags = 2, POSIX_FD, 0
            _, mc_gran = call(
                drv.cuMulticastGetGranularity(
                    mc_prop, drv.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED
                )
            )
            size = max(size, int(mc_gran))
            mc_align = int(mc_gran) if args.mc_align else 0
    size = -(-size // gran) * gran
    if mc_prop is not None:
        mc_prop.size = size
    env = Env(
        dev=args.dev,
        peer=args.peer,
        size=size,
        mc_align=mc_align,
        iters=args.iters,
        ctx=ctx,
        cudev=cudev,
        mc_prop=mc_prop,
    )
    return env, paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--paths", type=lambda s: s.split(","), default=["alloc", "peer", "mc"])
    p.add_argument("--arms", type=lambda s: s.split(","), default=list(ARMS))
    p.add_argument("--mib", type=int, default=256, help="block size, rounded up to granularity")
    p.add_argument("--iters", type=int, default=1, help="repeat each case before measuring")
    p.add_argument("--dev", type=int, default=0)
    p.add_argument("--peer", type=int, default=1, help="second GPU for the peer and mc paths")
    p.add_argument(
        "--mc-align",
        action="store_true",
        help="reserve the multicast VA at multicast granularity (torch passes 0)",
    )
    p.add_argument("--trace", action="store_true", help="print every cell's driver calls")
    p.add_argument("--no-facts", dest="facts", action="store_false")
    p.add_argument("--json", default=None, help="write all cells and verdicts here")
    p.add_argument("--timeout", type=float, default=120.0, help="abort if a driver call wedges")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bad = [a for a in args.arms if a not in ARMS] + [p for p in args.paths if p not in STEPS]
    if bad:
        print(f"unknown arm/path: {bad}; arms are {list(ARMS)}", file=sys.stderr)
        return 2
    # cuMulticastBindMem and cuMemMap can block forever; nothing else would notice.
    watchdog = threading.Timer(args.timeout, lambda: (print("TIMEOUT", flush=True), os._exit(3)))
    watchdog.daemon = True
    watchdog.start()

    env, paths = setup(args)
    print("=" * 100)
    print(f"symm_mem allocation rollback probe -- {PR}")
    print(f"block {env.size // MiB} MiB, iters {env.iters}, dev {env.dev}, peer {env.peer}")
    for g, what in GUARDS.items():
        print(f"  {g:<14} {what}")
    print("=" * 100)

    fact_rows = facts(env) if args.facts else []
    if fact_rows:
        print("\ndriver facts")
        for what, result, bears_on in fact_rows:
            print(f"  {what:<62} {result:<32} [{bears_on}]")

    cells = {}
    for path in paths:
        for arm in args.arms:
            for step in STEPS[path]:
                cells[(path, arm, step)] = cell = run_cell(env, path, arm, step)
                if args.trace:
                    print(f"  {path}/{arm}/{step}: {cell.outcome}  {' | '.join(cell.trace)}")
                    if cell.warnings or cell.hazards:
                        print(f"      warnings={cell.warnings} hazards={cell.hazards}")

    print("\nstill held after the case (and after free() on success):")
    print("  H handle  V VA reservation  M mapping  F fd  P<n> MiB pinned  w<n> cleanup")
    print("  warnings  ! hazard  ? reaper disagreed  D memory not restored  * step needs")
    print("  a sticky context error, bad_alloc or a second fault")
    for path in paths:
        print_matrix(cells, path, args.arms)

    result = verdicts(cells, paths, args.arms)
    if result:
        print("\nguards (PR minus one guard, vs the PR):")
        for g, v in result.items():
            print(f"  {g:<14} {v['verdict']}")
            for t in v["worse"]:
                print(f"      worse   {t}")
            for t in v["better"]:
                print(f"      better  {t}")

    if "base" in args.arms and "fix" in args.arms:
        print("\nbase -> fix:")
        for (path, arm, step), c in cells.items():
            if arm != "base":
                continue
            f = cells[(path, "fix", step)]
            if c.score() != f.score() or f.short() != "clean":
                print(f"  {path}/{step:<17} {c.short():<16} -> {f.short()}")

    dirty = [k for k, c in cells.items() if c.dirty]
    if dirty:
        print(f"\nWARNING: memory not restored after {dirty}; later cells may be off")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "pr": PR,
                    "block_mib": env.size // MiB,
                    "facts": fact_rows,
                    "cells": [asdict(c) for c in cells.values()],
                    "verdicts": result,
                },
                fh,
                indent=1,
            )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
