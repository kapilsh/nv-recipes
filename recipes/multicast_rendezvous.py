#!/usr/bin/env python3
"""Multicast rendezvous probe: CUDA multicast setup, step by step, in Python.

Replicates the shape of PyTorch's symmetric-memory multicast rendezvous
(``init_multicast_for_block()`` in
``torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu``) over
``cuda.bindings``, so every driver call's CUresult is visible and any single
step can be made to fail on any single rank.

Written to debug https://github.com/pytorch/pytorch/pull/195596, which fixes
both failure modes below.

Two things it makes reproducible:

1. ``cuMulticastBindMem`` blocks until all ``numDevices`` devices have joined
   the object. No timeout, not a NCCL op, so nothing in the stack notices. A
   rank that fails to join and skips its own bind does not rescue its peers --
   they are already inside the driver. ``cuMemMap`` blocks the same way.
2. Publishing success before the mapping exists lets peers commit to a
   multicast address a failed rank does not have.

The agreement points that prevent each are separately switchable, so you can
run the broken and fixed orderings side by side.

    torchrun --nproc_per_node=8 multicast_rendezvous.py [--help]

Notes: the control plane is gloo, keeping CUDA out of everything except the
calls under test. ``cuMulticastCreate`` needs at least 2 devices.
"""

from __future__ import annotations

import argparse
import array
import ctypes
import datetime
import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import cuda.bindings.driver as drv

OK = drv.CUresult.CUDA_SUCCESS
FABRIC = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
POSIX_FD = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

STEPS = (
    "create",  # cuMulticastCreate            (rank 0)
    "export",  # cuMemExportToShareableHandle (rank 0)
    "import",  # cuMemImportFromShareableHandle
    "add_device",  # cuMulticastAddDevice -- last non-blocking step
    "bind",  # cuMulticastBindMem   -- first blocking step
    "reserve",  # cuMemAddressReserve \
    "map",  # cuMemMap             } map_block()
    "set_access",  # cuMemSetAccess      /
)


def call(result: Any) -> tuple[drv.CUresult, Any]:
    """Unpack a cuda.bindings ``(CUresult, *out)`` tuple without raising."""
    if not isinstance(result, tuple):
        result = (result,)
    err, *out = result
    if not out:
        return err, None
    return err, out[0] if len(out) == 1 else tuple(out)


def as_int(handle: Any) -> int:
    """CUdeviceptr / CUmemGenericAllocationHandle -> plain int."""
    return 0 if handle is None else int(handle)


class Log:
    def __init__(self, rank: int) -> None:
        self.rank, self.n = rank, 0

    def step(self, name: str, err: Any, ms: float, extra: str = "") -> None:
        self.n += 1
        status = err if isinstance(err, str) else err.name
        mark = " " if status == "CUDA_SUCCESS" else "!"
        print(
            f"[rank {self.rank}] {self.n:02d}{mark} {name:<32} -> {status:<24}"
            f" {ms:7.2f} ms  {extra}".rstrip(),
            flush=True,
        )

    def note(self, msg: str) -> None:
        print(f"[rank {self.rank}]    . {msg}", flush=True)

    def event(self, msg: str) -> None:
        print(f"[rank {self.rank}] ** {msg}", flush=True)


class Control:
    """Rendezvous control plane over gloo. ``issued`` counts collectives."""

    def __init__(self, dist: Any, rank: int, world_size: int) -> None:
        self.dist, self.rank, self.world_size, self.issued = dist, rank, world_size, 0

    def all_gather(self, value: Any) -> list[Any]:
        out: list[Any] = [None] * self.world_size
        self.dist.all_gather_object(out, value)
        self.issued += 1
        return out

    def broadcast(self, value: Any, root: int = 0) -> Any:
        box = [value if self.rank == root else None]
        self.dist.broadcast_object_list(box, src=root)
        self.issued += 1
        return box[0]

    def barrier(self) -> None:
        self.dist.barrier()


class IpcChannel:
    """Unix datagram socket for SCM_RIGHTS fd passing (POSIX handle path)."""

    def __init__(self, rank: int, tag: str) -> None:
        self.tag = tag
        self.path = self._path(rank)
        if os.path.exists(self.path):
            os.unlink(self.path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(self.path)

    def _path(self, rank: int) -> str:
        return f"/tmp/mc_probe-{os.getuid()}-{self.tag}-{rank}"

    def send_fd(self, dst_rank: int, fd: int) -> None:
        # Not socket.send_fds: it accepts an `address` and then ignores it, so
        # an unconnected datagram socket gets ENOTCONN.
        rights = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
        self.sock.sendmsg([b"h"], rights, 0, self._path(dst_rank))

    def recv_fd(self) -> int:
        _, fds, _, _ = socket.recv_fds(self.sock, 1, 1)
        return fds[0] if fds else -1

    def close(self) -> None:
        self.sock.close()
        if os.path.exists(self.path):
            os.unlink(self.path)


@dataclass
class Setup:
    """Owns everything setup acquires; release() undoes it in reverse order."""

    dev: int
    size: int
    log: Log
    phys: int = 0
    created_mc: int = 0
    imported_mc: int = 0
    bound: bool = False
    va: int = 0
    mapped: bool = False
    access_set: bool = False
    owned_fd: int = -1
    committed: bool = False

    def live(self) -> str:
        bits = [
            n
            for n, on in (
                ("phys", self.phys),
                ("created_mc", self.created_mc),
                ("imported_mc", self.imported_mc),
                ("bound", self.bound),
                ("mapped", self.mapped),
                ("access", self.access_set),
            )
            if on
        ]
        if self.va:
            bits.append(f"va@{hex(self.va)}")
        if self.owned_fd >= 0:
            bits.append(f"fd={self.owned_fd}")
        return ",".join(bits) or "-"

    def release(self) -> None:
        def undo(result: Any, what: str) -> None:
            err, _ = call(result)
            self.log.note(f"{what} -> {err.name}")

        if self.committed:
            self.log.note("committed; only the received fd is still owned here")
        else:
            self.log.note(f"release() starting, live: {self.live()}")
            if self.mapped:
                undo(drv.cuMemUnmap(self.va, self.size), "cuMemUnmap")
                self.mapped = False
            if self.va:
                # PyTorch never calls this, so a failed-and-retried rendezvous
                # leaks the reservation on every attempt.
                undo(drv.cuMemAddressFree(self.va, self.size), "cuMemAddressFree")
                self.va = 0
            if self.imported_mc:
                if self.bound:
                    unbind = drv.cuMulticastUnbind(self.imported_mc, self.dev, 0, self.size)
                    undo(unbind, "cuMulticastUnbind")
                    self.bound = False
                undo(drv.cuMemRelease(self.imported_mc), "cuMemRelease(imported)")
                self.imported_mc = 0
            if self.created_mc:
                undo(drv.cuMemRelease(self.created_mc), "cuMemRelease(created)")
                self.created_mc = 0
        if self.owned_fd >= 0:
            os.close(self.owned_fd)
            self.owned_fd = -1
        if not self.committed and self.phys:
            undo(drv.cuMemRelease(self.phys), "cuMemRelease(phys)")
            self.phys = 0


def timed(fn: Any, log: Log, name: str, extra: str = "", block_timeout: float | None = None):
    """Run one driver call. Blocking calls get a watchdog thread.

    Returns ``(None, None)`` on BLOCKED -- the only way to observe
    cuMulticastBindMem / cuMemMap waiting on a device that never joined.
    """
    box: dict[str, Any] = {}
    t0 = time.monotonic()
    if block_timeout is None:
        box["r"] = call(fn())
    else:
        th = threading.Thread(target=lambda: box.update(r=call(fn())), daemon=True)
        th.start()
        th.join(block_timeout)
        if th.is_alive():
            log.step(name, f"BLOCKED >{block_timeout:g}s", (time.monotonic() - t0) * 1000)
            return None, None
    err, out = box["r"]
    val = f"= {hex(int(out))}" if hasattr(out, "__int__") and int(out) > 0xFFFF else ""
    log.step(name, err, (time.monotonic() - t0) * 1000, extra or val)
    return err, out


def map_block(setup: Setup, fails: Any, log: Log, args: Any, dev_id: int, align: int) -> bool:
    """The three driver calls behind map_block(), each individually failable.

    Only the first writes the pointer, so a failure in step 2 or 3 leaves a live
    VA reservation the caller cannot tell apart from a live mapping.
    """
    if fails("reserve"):
        return False
    err, va = timed(
        lambda: drv.cuMemAddressReserve(setup.size, align, 0, 0), log, "cuMemAddressReserve"
    )
    setup.va = as_int(va) if err == OK else 0
    if err != OK:
        return False

    if fails("map"):
        log.note(f"VA {hex(setup.va)} reserved but unmapped -- release() must free it")
        return False
    err, _ = timed(
        lambda: drv.cuMemMap(setup.va, setup.size, 0, setup.imported_mc, 0),
        log,
        "cuMemMap",
        block_timeout=args.blocking_timeout,
    )
    if err is None:
        log.event("cuMemMap BLOCKED -- same team-complete condition as the bind")
        return False
    if err != OK:
        return False
    setup.mapped = True

    if fails("set_access"):
        log.note(f"VA {hex(setup.va)} reserved AND mapped -- release() must undo both")
        return False
    desc = drv.CUmemAccessDesc()
    desc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = dev_id
    desc.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    err, _ = timed(
        lambda: drv.cuMemSetAccess(setup.va, setup.size, [desc], 1), log, "cuMemSetAccess"
    )
    setup.access_set = err == OK
    return setup.access_set


def verify(setup: Setup, log: Log, dev_id: int, ctl: Control) -> None:
    """Rank 0 writes through the multicast VA; every rank reads its own memory."""
    err, local = call(drv.cuMemAddressReserve(setup.size, 0, 0, 0))
    if err != OK:
        return
    local = as_int(local)
    call(drv.cuMemMap(local, setup.size, 0, setup.phys, 0))
    desc = drv.CUmemAccessDesc()
    desc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = dev_id
    desc.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    call(drv.cuMemSetAccess(local, setup.size, [desc], 1))
    call(drv.cuMemsetD32(local, 0, 4))
    call(drv.cuCtxSynchronize())
    ctl.barrier()

    if ctl.rank == 0:
        call(drv.cuMemsetD32(setup.va, 0xABCD, 4))
    call(drv.cuCtxSynchronize())
    ctl.barrier()

    buf = (ctypes.c_uint32 * 1)()
    call(drv.cuMemcpyDtoH(ctypes.addressof(buf), local, 4))
    log.note(f"verify: local word = {hex(buf[0])} (expect 0xabcd)")
    call(drv.cuMemUnmap(local, setup.size))
    call(drv.cuMemAddressFree(local, setup.size))


def rendezvous(  # noqa: C901 - one linear transcript of the C++, on purpose
    args: Any, ctl: Control, ipc: IpcChannel | None, rank: int, dev_id: int, log: Log
) -> tuple[dict[str, Any], Setup]:
    world_size = ctl.world_size
    htype = FABRIC if args.handle == "fabric" else POSIX_FD
    out: dict[str, Any] = {"status": "unknown", "mc_va": None, "live": None}

    def fails(step: str) -> bool:
        if rank == args.fail_rank and step == args.fail_step:
            log.event(f"INJECTING FAILURE at '{step}'")
            log.step(step, "INJECTED_FAILURE", 0.0)
            return True
        return False

    call(drv.cuInit(0))
    _, dev = call(drv.cuDeviceGet(dev_id))
    _, ctx = call(drv.cuDevicePrimaryCtxRetain(dev))
    call(drv.cuCtxSetCurrent(ctx))

    # Multicast size must be a multiple of the *multicast* granularity, which is
    # a different query from the allocation granularity.
    mc_prop = drv.CUmulticastObjectProp()
    mc_prop.numDevices, mc_prop.handleTypes, mc_prop.flags = world_size, htype, 0
    _, gran = call(
        drv.cuMulticastGetGranularity(
            mc_prop, drv.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED
        )
    )
    size = ((args.size + gran - 1) // gran) * gran
    mc_prop.size = size

    setup = Setup(dev=int(dev), size=size, log=log)
    try:
        prop = drv.CUmemAllocationProp()
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = dev_id
        prop.requestedHandleTypes = htype
        err, phys = timed(lambda: drv.cuMemCreate(size, prop, 0), log, "cuMemCreate", f"{size}B")
        setup.phys = as_int(phys)
        if err != OK:
            raise RuntimeError("cuMemCreate failed")

        # Phase 1: create + export the multicast object (rank 0 only).
        exported = None
        if rank == 0:
            if not fails("create"):
                err, created = timed(
                    lambda: drv.cuMulticastCreate(mc_prop), log, "cuMulticastCreate"
                )
                setup.created_mc = as_int(created) if err == OK else 0
            if setup.created_mc and not fails("export"):
                err, exported = timed(
                    lambda: drv.cuMemExportToShareableHandle(setup.created_mc, htype, 0),
                    log,
                    "cuMemExportToShareableHandle",
                )
                exported = exported if err == OK else None

        # Phase 2: distribute the handle. A fabric handle is 64 opaque bytes; a
        # POSIX fd has to go over a unix socket with SCM_RIGHTS.
        have = exported is not None
        if args.handle == "fabric":
            got = ctl.broadcast(bytes(exported.data) if have else None)
            if got is None:
                log.event("invalidator received: rank 0 could not export")
                out["status"] = "skipped_no_handle"
                return out, setup
            os_handle: Any = got  # raw bytes; a rebuilt CUmemFabricHandle is rejected
        else:
            if not ctl.broadcast(have):
                log.event("invalidator received: rank 0 could not export")
                out["status"] = "skipped_no_handle"
                return out, setup
            ctl.barrier()  # every socket bound before rank 0 sends
            if rank == 0:
                for peer in range(1, world_size):
                    ipc.send_fd(peer, exported)
                setup.owned_fd = exported
            else:
                setup.owned_fd = ipc.recv_fd()
            os_handle = setup.owned_fd

        # Phase 3: import and join the team. Both local and non-blocking.
        joined = not fails("import")
        if joined:
            err, imported = timed(
                lambda: drv.cuMemImportFromShareableHandle(os_handle, htype),
                log,
                "cuMemImportFromShareableHandle",
            )
            setup.imported_mc = as_int(imported) if err == OK else 0
            joined = err == OK
        if joined and not fails("add_device"):
            err, _ = timed(
                lambda: drv.cuMulticastAddDevice(setup.imported_mc, dev),
                log,
                "cuMulticastAddDevice",
            )
            joined = err == OK
        elif joined:
            joined = False

        # Agreement A. Without it a rank that failed above just skips its own
        # bind, and every peer already inside cuMulticastBindMem waits forever.
        if args.pre_bind_agreement:
            flags = ctl.all_gather(joined)
            log.note(f"pre-bind agreement: {['ok' if f else 'FAIL' for f in flags]}")
            if not all(flags):
                out["status"] = "skipped_pre_bind"
                return out, setup
        elif not joined:
            log.note("pre-bind agreement DISABLED")
            if args.short_circuit:
                log.event("failed to join; peers were not told")
                out["status"] = "failed_join_silently"
                return out, setup
            log.event("failed to join; going to the final exchange, where nobody will arrive")
            t0 = time.monotonic()
            try:
                ctl.all_gather(False)
            except Exception as exc:  # noqa: BLE001 - the point of the run
                log.event(
                    f"exchange failed after {time.monotonic() - t0:.1f}s "
                    f"(limit {args.pg_timeout:g}s): {type(exc).__name__}. This rank -- the "
                    "one that was NOT blocked -- is the only one that reports anything."
                )
                out["status"] = "timed_out_alone"
                return out, setup
            out["status"] = "failed_join_silently"
            return out, setup

        # Phase 4: bind. First call that blocks on other ranks.
        good = not fails("bind")
        if good:
            err, _ = timed(
                lambda: drv.cuMulticastBindMem(setup.imported_mc, 0, setup.phys, 0, size, 0),
                log,
                "cuMulticastBindMem",
                block_timeout=args.blocking_timeout,
            )
            if err is None:
                log.event("cuMulticastBindMem BLOCKED -- a device never joined the object")
                out["status"] = "blocked_in_bind"
                return out, setup
            setup.bound = good = err == OK

        # Phase 5: map, then agree. Agreeing *before* the map is what let a rank
        # publish a success it had not earned; --agree-before-map restores that.
        if args.agree_before_map:
            flags = ctl.all_gather(good)
            log.note(f"agreement AFTER BIND: {['ok' if f else 'FAIL' for f in flags]}")
            if not all(flags):
                out["status"] = "skipped_post_bind"
                return out, setup
            if not map_block(setup, fails, log, args, dev_id, gran):
                log.event("map failed after success was published; peers already committed")
                out["status"] = "threw_after_publish"
                return out, setup
        else:
            good = good and map_block(setup, fails, log, args, dev_id, gran)
            if args.post_map_agreement:
                flags = ctl.all_gather(good)
                log.note(f"final agreement: {['ok' if f else 'FAIL' for f in flags]}")
                if not all(flags):
                    out["status"] = "skipped_post_map"
                    return out, setup
            elif not good:
                log.event("failed to map but peers were not told")
                out["status"] = "failed_map_silently"
                return out, setup

        setup.committed = True
        out["status"], out["mc_va"] = "committed", hex(setup.va)
        log.event(f"COMMITTED multicast VA {hex(setup.va)}")
        if args.verify:
            verify(setup, log, dev_id, ctl)
        return out, setup
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not crash
        log.event(f"EXCEPTION: {type(exc).__name__}: {exc}")
        out["status"] = f"exception:{type(exc).__name__}"
        return out, setup


def summarize(outdir: str, world_size: int, wait: float) -> int:
    """Collect per-rank result files. File-based: a rank stuck in the driver
    never reaches another collective, and rank 0 is often one of them."""
    paths = {r: os.path.join(outdir, f"rank{r}.json") for r in range(world_size)}
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline and not all(os.path.exists(p) for p in paths.values()):
        time.sleep(0.1)

    results = {}
    for rank, path in paths.items():
        if os.path.exists(path):
            with open(path) as fh:
                results[rank] = json.load(fh)

    print("=" * 96)
    print(f"collectives issued (rank 0): {results.get(0, {}).get('collectives')}")
    for rank in range(world_size):
        got = results.get(rank)
        if got is None:
            print(f"  rank {rank}: NO RESULT (died or stuck)")
        else:
            print(f"  rank {rank}: {got['status']:<22} mc_va={got['mc_va']} live={got['live']}")
    statuses = {results.get(r, {}).get("status") for r in range(world_size)}
    consistent = len(statuses) == 1 and None not in statuses
    print(f"agreement: {'CONSISTENT' if consistent else 'DIVERGENT ' + str(statuses)}")
    print("=" * 96)
    return 0 if consistent else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--size", type=int, default=2 << 20, help="bytes, rounded to granularity")
    p.add_argument("--handle", choices=("fabric", "posix"), default="fabric")
    p.add_argument("--fail-rank", type=int, default=-1)
    p.add_argument("--fail-step", choices=STEPS, default=None)
    p.add_argument(
        "--no-pre-bind-agreement",
        dest="pre_bind_agreement",
        action="store_false",
        help="drop the agreement between AddDevice and BindMem (reproduces the hang)",
    )
    p.add_argument(
        "--no-post-map-agreement",
        dest="post_map_agreement",
        action="store_false",
        help="drop the final success exchange",
    )
    p.add_argument(
        "--agree-before-map",
        action="store_true",
        help="publish success before mapping (the pre-fix ordering)",
    )
    p.add_argument(
        "--no-short-circuit",
        dest="short_circuit",
        action="store_false",
        help="a failed rank still enters the next exchange, and waits there alone",
    )
    p.add_argument(
        "--blocking-timeout",
        type=float,
        default=3.0,
        help="seconds before a blocking driver call is reported BLOCKED",
    )
    p.add_argument("--pg-timeout", type=float, default=30.0, help="gloo + rendezvous store timeout")
    p.add_argument("--summary-wait", type=float, default=10.0)
    p.add_argument("--verify", action="store_true", help="write via multicast, read back locally")
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def banner(args: argparse.Namespace, world_size: int) -> None:
    print("=" * 96)
    print(
        f"multicast rendezvous probe: world_size={world_size} handle={args.handle} "
        f"pre_bind={args.pre_bind_agreement} post_map={args.post_map_agreement} "
        f"agree_before_map={args.agree_before_map}"
    )
    if args.fail_rank >= 0:
        print(f"  injecting: rank {args.fail_rank} fails at '{args.fail_step}'")
    if not args.short_circuit and args.blocking_timeout <= args.pg_timeout:
        print(
            f"  WARNING: --blocking-timeout ({args.blocking_timeout:g}s) <= --pg-timeout "
            f"({args.pg_timeout:g}s); blocked peers will exit before the failing rank's "
            "exchange times out, so you get a connection error, not a timeout"
        )
    print("=" * 96, flush=True)


def main() -> int:
    args = parse_args()
    if "RANK" not in os.environ:
        print(
            "run under torchrun: torchrun --nproc_per_node=8 multicast_rendezvous.py",
            file=sys.stderr,
        )
        return 2

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dev_id = int(os.environ.get("LOCAL_RANK", rank))
    tag = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getppid()))
    args.outdir = args.outdir or f"/tmp/mc_probe-{os.getuid()}-{tag}"

    if world_size < 2:
        print("cuMulticastCreate needs numDevices >= 2", file=sys.stderr)
        return 2
    if (args.fail_rank >= 0) != (args.fail_step is not None):
        print("--fail-rank and --fail-step go together", file=sys.stderr)
        return 2
    if args.fail_step in ("create", "export") and args.fail_rank not in (-1, 0):
        print(f"step '{args.fail_step}' only runs on rank 0", file=sys.stderr)
        return 2

    # gloo, not nccl: the control plane must not create CUDA state of its own.
    import torch.distributed as dist

    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=args.pg_timeout))
    ctl, log = Control(dist, rank, world_size), Log(rank)

    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        for stale in os.listdir(args.outdir):
            os.unlink(os.path.join(args.outdir, stale))
        banner(args, world_size)
    ctl.barrier()

    ipc = IpcChannel(rank, tag) if args.handle == "posix" else None
    try:
        out, setup = rendezvous(args, ctl, ipc, rank, dev_id, log)
        blocked = out["status"] == "blocked_in_bind"
        out["live"], out["collectives"] = setup.live(), ctl.issued
        setup.release()
    finally:
        if ipc is not None:
            ipc.close()

    with open(os.path.join(args.outdir, f"rank{rank}.json"), "w") as fh:
        json.dump(out, fh)

    rc = summarize(args.outdir, world_size, args.summary_wait) if rank == 0 else 0
    if blocked:
        # A thread is still parked in the driver; a normal shutdown would hang.
        sys.stdout.flush()
        os._exit(rc)
    dist.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())
