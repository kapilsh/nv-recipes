#!/usr/bin/env python3
"""Minimal NCCL GIN (GPU-Initiated Networking) over plain TCP sockets, in nccl4py.

NCCL 2.32.3 added a built-in socket RMA backend ("RMA_SOCKET_PROXY",
src/transport/rma_socket*.cc). The GPU kernel posts puts/signals into the
GIN proxy queue, and a CPU proxy thread moves the bytes over TCP. No
InfiniBand/RoCE NIC is needed, so this is handy for dev boxes, cloud VMs,
and CI.

The example is a one-shot ring: each rank does a GPU-initiated put of its
buffer into the next rank's window, attaching a signal increment. It then
waits for the signal from the previous rank, and the host checks the data.

Selecting the socket backend (environment):
  NCCL_GDRCOPY_ENABLE=1   required: the socket backend needs GDRCopy. Either
                          install libgdrapi + the gdrdrv kernel module, or
                          rely on NCCL's internal DMA-BUF mmap fallback,
                          which needs a CUDA driver >= 13.3.
  NCCL_IB_DISABLE=1       skips the IB GDAKI and IB proxy backends, which
                          are otherwise picked before sockets.
  NCCL_SOCKET_IFNAME=eth0 (optional) chooses the TCP interface
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
                          look for "RMA/Plugin: Assigned plugin RMA_SOCKET_PROXY"

Run with MPI, one rank per GPU (any rank count >= 2). MPI only shares the NCCL
unique id; NCCL itself does all GPU communication. Needs NCCL >= 2.32.3 and
nccl4py >= 0.6.0, from the `gin` extra (`uv sync --extra gin`).
  export NCCL_GDRCOPY_ENABLE=1 NCCL_IB_DISABLE=1
  # one node, 2 GPUs
  mpiexec -n 2 python recipes/gin_socket_ring.py
  # two nodes, 1 GPU each (real TCP path)
  mpiexec -n 2 -ppn 1 -hosts node1,node2 python recipes/gin_socket_ring.py
"""

import sys

import cupy as cp
import cutlass
import cutlass.cute as cute
import nccl.core as nccl
import nccl.core.device.cute as nccl_cute
from cuda.core import Device
from mpi4py import MPI

NUM_ELEMS = 1 << 19  # 512K int64 = 4 MiB
THREADS = 128
SIGNAL_ID = 0
SIGNAL_OP_INC = 0  # ncclGinSignalOp_t: 0 = Inc, 1 = Add


@cute.kernel
def ring_kernel(
    dev_comm: nccl_cute.DevComm, send_win: nccl_cute.Window, recv_win: nccl_cute.Window
):
    """Put the whole send buffer into the next rank's recv window, then wait
    for the previous rank's put to land here."""
    world = dev_comm.team_world
    gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)  # GIN context 0
    coop = nccl_cute.cta()
    next_rank = (world.rank + 1) % world.nRanks

    send = send_win.tensor(cutlass.Int64, cute.make_layout(NUM_ELEMS))
    recv = recv_win.tensor(cutlass.Int64, cute.make_layout(NUM_ELEMS))

    # The whole CTA issues one put; the signal is bumped on the peer once
    # the data has landed there.
    gin.put(
        world,
        next_rank,
        recv_win,
        recv,  # destination (on the peer)
        send_win,
        send,  # source (local)
        coop,
        is_signal=True,
        signal_id=SIGNAL_ID,
        signal_op=SIGNAL_OP_INC,
        signal_op_arg=1,
    )
    # Signals start at 0 on a fresh devComm, so exactly one put lands here.
    gin.wait_signal(coop, signal=SIGNAL_ID, least=1)
    # Make sure our own outgoing put has drained before the kernel exits.
    gin.flush(coop)


@cute.jit
def ring(dev_comm: nccl_cute.DevComm, send_win: nccl_cute.Window, recv_win: nccl_cute.Window):
    ring_kernel(dev_comm, send_win, recv_win).launch(
        grid=[1, 1, 1], block=[THREADS, 1, 1], cooperative=True
    )


def main():
    mpi = MPI.COMM_WORLD
    rank, nranks = mpi.Get_rank(), mpi.Get_size()
    lib = nccl.get_version().libnccl
    if lib is None or lib.version.release < (2, 32, 3):
        print(f"ERROR: need libnccl >= 2.32.3 for GIN over sockets, loaded {lib}")
        return 1
    if nranks < 2:
        print("ERROR: needs at least 2 ranks")
        return 1

    # One GPU per rank, indexed by the rank's position on its node.
    local_rank = mpi.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
    device = Device(local_rank)
    device.set_current()

    # Rank 0 creates the NCCL unique id; MPI broadcasts it (UniqueId pickles).
    unique_id = mpi.bcast(nccl.get_unique_id() if rank == 0 else None, root=0)
    comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    if not comm.device_api_support or comm.gin_type == nccl.NcclGinType.NONE:
        print(
            f"rank {rank}: no GIN support (gin_type={comm.gin_type.name}). "
            "Did you set NCCL_GDRCOPY_ENABLE=1?",
            flush=True,
        )
        comm.destroy()
        return 1
    print(
        f"rank {rank}/{nranks} on GPU {device.device_id}: gin_type={comm.gin_type.name}"
        + ("" if comm.gin_type == nccl.NcclGinType.PROXY else " (socket backend not selected)"),
        flush=True,
    )

    # Symmetric buffers registered as windows so peers can target them.
    send_buf = nccl.cupy.empty(NUM_ELEMS, dtype="int64")
    recv_buf = nccl.cupy.empty(NUM_ELEMS, dtype="int64")
    send_buf[:] = cp.arange(NUM_ELEMS, dtype="int64") + rank * 1_000_000_000
    recv_buf[:] = 0
    device.sync()
    send_win = comm.register_window(send_buf)
    recv_win = comm.register_window(recv_buf)

    # Device communicator: one signal, full GIN connectivity, proxy-type GIN
    # (socket RMA is a proxy backend).
    reqs = nccl.NCCLDevCommRequirements(
        gin_connection_type=nccl.NcclGinConnectionType.FULL,
        gin_signal_count=SIGNAL_ID + 1,
        gin_type=nccl.NcclGinType.PROXY,
    )
    dev_comm = comm.create_dev_comm(requirements=reqs)

    ring(nccl_cute.DevComm(dev_comm), nccl_cute.Window(send_win), nccl_cute.Window(recv_win))
    device.sync()

    # Check that recv_buf holds what the previous rank sent.
    prev = (rank - 1) % nranks
    expected = cp.arange(NUM_ELEMS, dtype="int64") + prev * 1_000_000_000
    errors = int(cp.count_nonzero(recv_buf != expected).item())
    print(
        f"rank {rank}: received {NUM_ELEMS * 8} bytes from rank {prev} over GIN/socket -> "
        f"{'FAILED' if errors else 'PASSED'}",
        flush=True,
    )
    errors = mpi.allreduce(errors)

    mpi.Barrier()  # nobody tears down a window a peer may still target
    dev_comm.close()
    send_win.close()
    recv_win.close()
    comm.destroy()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
