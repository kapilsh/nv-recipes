# nv-recipes

Small scripts to test out CUDA and NCCL behaviors.

Each recipe is standalone and driven by `torchrun`. They talk to the CUDA driver
through `cuda.bindings` directly, so every call's `CUresult` is visible and
individual steps can be made to fail on individual ranks.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU.

```bash
uv sync
source .venv/bin/activate
```

That installs `cuda-python`. PyTorch is a separate step, because you may want a
nightly, a stable release, or your own build. Pick one.

**Nightly** -- match the CUDA tag to your toolkit (`nvcc --version`). `cu126`,
`cu128`, `cu129` and `cu130` are all published:

```bash
uv pip install --prerelease=allow --index-url https://download.pytorch.org/whl/nightly/cu130 torch
```

**Stable:**

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu130 torch
```

**Your own build:**

```bash
uv pip install -e /path/to/pytorch      # or just run the recipes from that venv
```

Check it:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

Use `--index-url` rather than `--extra-index-url` for the nightly. Nightly torch
pins an exact `+git` triton build that only exists in the same index, and mixing
in PyPI can leave the resolver unable to satisfy it. If a nightly still fails to
resolve, the torch wheel is briefly ahead of its pinned triton -- pick the
previous day's build or retry later.

## Recipes

### `recipes/multicast_rendezvous.py`

Replicates the shape of PyTorch's symmetric-memory multicast rendezvous
(`init_multicast_for_block()` in
`torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu`) step by step, and
lets you fail any one of them on any one rank.

Written to debug https://github.com/pytorch/pytorch/pull/195596, which fixes
both failure modes below.

It exists to make two failure modes reproducible without rebuilding PyTorch:

1. **`cuMulticastBindMem` blocks** until all `numDevices` devices have joined
   the multicast object. There is no timeout, no async variant, and it is not a
   NCCL op, so no watchdog sees it. A rank that fails to join and skips its own
   bind does not rescue its peers -- they are already inside the driver.
   `cuMemMap` blocks on the same condition.
2. **Publishing success before the mapping exists** lets peers commit to a
   multicast address a failed rank does not have.

```bash
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --help
```

Needs at least 2 GPUs: `cuMulticastCreate` rejects `numDevices < 2`.

#### Failable steps

| step | driver call | notes |
|---|---|---|
| `create` | `cuMulticastCreate` | rank 0 only |
| `export` | `cuMemExportToShareableHandle` | rank 0 only |
| `import` | `cuMemImportFromShareableHandle` | |
| `add_device` | `cuMulticastAddDevice` | last non-blocking step |
| `bind` | `cuMulticastBindMem` | **blocks** until every device joined |
| `reserve` | `cuMemAddressReserve` | inside `map_block()` |
| `map` | `cuMemMap` | inside `map_block()`; **blocks** |
| `set_access` | `cuMemSetAccess` | inside `map_block()` |

Blocking calls run on a watchdog thread and report `BLOCKED >Ns` instead of
wedging the run. The summary is gathered from per-rank JSON files rather than a
final collective, because a rank parked in the driver never reaches another
collective -- and rank 0 is often one of them.

#### Baseline

```bash
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --verify
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --handle posix
```

`--verify` has rank 0 write once through the multicast VA; every rank then reads
its *own* physical allocation and should see `0xabcd`. `--handle posix` swaps
fabric handles for fd passing over a unix socket with `SCM_RIGHTS`.

#### The bind hang, and the agreement that prevents it

```bash
# no pre-bind agreement: 7 of 8 ranks wedge in cuMulticastBindMem
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --fail-rank 3 --fail-step add_device --no-pre-bind-agreement

# with it: every rank skips together
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --fail-rank 3 --fail-step add_device
```

To watch *who* actually detects the hang, send the failing rank on into the next
exchange instead of returning early. `--blocking-timeout` must exceed
`--pg-timeout`, or the blocked peers exit first and you see a connection error
rather than a timeout (the script warns if you get this backwards):

```bash
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --fail-rank 3 --fail-step add_device --no-pre-bind-agreement --no-short-circuit --pg-timeout 30 --blocking-timeout 45
```

```
[rank 3] ** failed to join; going to the final exchange, where nobody will arrive
[rank 3] ** exchange failed after 30.1s (limit 30s): RuntimeError. This rank --
             the one that was NOT blocked -- is the only one that reports anything.
[rank 0] 06! cuMulticastBindMem -> BLOCKED >45s      (and ranks 1,2,4,5,6,7)
agreement: DIVERGENT {'blocked_in_bind', 'timed_out_alone'}
```

The ranks that are actually stuck emit nothing. The error surfaces on the one
rank that was fine, and only after its own collective gives up.

#### Publishing success too early

`--agree-before-map` agrees right after the bind and maps afterwards:

```bash
# 7 ranks commit an address the 8th does not have
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --fail-rank 3 --fail-step map --agree-before-map

# agreeing after the map instead: everyone skips together
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --fail-rank 3 --fail-step map
```

Note that agreeing after the bind still catches *bind* failures cleanly -- it
only misses map failures. Same cost, strictly less coverage:

```bash
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --fail-rank 3 --fail-step bind --agree-before-map   # consistent
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --fail-rank 3 --fail-step map  --agree-before-map   # divergent
```

#### Partial teardown

```bash
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --fail-rank 3 --fail-step map          # VA reserved, not mapped
torchrun --nproc_per_node=8 recipes/multicast_rendezvous.py --fail-rank 3 --fail-step set_access   # VA reserved AND mapped
```

`live` in the summary shows exactly what each rank still held at exit. The
reserved-but-unmapped case is the one a caller cannot distinguish from a live
mapping, since `cuMemAddressReserve` writes the pointer before anything is
mapped to it.

### `recipes/symm_mem_rendezvous_leak.py`

Checks whether a rejected CUDA SymmetricMemory rendezvous leaks GPU memory
([pytorch/pytorch#199896](https://github.com/pytorch/pytorch/issues/199896)).
Unlike the other recipes this one drives real `torch.distributed._symmetric_memory`,
so it tests whatever torch you have installed.

`make_peer_alloc_info()` exports the block as a POSIX fd *before* it validates
the rendezvous requests, and only closes that fd after a successful import loop.
Any throw in between leaves the fd open, and the open fd keeps the physical
allocation alive after the tensor is freed. Fabric handles have no fd, so the
validation failures do not leak there.

```bash
torchrun --nproc_per_node=2 recipes/symm_mem_rendezvous_leak.py                 # torch's choice
torchrun --nproc_per_node=2 recipes/symm_mem_rendezvous_leak.py --handle posix  # force the fd path
```

| scenario | what happens |
|---|---|
| `ok` | control: own GPU per rank, same size, rendezvous succeeds |
| `size` | odd ranks allocate 2x, rejected by the `block_size` check |
| `overlap` | every rank on `cuda:0`, rejected by the overlapping-devices check (the issue's repro) |
| `overlap-close` | `overlap`, then close the `/dev/nvidia*` fds each failure left open |

Each scenario loops allocate, rendezvous (error caught), free, and tracks free
device memory and open fds. If `overlap` leaks and `overlap-close` does not, the
leftover exported fd is what pins the memory.

On fabric-capable boxes (H100/GB200 with an IMEX channel) torch always picks
fabric handles and has no knob to override it. `--handle posix` re-execs each
rank with a small `LD_PRELOAD` shim, built once with `cc` into
`~/.cache/nv-recipes/`, that makes `nvmlDeviceGetGpuFabricInfoV` report
NOT_SUPPORTED. The report states which path torch actually took, read from its
INFO log.

Rank 0 prints a verdict table and writes `report.md`, `report.json` and per-rank
torch logs to `reports/symm_mem_leak-<time>-<host>-<handle>/` (override with
`--outdir`). Other knobs: `--scenarios`, `--iters`, `--mib`.

### `recipes/symm_mem_alloc_rollback.py`

Checks the guards in [pytorch/pytorch#201284](https://github.com/pytorch/pytorch/pull/201284),
which rolls back a partially set-up CUDA VMM allocation in
`CUDASymmetricMemoryAllocator::alloc()` and `map_block()`. Replays `alloc()`,
`map_block()`, one peer of the import loop and `MulticastSetup::bind_and_map()`
over `cuda.bindings`, injects a failure at each step, and reports what is still
held afterwards.

```bash
python recipes/symm_mem_alloc_rollback.py                   # every path, every arm
python recipes/symm_mem_alloc_rollback.py --paths alloc --trace
```

One process, no torch, no torchrun. `peer` and `mc` need a second GPU (`--peer`).

Arms: `old` is the code before
[#201842](https://github.com/pytorch/pytorch/pull/201842) (teardown never frees
the VA), `base` is main with #201842, `fix` is `base` plus the PR, and
`fix-<guard>` is `fix` with one guard taken out. A guard whose removal changes no cell is
**redundant**. One that only matters for steps marked `*` (a sticky context
error, `std::bad_alloc`, or a second fault during rollback) is **defensive
only**.

| guard | what it does |
|---|---|
| `unwind` | `map_block` undoes its own reserve/map when a later step throws |
| `late_publish` | `map_block` writes `*ptr` only once the mapping is complete |
| `unmap_gate` | a rollback skips `cuMemAddressFree` when its `cuMemUnmap` failed |
| `alloc_release` | `alloc()` scope guard releases the handle until `AllocationRef` owns it |
| `alloc_unmap` | that guard also unmaps/frees `alloc_base` if it was published |
| `early_owner` | `AllocationRef` is built before the memset/sync instead of after |

What is held comes from a ledger of the driver calls, free device memory
before and after (`P<n>` = MiB pinned), and a reaper whose `CUDA_SUCCESS` is the
driver confirming each leftover was really live. A short "driver facts" section
first measures the driver behaviour the guards depend on, for example whether
`cuMemAddressFree` on a mapped VA is refused, and whether a freed VA comes back
from the next reserve. `--iters N` repeats each case to show leaks accumulating.
`--json` dumps every cell.
