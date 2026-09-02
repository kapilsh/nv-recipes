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
