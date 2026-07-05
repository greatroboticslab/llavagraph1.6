#!/bin/bash
# Run this ONCE on a compute node (srun --pty bash) to pre-compile all Triton modules.
# After this, sbatch jobs will use the cached .so files and never hit GCC again.

set -e
echo "=== Step 1: Patch Triton build.py to show GCC errors ==="
python3 - << 'PYEOF'
import pathlib
p = pathlib.Path('/home/ya4v/.conda/envs/mamba_env/lib/python3.10/site-packages/triton/runtime/build.py')
src = p.read_text()
# Make GCC output visible (no capture)
if 'capture_output=True' in src:
    src = src.replace('_r = subprocess.run(cc_cmd, capture_output=True)', '_r = subprocess.run(cc_cmd)')
    p.write_text(src)
    print('Patched: GCC output will now be visible')
else:
    print('Already patched or different format')
lines = src.splitlines()
print('Lines 50-65:')
for i, l in enumerate(lines[49:65], 50):
    print(f'  {i}: {l}')
PYEOF

echo ""
echo "=== Step 2: Show what class compiles __triton_launcher.c ==="
python3 - << 'PYEOF'
import pathlib
p = pathlib.Path('/home/ya4v/.conda/envs/mamba_env/lib/python3.10/site-packages/triton/backends/nvidia/driver.py')
lines = p.read_text().splitlines()
print('Lines 675-700:')
for i, l in enumerate(lines[674:700], 675):
    print(f'  {i}: {l}')
PYEOF

echo ""
echo "=== Step 3: Pre-compile cuda_utils ==="
python3 -c "from triton.backends.nvidia.driver import CudaUtils; u = CudaUtils(); print('cuda_utils: OK')"

echo ""
echo "=== Step 4: Pre-compile __triton_launcher by running a tiny Triton kernel ==="
# @triton.jit requires source in a real .py file (not stdin/heredoc)
cat > /tmp/_triton_warmup.py << 'PYEOF'
import torch
import triton
import triton.language as tl

@triton.jit
def _noop(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, x, mask=mask)

n = 16
x = torch.zeros(n, device='cuda')
_noop[(1,)](x, n, BLOCK=n)
print('__triton_launcher: OK')
PYEOF
python3 /tmp/_triton_warmup.py

echo ""
echo "=== Done. All Triton modules cached. You can now exit and sbatch. ==="
