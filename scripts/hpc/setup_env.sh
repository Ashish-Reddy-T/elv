#!/bin/bash
# =================================================================
# SpatialVLM — one-time HPC environment setup
# =================================================================
# Run this ONCE inside a Singularity shell to bootstrap conda + deps
# into your overlay.
#
# Prerequisites:
#   1. Create overlay:
#        cd /scratch/$USER
#        cp /share/apps/overlay-fs-ext3/overlay-50G-10M.ext3.gz .
#        gunzip overlay-50G-10M.ext3.gz
#        mv overlay-50G-10M.ext3 spatialvlm_env.ext3
#
#   2. Launch container (rw mode for setup):
#        singularity exec --nv \
#          --overlay /scratch/$USER/spatialvlm_env.ext3:rw \
#          /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
#          /bin/bash
#
#   3. Inside container, cd to repo and run:
#        bash scripts/hpc/setup_env.sh
# =================================================================
set -euo pipefail

echo "=== SpatialVLM HPC environment setup ==="

# Step 1: Install Miniconda (skip if already present)
if [ ! -d /ext3/miniconda3 ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /ext3/miniconda3
    rm /tmp/miniconda.sh
else
    echo "Miniconda already installed, skipping."
fi

# Step 2: Create activation script
cat << 'EOF' > /ext3/env.sh
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
conda activate spatialvlm
EOF

# Step 3: Create conda environment
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH

if conda env list | grep -q spatialvlm; then
    echo "Conda env 'spatialvlm' exists, updating..."
    conda activate spatialvlm
else
    echo "Creating conda env 'spatialvlm'..."
    conda create -n spatialvlm python=3.10 -y
    conda activate spatialvlm
fi

# Step 4: Install PyTorch (CUDA 12.1)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 5: Install SpatialVLM + dependencies
echo "Installing SpatialVLM..."
pip install -e .
pip install -e REPOS/geometric-algebra-transformer --no-deps

# Step 6: wandb + ipykernel (for Jupyter use)
pip install wandb ipykernel

# Step 7: Verify
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
python -c "import spatialvlm; print('spatialvlm: OK')"
python -c "import gatr; print('gatr: OK')"

echo ""
echo "=== Setup complete ==="
echo "To activate in future sessions: source /ext3/env.sh"
