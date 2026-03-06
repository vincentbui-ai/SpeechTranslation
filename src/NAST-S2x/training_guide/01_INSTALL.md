# 01 — Installation Guide

Setup environment for training NAST-S2X on Vietnamese-English speech translation.

---

## Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU training)
- 32GB+ RAM recommended
- GPU with 16GB+ VRAM (for training)

---

## Step 1: Install Dependencies

### Install fairseq (with NAST modifications)

```bash
cd src/NAST-S2x

# Install fairseq
pip install -e fairseq/

# Verify installation
python -c "import fairseq; print(fairseq.__version__)"
```

### Install SimulEval (for streaming evaluation)

```bash
# Install customized SimulEval
pip install -e SimulEval/

# Verify installation
python -c "from simuleval import __version__; print(__version__)"
```

### Install Python dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Audio processing
pip install torchaudio soundfile librosa

# Text processing
pip install sentencepiece sacrebleu

# Utilities
pip install tqdm hydra-core omegaconf

# For inference
pip install transformers
```

---

## Step 2: Install NAST modules

The NAST-specific code is in `src/NAST-S2x/nast/`:

```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/NAST-S2x"

# Or install as editable
pip install -e src/NAST-S2x/
```

---

## Step 3: Verify Installation

Create a test script:

```bash
cat > test_install.py << 'EOF'
import torch
import fairseq
from nast.models import NASTTransformerModel
from nast.tasks import NATSpeechToUnitTask

print("✓ PyTorch version:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("✓ fairseq imported successfully")
print("✓ NAST modules imported successfully")
EOF

python test_install.py
```

Expected output:
```
✓ PyTorch version: 2.0.1+cu118
✓ CUDA available: True
✓ fairseq imported successfully
✓ NAST modules imported successfully
```

---

## Step 4: Download Pretrained Components

### Vocoder (Required for inference)

```bash
mkdir -p checkpoints/vocoder
cd checkpoints/vocoder

# Download from HuggingFace
python << 'EOF'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id='ICTNLP/NAST-S2X',
    filename='vocoder/mhubert_lyr11_km1000_en/config.json',
    local_dir='.'
)
hf_hub_download(
    repo_id='ICTNLP/NAST-S2X',
    filename='vocoder/mhubert_lyr11_km1000_en/g_00500000',
    local_dir='.'
)
EOF

cd ../..
```

### Optional: Download FR-EN checkpoints for fine-tuning

```bash
mkdir -p checkpoints/fr-en
cd checkpoints/fr-en

# Download offline checkpoint
python << 'EOF'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id='ICTNLP/NAST-S2X',
    filename='Offline.pt',
    local_dir='.'
)
EOF

cd ../..
```

---

## Step 5: Environment Variables

Add to your `.bashrc` or `.zshrc`:

```bash
# NAST-S2X paths
export NAST_ROOT="/path/to/SpeechTranslation/src/NAST-S2x"
export PYTHONPATH="${PYTHONPATH}:${NAST_ROOT}"

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your GPUs
```

---

## Troubleshooting

### Issue: `fairseq` not found

```bash
# Reinstall fairseq
cd src/NAST-S2x/fairseq
pip install -e .
```

### Issue: CUDA out of memory

```bash
# Reduce batch size in training scripts
# Edit train scripts and reduce --max-tokens value
```

### Issue: ImportError for NAST modules

```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/NAST-S2x"
```

---

## Next Steps

→ Proceed to [02_DATA_PREP.md](02_DATA_PREP.md) for data preparation.
