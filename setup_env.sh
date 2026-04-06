# setup_env.sh  (Mac/Linux)
# Windows users: adapt to setup_env.bat (replace conda activate with conda.bat)

#!/bin/bash
set -e

ENV_NAME="EDU"
echo "Creating conda environment: $ENV_NAME"

conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# Core deep learning
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Segmentation models
pip install segmentation-models-pytorch
pip install transformers accelerate

# Augmentation + imaging
pip install albumentations opencv-python-headless pillow

# Hyperparameter tuning
pip install optuna optuna-dashboard

# TigerGraph
pip install pyTigerGraph

# Metrics + visualization
pip install scikit-learn seaborn matplotlib

# Utilities
pip install tqdm rich

echo ""
echo "✅ Environment '$ENV_NAME' ready."
echo "Activate with: conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "  1. Edit TG_HOST / TG_USERNAME / TG_PASSWORD in tigergraph_integration.py"
echo "  2. Run: python train.py --data_root ./dataset --arch segformer-b4 --tune"
echo "  3. Run: python test.py --checkpoint ./runs/best.pth --img_dir ./dataset/testImages"
echo "  4. Run: python tigergraph_integration.py --setup --mask_dir ./dataset/train/segmented"
