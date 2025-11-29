#!/bin/bash
# Setup script for rlft_ms3 conda environment
# This environment is for ManiSkill3 tasks (separate from the rlft environment)

set -e

ENV_NAME="rlft_ms3"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "Setting up $ENV_NAME conda environment"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Create conda environment
echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install requirements
echo "Installing requirements from requirements_ms3.txt..."
pip install -r requirements_ms3.txt

# Verify ManiSkill3 installation
echo ""
echo "Verifying ManiSkill3 installation..."
python -c "import mani_skill; print(f'ManiSkill version: {mani_skill.__version__}')"

# Test Vulkan (optional)
echo ""
echo "Testing Vulkan support..."
if command -v vulkaninfo &> /dev/null; then
    vulkaninfo --summary 2>/dev/null || echo "Warning: Vulkan info not available (may still work)"
else
    echo "Note: vulkaninfo not found. Install vulkan-tools to verify Vulkan setup."
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test ManiSkill3 PickCube environment:"
echo "  python -c \"import gymnasium as gym; import mani_skill.envs; env = gym.make('PickCube-v1', num_envs=1, obs_mode='state'); print('Success!')\""
echo ""
