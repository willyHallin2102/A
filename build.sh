#!/usr/bin/env bash
set -euo pipefail

VENV="venv"

# Allow override: USE 3.10 python 3.13 is crap
# Massive compatible issues NOTE: CALL ./build.sh --python3.10 
PYTHON="${1:-python3}"

command_exists() {
    command -v "$1" &>/dev/null
}

# Ensure python exists
if ! command_exists "$PYTHON"; then
    echo "[Error]: ${PYTHON} not found, this is required to be installed" >&2
    exit 1
fi

# Ensure git exists (needed for Sionna)
if ! command_exists git; then
    echo "[Error]: git is required but not installed" >&2
    exit 1
fi

# Handle existing virtual environment, annoyance, reinstallations
if [ -d "$VENV" ]; then
    echo "[Warning]: Virtual environment '$VENV' already exists."
    read -rp "Remove and recreate it? [y/N]: " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "[Info]: Removing existing virtual environment..."
        rm -rf "$VENV"
    else
        echo "[Info]: Reusing existing virtual environment."
    fi
fi

# Create venv if missing
if [ ! -d "$VENV" ]; then
    echo "[Info]: Creating virtual environment..."
    "$PYTHON" -m venv "$VENV"
fi

# Activate venv... Must activate environment prior add dependencies.
echo "[Info]: Activating the virtual environment '$VENV'..."

# shellcheck disable=SC1090
source "$VENV/bin/activate"

# Upgrade pip and core tools
echo "[Info]: Upgrading pip, setuptools, wheel..."
python -m ensurepip --upgrade || true
pip install --upgrade pip setuptools wheel

# Install dependencies (could use requirements.txt for consistency =))
if [ -f "requirements.txt" ]; then
    echo "[Warning]: requirements.txt found"
    read -rp "Install from requirements.txt? [Y/n]: " use_requirements
    if [[ ! "$use_requirements" =~ ^[Nn]$ ]]; then
        pip install -r requirements.txt
    fi
else
    echo "[Info]: Installing base dependencies..."
    pip install pandas seaborn scikit-learn tqdm orjson pyarrow
fi

# Install Sionna only if missing
if ! pip show sionna &>/dev/null; then
    echo "[Info]: Installing Sionna from GitHub..."
    pip install git+https://github.com/NVlabs/sionna.git@main
else
    echo "[Ok]: Sionna already installed."
fi

# Always update requirements else consistency issues 
read -rp "Update requirements.txt with current packages? [Y/n]: " update_requirements
if [[ ! "$update_requirements" =~ ^[Nn]$ ]]; then
    echo "[Info]: Updating requirements.txt..."
    pip freeze > requirements.txt
fi

echo "[Success]: Setup completed successfully!"
deactivate
