#!/usr/bin/env bash

# This script installs Miniconda and creates the 'tf2' conda environment
# defined in environment.yml. It is intended for reproducing the environment
# used in the Hands-on Machine Learning project.

set -e

ENV_NAME="${1:-tf2}"
MINICONDA_DIR="$HOME/miniconda3"

# Download and install Miniconda if it is not already installed
if [ ! -x "$MINICONDA_DIR/bin/conda" ]; then
  echo "Installing Miniconda to $MINICONDA_DIR..."
  curl -L -o /tmp/miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
  rm /tmp/miniconda.sh
fi

export PATH="$MINICONDA_DIR/bin:$PATH"

# Initialize conda (needed if running non-interactively)
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# Create the environment from environment.yml
conda env create -f environment.yml -n "$ENV_NAME"

echo "\nEnvironment '$ENV_NAME' created. Activate it with:\n"
echo "  source $MINICONDA_DIR/bin/activate $ENV_NAME"
