#!/bin/sh

# Check that the env folder exists or create it
if [ -d ~/envs ]
then
    echo "Found existing envs folder"
else
    echo "Did not find envs folder, creating"
    mkdir ~/envs
fi

# Check if conda environment exists. if it does, remove it.
if [ -d "~/envs/fedomics_python_$1" ]
then
    echo "Found existing fedomics conda environment, removing"
    conda env remove --prefix "~/envs/fedomics_python_$1" -y
fi
conda init bash
. ~/.bashrc
#
echo "Creating environment"
yes | conda create --prefix "~/envs/fedomics_python_$1" python="$1"
echo "Created env fedomics_python_$1"
eval "$(conda shell.bash hook)"
conda activate "~/envs/fedomics_python_$1"
