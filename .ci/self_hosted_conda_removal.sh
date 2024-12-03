#!/bin/sh

# Check if conda environment exists. if it does, remove it.
if [ -d "~/envs/fedomics_python_$1" ]
then
    echo "Found existing fedomics conda environment, removing"
    conda env remove --prefix "~/envs/fedomics_python_$1" -y
fi
