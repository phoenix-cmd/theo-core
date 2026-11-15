#!/bin/bash
set -e
SLUG=theo-6c5-inputs
export THEO_ROOT=/kaggle/input/$SLUG/THEO
export THEO_OUT_B=/kaggle/working/M1_Qwen2.5-0.5B-Instruct-6c5-B
export THEO_OUT_C=/kaggle/working/M1_Qwen2.5-0.5B-Instruct-6c5-C
mkdir -p $THEO_OUT_B $THEO_OUT_C
cd $THEO_ROOT/theo-core/scripts/model_selection/phase6c5
python cloud_6c5.py
