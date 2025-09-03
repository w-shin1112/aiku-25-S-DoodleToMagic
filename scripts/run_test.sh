#!/bin/bash
# example: ./run_test.sh 1 "cute tiger pokemon character"
NUM=$1
USER_PROMPT=$2

if [ -z "$NUM" ] || [ -z "$USER_PROMPT" ]; then
  echo "Usage: ./run_test.sh [NUM] [USER_PROMPT]"
  exit 1
fi

echo ">>> Step 1: Running 2D inference with scribble-lora env"
conda run -n scribble-lora python /home/aikusrv04/Doodle/FINAL/inference_2d/pokemon_inference.py $NUM "$USER_PROMPT"

echo ">>> Step 2: Running 3D inference with Fin env"
conda run -n Fin python /home/aikusrv04/Doodle/FINAL/inference_3d/pokemon_inference.py $NUM

echo ">>> Pipeline finished successfully!"
