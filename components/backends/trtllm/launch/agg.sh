#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables with defaults
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-0.6B"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"engine_configs/agg.yaml"}
export MODALITY=${MODALITY:-"text"}
# If you want to use multimodal, set MODALITY to "multimodal"
#export MODALITY=${MODALITY:-"multimodal"}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID 2>/dev/null || true
    wait $DYNAMO_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM


# run frontend
python3 -m dynamo.frontend --http-port 8000 &
DYNAMO_PID=$!

# run worker
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --modality "$MODALITY" \
  --extra-engine-args "$AGG_ENGINE_ARGS"
