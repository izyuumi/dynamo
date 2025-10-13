#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID 2>/dev/null || true
    wait $DYNAMO_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM


# run ingress
python3 -m dynamo.frontend --http-port=8000 &
DYNAMO_PID=$!

# run worker
python3 -m dynamo.sglang \
  --embedding-worker \
  --model-path Qwen/Qwen3-Embedding-4B \
  --served-model-name Qwen/Qwen3-Embedding-4B \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --use-sglang-tokenizer
