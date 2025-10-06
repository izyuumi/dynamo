#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Unset all environment variables from multimodal_script.env

unset MODALITY
unset IMAGE
unset MOUNTS
unset MODEL_PATH
unset SERVED_MODEL_NAME
unset NUM_PREFILL_NODES
unset NUM_DECODE_NODES
unset NUM_GPUS_PER_NODE
unset PREFILL_ENGINE_CONFIG
unset DECODE_ENGINE_CONFIG

echo "All multimodal environment variables have been unset."
