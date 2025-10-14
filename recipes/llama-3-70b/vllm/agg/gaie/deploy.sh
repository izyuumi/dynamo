#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env bash
set -Eeuo pipefail

# ===== Config (env overridable) =====
: "${NAMESPACE:=dynamo}"
: "${EPP_IMAGE:?EPP_IMAGE must be set, e.g. nvcr.io/your/epp:tag}"

# ===== Pre-flight checks =====
command -v helm >/dev/null 2>&1 || { echo "ERROR: helm not found"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "ERROR: kubectl not found"; exit 1; }

# ===== Namespace ensure =====
if ! kubectl get ns "$NAMESPACE" >/dev/null 2>&1; then
  kubectl create namespace "$NAMESPACE"
fi

# ===== Setup GAIE =====


# ===== Apply manifests =====
kubectl apply -f ./k8s-manifests -n "$NAMESPACE"

echo "Done."

