..
    SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

..
   Main Page
..

Welcome to NVIDIA Dynamo
========================

The NVIDIA Dynamo Platform is a high-performance, low-latency inference framework designed to serve all AI models—across any framework, architecture, or deployment scale.

.. admonition:: 💎 Discover the latest developments!
   :class: seealso

   This guide is a snapshot of a specific point in time. For the latest information, examples, and Release Assets, see the `Dynamo GitHub repository <https://github.com/ai-dynamo/dynamo/releases/latest>`_.

Quickstart
==========
.. include:: _includes/quick_start_local.rst

..
   Sidebar
..

.. toctree::
   :hidden:
   :caption: Getting Started

   Quickstart <self>
   Installation <_sections/installation>
   Support Matrix <support_matrix.md>
   Architecture <_sections/architecture>
   Examples <_sections/examples>

.. toctree::
   :hidden:
   :caption: Kubernetes Deployment

   Quickstart (K8s) <kubernetes/README.md>
   Installation Guide <kubernetes/installation.md>
   Operator <kubernetes/operator.md>
   API Reference <kubernetes/api-reference.md>
   Metrics <kubernetes/observability/metrics.md>
   Logging <kubernetes/observability/logging.md>
   Multinode <kubernetes/advanced/multinode.md>
   Minikube Setup <kubernetes/minikube.md>

.. toctree::
   :hidden:
   :caption: Components

   Backends <_sections/backends>
   Router <components/router/README>
   Planner <components/planner/overview>
   KVBM <components/kvbm/overview>
   NIXL <components/nixl/README>

.. toctree::
   :hidden:
   :caption: Performance

   Benchmarking <performance/benchmarking.md>
   Performance Tuning <performance/tuning.md>

.. toctree::
   :hidden:
   :caption: Observability

   Metrics <observability/metrics.md>
   Logging <observability/logging.md>
   Health Checks <observability/health-checks.md>

.. toctree::
   :hidden:
   :caption: Development

   Writing Custom Backends <development/custom-backend.md>
   Dynamo Run <development/dynamo-run.md>

.. toctree::
   :hidden:
   :caption: Reference

   Glossary <dynamo_glossary.md>
