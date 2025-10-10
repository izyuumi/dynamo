:orphan:

..
    SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

.. This hidden toctree includes readmes etc that aren't meant to be in the main table of contents but should be accounted for in the sphinx project structure


.. toctree::
   :maxdepth: 2
   :hidden:

   components/nixl/connector.md
   components/nixl/descriptor.md
   components/nixl/device.md
   components/nixl/device_kind.md
   components/nixl/operation_status.md
   components/nixl/rdma_metadata.md
   components/nixl/readable_operation.md
   components/nixl/writable_operation.md
   components/nixl/read_operation.md
   components/nixl/write_operation.md
   components/nixl/README.md

   kubernetes/api-reference.md
   kubernetes/create-deployment.md
   kubernetes/advanced/fluxcd-gitops.md
   kubernetes/advanced/grove.md
   kubernetes/advanced/fluid-caching.md
   kubernetes/README.md

   development/dynamo-run.md
   observability/metrics.md
   components/kvbm/vllm-guide.md
   components/kvbm/trtllm-guide.md

   components/router/kv-routing.md
   components/planner/load-planner.md
   architecture/request_migration.md

   backends/trtllm/multinode/multinode-examples.md
   backends/trtllm/multinode/multinode-multimodal-example.md
   backends/trtllm/llama4_plus_eagle.md
   backends/trtllm/kv-cache-transfer.md
   backends/trtllm/multimodal_support.md
   backends/trtllm/multimodal_epd.md
   backends/trtllm/gemma3_sliding_window_attention.md
   backends/trtllm/gpt-oss.md

   backends/sglang/multinode-examples.md
   backends/sglang/dsr1-wideep-gb200.md
   backends/sglang/dsr1-wideep-h100.md
   backends/sglang/expert-distribution-eplb.md
   backends/sglang/gpt-oss.md
   backends/sglang/multimodal_epd.md
   backends/sglang/sgl-hicache-example.md

   examples/README.md
   examples/runtime/hello_world/README.md

   architecture/distributed_runtime.md
   architecture/dynamo_flow.md

   backends/vllm/deepseek-r1.md
   backends/vllm/gpt-oss.md
   backends/vllm/multi-node.md


..   TODO: architecture/distributed_runtime.md and architecture/dynamo_flow.md
     have some outdated names/references and need a refresh.
