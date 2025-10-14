// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub fn get_barrier_id_prefix(module_id: Option<String>) -> String {
    let env_prefix = std::env::var("DYN_KVBM_BARRIER_ID_PREFIX")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "kvbm".to_string());

    if let Some(module_id) = module_id {
        format!("{}-{}", env_prefix, module_id)
    } else {
        env_prefix
    }
}
