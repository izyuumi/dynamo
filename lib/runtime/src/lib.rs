// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo

#![allow(dead_code)]
#![allow(unused_imports)]

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
};

pub use anyhow::{
    Context as ErrorContext, Error, Ok as OK, Result, anyhow as error, bail as raise,
};

use async_once_cell::OnceCell;

mod config;
pub use config::RuntimeConfig;

pub mod component;
pub mod compute;
pub mod discovery;
pub mod engine;
pub mod health_check;
pub mod system_status_server;
pub use system_status_server::SystemStatusServerInfo;
pub mod instances;
pub mod logging;
pub mod metrics;
pub mod pipeline;
pub mod prelude;
pub mod protocols;
pub mod runnable;
pub mod runtime;
pub mod service;
pub mod slug;
pub mod storage;
pub mod system_health;
pub mod traits;
pub mod transports;
pub mod utils;
pub mod worker;

pub mod distributed;
pub use distributed::distributed_test_utils;
pub use futures::stream;
pub use system_health::{HealthCheckTarget, SystemHealth};
pub use tokio_util::sync::CancellationToken;
pub use worker::Worker;

use crate::{
    metrics::prometheus_names::distributed_runtime, storage::key_value_store::KeyValueStore,
};

use component::{Endpoint, InstanceSource};
use utils::GracefulShutdownTracker;

use config::HealthStatus;

/// Types of Tokio runtimes that can be used to construct a Dynamo [Runtime].
#[derive(Clone)]
enum RuntimeType {
    Shared(Arc<tokio::runtime::Runtime>),
    External(tokio::runtime::Handle),
}

/// Local [Runtime] which provides access to shared resources local to the physical node/machine.
#[derive(Debug, Clone)]
pub struct Runtime {
    id: Arc<String>,
    primary: RuntimeType,
    secondary: RuntimeType,
    cancellation_token: CancellationToken,
    endpoint_shutdown_token: CancellationToken,
    graceful_shutdown_tracker: Arc<GracefulShutdownTracker>,
    compute_pool: Option<Arc<compute::ComputePool>>,
    block_in_place_permits: Option<Arc<tokio::sync::Semaphore>>,
}

/// Type alias for runtime callback functions to reduce complexity
///
/// This type represents an Arc-wrapped callback function that can be:
/// - Shared efficiently across multiple threads and contexts
/// - Cloned without duplicating the underlying closure
/// - Used in generic contexts requiring 'static lifetime
///
/// The Arc wrapper is included in the type to make sharing explicit.
type PrometheusUpdateCallback = Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync + 'static>;

/// Type alias for exposition text callback functions that return Prometheus text
type PrometheusExpositionFormatCallback =
    Arc<dyn Fn() -> anyhow::Result<String> + Send + Sync + 'static>;

/// Structure to hold Prometheus registries and associated callbacks for a given hierarchy
pub struct MetricsRegistryEntry {
    /// The Prometheus registry for this prefix
    pub prometheus_registry: prometheus::Registry,
    /// List of update callbacks invoked before metrics are scraped
    pub prometheus_update_callbacks: Vec<PrometheusUpdateCallback>,
    /// List of callbacks that return Prometheus exposition text to be appended to metrics output
    pub prometheus_expfmt_callbacks: Vec<PrometheusExpositionFormatCallback>,
}

impl MetricsRegistryEntry {
    /// Create a new metrics registry entry with an empty registry and no callbacks
    pub fn new() -> Self {
        Self {
            prometheus_registry: prometheus::Registry::new(),
            prometheus_update_callbacks: Vec::new(),
            prometheus_expfmt_callbacks: Vec::new(),
        }
    }

    /// Add a callback function that receives a reference to any MetricsRegistry
    pub fn add_prometheus_update_callback(&mut self, callback: PrometheusUpdateCallback) {
        self.prometheus_update_callbacks.push(callback);
    }

    /// Add an exposition text callback that returns Prometheus text
    pub fn add_prometheus_expfmt_callback(&mut self, callback: PrometheusExpositionFormatCallback) {
        self.prometheus_expfmt_callbacks.push(callback);
    }

    /// Execute all update callbacks and return their results
    pub fn execute_prometheus_update_callbacks(&self) -> Vec<anyhow::Result<()>> {
        self.prometheus_update_callbacks
            .iter()
            .map(|callback| callback())
            .collect()
    }

    /// Execute all exposition text callbacks and return their concatenated text
    pub fn execute_prometheus_expfmt_callbacks(&self) -> String {
        let mut result = String::new();
        for callback in &self.prometheus_expfmt_callbacks {
            match callback() {
                Ok(text) => {
                    if !text.is_empty() {
                        if !result.is_empty() && !result.ends_with('\n') {
                            result.push('\n');
                        }
                        result.push_str(&text);
                    }
                }
                Err(e) => {
                    tracing::error!("Error executing exposition text callback: {}", e);
                }
            }
        }
        result
    }

    /// Returns true if a metric with the given name already exists in the Prometheus registry
    pub fn has_metric_named(&self, metric_name: &str) -> bool {
        self.prometheus_registry
            .gather()
            .iter()
            .any(|mf| mf.name() == metric_name)
    }
}

impl Default for MetricsRegistryEntry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MetricsRegistryEntry {
    fn clone(&self) -> Self {
        Self {
            prometheus_registry: self.prometheus_registry.clone(),
            prometheus_update_callbacks: Vec::new(), // Callbacks cannot be cloned, so we start with an empty list
            prometheus_expfmt_callbacks: Vec::new(), // Callbacks cannot be cloned, so we start with an empty list
        }
    }
}

/// Distributed [Runtime] which provides access to shared resources across the cluster, this includes
/// communication protocols and transports.
#[derive(Clone)]
pub struct DistributedRuntime {
    // local runtime
    runtime: Runtime,

    // we might consider a unifed transport manager here
    etcd_client: Option<transports::etcd::Client>,
    nats_client: transports::nats::Client,
    store: Arc<dyn KeyValueStore>,
    tcp_server: Arc<OnceCell<Arc<transports::tcp::server::TcpStreamServer>>>,
    system_status_server: Arc<OnceLock<Arc<system_status_server::SystemStatusServerInfo>>>,

    // local registry for components
    // the registry allows us to use share runtime resources across instances of the same component object.
    // take for example two instances of a client to the same remote component. The registry allows us to use
    // a single endpoint watcher for both clients, this keeps the number background tasking watching specific
    // paths in etcd to a minimum.
    component_registry: component::Registry,

    // Will only have static components that are not discoverable via etcd, they must be know at
    // startup. Will not start etcd.
    is_static: bool,

    instance_sources: Arc<tokio::sync::Mutex<HashMap<Endpoint, Weak<InstanceSource>>>>,

    // Health Status
    system_health: Arc<std::sync::Mutex<SystemHealth>>,

    // This map associates metric prefixes with their corresponding Prometheus registries and callbacks.
    // Uses RwLock for better concurrency - multiple threads can read (execute callbacks) simultaneously.
    hierarchy_to_metricsregistry: Arc<std::sync::RwLock<HashMap<String, MetricsRegistryEntry>>>,
}
