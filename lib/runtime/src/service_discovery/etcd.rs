// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ETCD implementation of the ServiceDiscovery interface

use super::{
    Instance, InstanceEvent, InstanceEventStream, InstanceHandle, InstanceStatus, ServiceDiscovery,
    ServiceRegistry,
};
use crate::{transports::etcd, Result};
use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, pin::Pin, sync::Arc};
use tokio::sync::mpsc;

/// ETCD-specific metadata for instances
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EtcdInstanceMetadata {
    /// Instance metadata as key-value pairs
    pub metadata: HashMap<String, serde_json::Value>,
    /// Instance readiness status
    pub status: InstanceStatus,
    /// Network address of the instance
    pub address: String,
}

/// ETCD implementation of InstanceHandle
pub struct EtcdInstanceHandle {
    /// Unique instance identifier (lease ID in hex format)
    instance_id: String,
    /// ETCD client for operations
    etcd_client: etcd::Client,
    /// ETCD key for this instance
    etcd_key: String,
    /// Current metadata
    metadata: tokio::sync::RwLock<EtcdInstanceMetadata>,
}

impl EtcdInstanceHandle {
    pub fn new(
        instance_id: String,
        etcd_client: etcd::Client,
        etcd_key: String,
        address: String,
    ) -> Self {
        let metadata = EtcdInstanceMetadata {
            metadata: HashMap::new(),
            status: InstanceStatus::NotReady,
            address,
        };

        Self {
            instance_id,
            etcd_client,
            etcd_key,
            metadata: tokio::sync::RwLock::new(metadata),
        }
    }

    /// Update the instance data in ETCD
    async fn update_etcd(&self) -> Result<()> {
        let metadata = self.metadata.read().await;
        let serialized = serde_json::to_vec(&*metadata)?;
        self.etcd_client
            .kv_put(&self.etcd_key, serialized, None)
            .await
    }
}

#[async_trait]
impl InstanceHandle for EtcdInstanceHandle {
    fn instance_id(&self) -> &str {
        &self.instance_id
    }

    async fn set_metadata(&self, metadata: HashMap<String, serde_json::Value>) -> Result<()> {
        {
            let mut current_metadata = self.metadata.write().await;
            current_metadata.metadata = metadata;
        }
        self.update_etcd().await
    }

    async fn set_ready(&self, status: InstanceStatus) -> Result<()> {
        {
            let mut current_metadata = self.metadata.write().await;
            current_metadata.status = status;
        }
        self.update_etcd().await
    }
}

/// ETCD implementation of ServiceDiscovery and ServiceRegistry
#[derive(Clone)]
pub struct EtcdServiceDiscovery {
    etcd_client: etcd::Client,
    /// Base path for service discovery keys in ETCD
    base_path: String,
}

impl EtcdServiceDiscovery {
    /// Create a new ETCD service discovery client
    pub fn new(etcd_client: etcd::Client) -> Self {
        Self {
            etcd_client,
            base_path: "dynamo://services".to_string(),
        }
    }

    /// Create a new ETCD service discovery client with custom base path
    pub fn with_base_path(etcd_client: etcd::Client, base_path: String) -> Self {
        Self {
            etcd_client,
            base_path,
        }
    }

    /// Get the ETCD key prefix for a given namespace and component
    fn get_key_prefix(&self, namespace: &str, component: &str) -> String {
        format!("{}/{}/{}/", self.base_path, namespace, component)
    }

    /// Parse instance from ETCD key-value pair
    fn parse_instance(&self, kv: &etcd::KeyValue, namespace: &str, component: &str) -> Result<Option<Instance>> {
        let key = String::from_utf8_lossy(kv.key());
        let value = kv.value();

        // Extract instance ID from key (last part after final '/')
        let instance_id = key
            .split('/')
            .last()
            .ok_or_else(|| crate::error!("Invalid ETCD key format: {}", key))?
            .to_string();

        // Parse metadata
        let etcd_metadata: EtcdInstanceMetadata = serde_json::from_slice(value)
            .map_err(|e| crate::error!("Failed to parse instance metadata: {}", e))?;

        // Only return instances that are ready
        if etcd_metadata.status != InstanceStatus::Ready {
            return Ok(None);
        }

        Ok(Some(Instance::new(
            instance_id,
            etcd_metadata.address,
            namespace.to_string(),
            component.to_string(),
        )))
    }
}

#[async_trait]
impl ServiceDiscovery for EtcdServiceDiscovery {
    async fn list_instances(&self, namespace: &str, component: &str) -> Result<Vec<Instance>> {
        let prefix = self.get_key_prefix(namespace, component);
        let kvs = self.etcd_client.kv_get_prefix(&prefix).await?;

        let mut instances = Vec::new();
        for kv in kvs {
            if let Some(instance) = self.parse_instance(&kv, namespace, component)? {
                instances.push(instance);
            }
        }

        Ok(instances)
    }

    async fn watch(&self, namespace: &str, component: &str) -> Result<InstanceEventStream> {
        let prefix = self.get_key_prefix(namespace, component);
        let watcher = self.etcd_client.kv_get_and_watch_prefix(&prefix).await?;
        let (_prefix, _watcher, rx) = watcher.dissolve();

        let namespace = namespace.to_string();
        let component = component.to_string();

        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(event) = rx.recv().await {
                match event {
                    etcd::WatchEvent::Put(kv) => {
                        // Parse instance inline to avoid lifetime issues
                        let key = String::from_utf8_lossy(kv.key());
                        let value = kv.value();

                        // Extract instance ID from key (last part after final '/')
                        if let Some(instance_id) = key.split('/').last() {
                            // Parse metadata
                            if let Ok(etcd_metadata) = serde_json::from_slice::<EtcdInstanceMetadata>(value) {
                                // Only return instances that are ready
                                if etcd_metadata.status == InstanceStatus::Ready {
                                    let instance = Instance::new(
                                        instance_id.to_string(),
                                        etcd_metadata.address,
                                        namespace.clone(),
                                        component.clone(),
                                    );
                                    yield InstanceEvent::Added(instance);
                                }
                            }
                        }
                    }
                    etcd::WatchEvent::Delete(kv) => {
                        // For delete events, we need to reconstruct the instance from the key
                        let key = String::from_utf8_lossy(kv.key());
                        if let Some(instance_id) = key.split('/').last() {
                            // We don't have the address for deleted instances, so use empty string
                            let instance = Instance::new(
                                instance_id.to_string(),
                                "".to_string(),
                                namespace.clone(),
                                component.clone(),
                            );
                            yield InstanceEvent::Removed(instance);
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl ServiceRegistry for EtcdServiceDiscovery {
    async fn register_instance(
        &self,
        namespace: &str,
        component: &str,
    ) -> Result<Arc<dyn InstanceHandle>> {
        // Use the primary lease ID as the instance ID
        let lease_id = self.etcd_client.lease_id();
        let instance_id = format!("{:x}", lease_id);

        // Create ETCD key for this instance
        let etcd_key = format!("{}{}", self.get_key_prefix(namespace, component), instance_id);

        // TODO: Get actual network address - for now use placeholder
        let address = "localhost:8080".to_string();

        let handle = Arc::new(EtcdInstanceHandle::new(
            instance_id,
            self.etcd_client.clone(),
            etcd_key,
            address,
        ));

        // Initialize the instance in ETCD with NotReady status
        handle.update_etcd().await?;

        Ok(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Runtime;

    #[tokio::test]
    #[ignore = "requires ETCD server"]
    async fn test_etcd_service_discovery() -> Result<()> {
        // This test requires a running ETCD server
        let runtime = Runtime::single_threaded()?;
        let etcd_options = etcd::ClientOptions::default();
        let etcd_client = etcd::Client::new(etcd_options, runtime).await?;

        let discovery = EtcdServiceDiscovery::new(etcd_client);

        // Test registration
        let handle = discovery.register_instance("test-ns", "test-component").await?;
        
        // Set metadata
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), serde_json::Value::String("value1".to_string()));
        handle.set_metadata(metadata).await?;

        // Set ready
        handle.set_ready(InstanceStatus::Ready).await?;

        // Test discovery
        let instances = discovery.list_instances("test-ns", "test-component").await?;
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].id, handle.instance_id());

        Ok(())
    }
}

