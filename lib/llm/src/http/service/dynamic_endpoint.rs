// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{RouteDoc, service_v2};
use crate::types::Annotated;
use axum::{
    Json, Router,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
};
use dynamo_runtime::component::Client;
use dynamo_runtime::instances::list_all_instances;
use dynamo_runtime::{pipeline::PushRouter, stream::StreamExt};
use std::sync::Arc;

pub fn dynamic_endpoint_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let wildcard_path = "/{*path}";
    let path = path.unwrap_or_else(|| wildcard_path.to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::POST, &path)];

    let router = Router::new()
        .route(&path, post(dynamic_endpoint_handler))
        .with_state(state);

    (docs, router)
}

/// Dynamic endpoint handler that discovers component instances from the discovery plane and fans out
/// requests to all instances that registered the matching HTTP endpoint path.
///
/// Example: POST to `/get_model_info` discovers all instances with `http_endpoint_path = "/get_model_info"`,
/// queries each one, and returns `{"responses": [instance1_result, instance2_result, ...]}`.
///
/// Returns 404 if no instances have registered the endpoint.
async fn inner_dynamic_endpoint_handler(
    state: Arc<service_v2::State>,
    path: String,
) -> Result<impl IntoResponse, (StatusCode, &'static str)> {
    let drt = state.distributed_runtime().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Failed to get distributed runtime",
    ))?;
    let etcd_client = drt.etcd_client().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Failed to get etcd client",
    ))?;

    let instances = list_all_instances(&etcd_client)
        .await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Failed to get instances"))?;

    let dynamic_endpoints = instances
        .iter()
        .filter_map(|instance| instance.http_endpoint_path.clone())
        .collect::<Vec<String>>();

    let fmt_path = format!("/{}", &path);
    if !dynamic_endpoints.contains(&fmt_path) {
        return Err((StatusCode::NOT_FOUND, "Endpoint not found"));
    }

    let target_instances = instances
        .iter()
        .filter(|instance| instance.http_endpoint_path == Some(fmt_path.clone()))
        .collect::<Vec<_>>();

    let mut target_clients: Vec<Client> = Vec::new();
    for instance in target_instances {
        let ns = drt
            .namespace(instance.namespace.clone())
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Failed to get namespace"))?;
        let c = ns
            .component(instance.component.clone())
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Failed to get component"))?;
        let ep = c.endpoint(instance.endpoint.clone());
        let client = ep
            .client()
            .await
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Failed to get client"))?;
        client.wait_for_instances().await.map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to wait for instances",
            )
        })?;
        target_clients.push(client);
    }

    let mut all_responses = Vec::new();
    for client in target_clients {
        let router =
            PushRouter::<(), Annotated<serde_json::Value>>::from_client(client, Default::default())
                .await
                .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Failed to get router"))?;

        let mut stream = router.round_robin(().into()).await.map_err(|e| {
            tracing::error!("Failed to route: {:?}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to route")
        })?;

        while let Some(resp) = stream.next().await {
            all_responses.push(resp);
        }
    }

    Ok(Json(serde_json::json!({
        "responses": all_responses
    })))
}

async fn dynamic_endpoint_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
    axum::extract::Path(path): axum::extract::Path<String>,
) -> impl IntoResponse {
    inner_dynamic_endpoint_handler(state, path)
        .await
        .map_err(|(status_code, err_string)| {
            (
                status_code,
                Json(serde_json::json!({
                    "message": err_string
                })),
            )
        })
}
