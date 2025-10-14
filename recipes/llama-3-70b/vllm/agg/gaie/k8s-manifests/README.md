# Kubernetes Manifests for Dynamo GAIE with EPP

This directory contains converted Kubernetes Custom Resource (CR) manifests for deploying Dynamo GAIE with the Dynamo EPP image (`epp.useDynamo=true` mode).

## Files

1. **cluster-role.yaml** - ClusterRole for EPP to access inference resources and pods
2. **cluster-role-binding.yaml** - Binds the ClusterRole to the default service account
3. **epp-configmap.yaml** - ConfigMap containing EPP configuration for Dynamo routing
4. **epp-deployment.yaml** - Deployment for the Dynamo EPP container
5. **epp-service.yaml** - Service exposing the EPP on port 9002
6. **inference-pool.yaml** - InferencePool CR for managing backend pods
7. **inference-model.yaml** - InferenceModel CR defining the model
8. **http-route.yaml** - HTTPRoute for routing traffic through the gateway

## Before Applying

**IMPORTANT**: You must update the namespace in all manifest files before applying them. Search for `dynamo` and replace with your actual deployment namespace.

Files that need namespace updates:

- cluster-role-binding.yaml
- epp-configmap.yaml
- epp-deployment.yaml (multiple locations)
- epp-service.yaml
- inference-pool.yaml
- inference-model.yaml
- http-route.yaml (multiple locations)

### Additional Customizations

If you need to customize other values, edit the following:

#### Model Configuration

- **inference-model.yaml**: Update `modelName` if using a different model
- **inference-pool.yaml**: Update `selector.nvidia.com/dynamo-namespace` if your Dynamo namespace differs

#### Platform Configuration

- **epp-deployment.yaml**:
  - Update `ETCD_ENDPOINTS` if using a different platform name or namespace
  - Update `NATS_SERVER` if using a different platform name or namespace
  - Update `DYN_NAMESPACE` if your Dynamo namespace differs

#### Gateway Configuration

- **http-route.yaml**: Update `gatewayName` if using a different gateway

## Deployment

### Prerequisites

1. Ensure the Gateway API CRDs are installed
2. Ensure the Inference Gateway Extension CRDs are installed
3. Ensure the Dynamo platform is deployed (ETCD, NATS)
4. Ensure your Dynamo backend pods have the appropriate labels:
   - `nvidia.com/dynamo-component: Frontend`
   - `nvidia.com/dynamo-namespace: vllm-agg`

### Apply the Manifests

Kustomize makes it easy to customize the namespace and apply all resources at once:

```bash
# Set your namespace
cd /path/to/k8s-manifests
kustomize edit set namespace -n ${NAMESPACE}

# Preview what will be applied
kubectl kustomize .

# Apply all resources
kubectl apply -k .
```

```bash
kubectl apply -f . -n ${NAMESPACE}
```

### Verify Deployment

```bash
# Check EPP deployment
kubectl get deployment qwen-epp -n YOUR-NAMESPACE
kubectl get pods -l app=qwen-epp -n YOUR-NAMESPACE

# Check inference resources
kubectl get inferencepool qwen-pool -n YOUR-NAMESPACE
kubectl get inferencemodel qwen-model -n YOUR-NAMESPACE

# Check HTTPRoute
kubectl get httproute qwen-route -n YOUR-NAMESPACE

# Check EPP logs
kubectl logs -f deployment/qwen-epp -n YOUR-NAMESPACE
```