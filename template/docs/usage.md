# **Using the Llama Stack Agentic AI Workflow Template**

This AI Software Template allows you to customize and deploy a complete agentic AI workflow system. Follow this guide to configure and use the template effectively.

## **Template Parameters**

### **Application Information**

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| **Name** | Unique name for your application | - | Yes |
| **Owner** | Owner of the component in RHDH | `user:guest` | Yes |
| **ArgoCD Namespace** | Target namespace for ArgoCD | `openshift-gitops` | Yes |
| **ArgoCD Instance** | ArgoCD instance name | `default` | Yes |
| **ArgoCD Project** | ArgoCD project name | `default` | Yes |
| **Include ArgoCD App Label** | Include a user provided ArgoCD Application Label | `true` | No |
| **ArgoCD Application Label** | Label RHDH uses to identify ArgoCD Applications | `rolling-demo` | Conditional |

### **Llama Stack Configuration**

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| **vLLM Server URL** | URL of the vLLM inference server endpoint | - | Yes |
| **Safety Model** | Model for content safety guardrails | `ollama/llama-guard3:8b` | No |

### **Application Configuration**

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| **Inference Model** | Model ID for classification and inference | - | Yes |
| **MCP Tool Model** | Model for MCP Kubernetes tool calls (must support function calling) | - | Yes |
| **GitHub Repository URL** | Target repository for the GitHub MCP Tool for issue creation and commenting | - | Yes |

!!! tip "Model Selection"
    
    For optimal performance, models that grade well for tool calling are recommended:
    
    - `redhataiqwen3-8b-fp8-dynamic` via vLLM
    - `gpt-4o` or `gpt-4o-mini` via OpenAI

### **Repository Information**

Two repositories will be created based on your settings: `{name}` for source code and `{name}-gitops` for Kubernetes manifests synced by ArgoCD.

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| **Host Type** | GitHub or GitLab | `GitHub` | Yes |
| **Repository Server** | Git server URL | `github.com` or `gitlab.com` | Yes |
| **Repository Owner** | Organization or user | - | Yes |
| **Repository Name** | Creates two repos: source code (for building/deploying) and gitops (Kubernetes manifests) | - | Yes |
| **Branch** | Default branch name | `main` | Yes |

### **Namespace and Secrets Configuration**

All secrets must exist in the specified namespace before deployment.

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| **Deployment Namespace** | Kubernetes namespace where the application and secrets will be deployed | `rhdh-app` | Yes |
| **Llama Stack Secrets Name** | Secret containing `VLLM_API_KEY` and `OPENAI_API_KEY` | `llama-stack-secrets` | Yes |
| **Platform Credentials Secret Name** | Secret containing `GITHUB_TOKEN`, `GITLAB_TOKEN`, `WEBHOOK_SECRET`, `QUAY_DOCKERCONFIGJSON` | `platform-credentials` | Yes |
| **Secrets Acknowledgment** | Checkbox confirming secrets exist or will be created | - | Yes |

### **Deployment Information**

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| **Image Registry** | Container registry host | `quay.io` | Yes |
| **Image Organization** | Registry organization | - | Yes |
| **Image Name** | Name for container image | - | Yes |

## **Required Secrets**

Before deploying, you need to create the following secrets in your target namespace. See the examples below for the required structure.

### **Llama Stack Secrets**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: llama-stack-secrets  # Must match "Llama Stack Secrets Name" parameter
type: Opaque
stringData:
  # API key for vLLM server - required for inference
  VLLM_API_KEY: "<your-vllm-api-key>"
  # OpenAI API key - required for embeddings and inference
  OPENAI_API_KEY: "<your-openai-api-key>"
```

### **Platform Credentials**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: platform-credentials  # Must match "Platform Credentials Secret Name" parameter
type: Opaque
stringData:
  # GitHub PAT with 'repo' scope
  # Used by: Application (GitHub MCP Tool) + CI/CD (Tekton GitOps)
  GITHUB_TOKEN: "<your-github-personal-access-token>"
  # GitLab PAT (only if using GitLab instead of GitHub)
  GITLAB_TOKEN: "<your-gitlab-pat>"
  # Pipelines as Code webhook secret (GitHub/GitLab)
  WEBHOOK_SECRET: "<your-webhook-secret>"
  # Docker config JSON for pushing images to registry
  # Format: '{"auths":{"quay.io":{"auth":"<base64-user:token>","email":"<email>"}}}'
  QUAY_DOCKERCONFIGJSON: '{"auths":{"quay.io":{"auth":"","email":""}}}'
```

!!! warning "Secret Management"
    
    For production deployments, use **Sealed Secrets** or **External Secrets Operator** 
    instead of plain Kubernetes secrets.

## **MCP Server Permissions**

The Kubernetes MCP Server requires read access to cluster resources. The template creates:

- **ServiceAccount** - `<app-name>-mcp-sa`
- **ClusterRole** - `<app-name>-mcp-reader` with read-only permissions
- **ClusterRoleBinding** - Binds the role to the service account

!!! note "Cluster-Wide Access"
    
    The MCP Server has cluster-wide read access to enable cross-namespace 
    troubleshooting. Modify the ClusterRole if you need to restrict scope.

## **Post-Deployment Steps**

1. **Verify Deployments** - Check that all four pods are running:
   ```bash
   kubectl get pods -l app.kubernetes.io/part-of=<app-name>
   ```

2. **Check Llama Stack Health**:
   ```bash
   kubectl port-forward svc/<app-name>-llama-stack 8321:8321
   curl http://localhost:8321/v1/health
   ```

3. **Access the UI** - Navigate to the OpenShift Route:
   ```bash
   kubectl get route <app-name> -o jsonpath='{.spec.host}'
   ```

4. **Monitor Ingestion** - The application will automatically ingest documents on first startup

## **Git Repository Options**

!!! tip "Git Repositories"
    
    You can choose between **GitHub** and **GitLab** as your desired Source Code 
    Management (SCM) platform, and the template fields will update accordingly!

## **Troubleshooting**

### Llama Stack Not Starting

Check the ConfigMap and Secret are correctly configured:
```bash
kubectl get configmap <app-name>-llama-stack-env -o yaml
kubectl get secret llama-stack-secrets
```

### MCP Server Permission Denied

Verify the ClusterRoleBinding exists:
```bash
kubectl get clusterrolebinding <app-name>-mcp-reader-binding
```

### Ollama Model Not Loaded

Check the Ollama pod logs and verify the safety model was pulled:
```bash
kubectl logs deployment/<app-name>-ollama -c pull-safety-model
kubectl logs deployment/<app-name>-ollama -c ollama
```

### Vector Store Errors

Check the Ollama PVC is bound (Llama Stack uses ephemeral storage):
```bash
kubectl get pvc <app-name>-ollama-data
```
