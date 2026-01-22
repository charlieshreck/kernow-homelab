---
name: mcp-builder
description: |
  Guide for creating high-quality MCP (Model Context Protocol) servers that enable LLMs to interact with external services through well-designed tools. Use when building MCP servers to integrate external APIs or services, whether in Python (FastMCP) or Node/TypeScript (MCP SDK). Specifically for Kernow homelab: consolidating multiple MCPs into domain MCPs (e.g., plex + arr-suite → media-mcp), creating new domain MCPs from scratch, or following the pattern established by observability-mcp, external-mcp, media-mcp. Triggers: "create MCP", "new domain MCP", "consolidate MCPs", "build MCP server", "add MCP domain".
---

# MCP Domain Builder Skill

Guide for creating consolidated domain MCP servers that combine multiple individual MCPs into a single pre-built Docker image.

## Architecture Overview

```
mcp-servers/                          # Monorepo for all domain MCPs
├── domains/<domain>/                 # Each domain has its own directory
│   ├── src/<domain>_mcp/            # Python package (underscores!)
│   │   ├── __init__.py
│   │   ├── server.py                # FastMCP server with health endpoints
│   │   └── tools/                   # One file per consolidated service
│   │       ├── __init__.py
│   │       └── <service>.py         # e.g., plex.py, sonarr.py
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── README.md
├── shared/kernow_mcp_common/        # Shared utilities
└── kubernetes/domains/<domain>.yaml  # K8s manifest
```

## Naming Conventions

| Item | Pattern | Example |
|------|---------|---------|
| Domain directory | `domains/<domain>/` | `domains/media/` |
| Python package | `<domain>_mcp` | `media_mcp` |
| Docker image | `ghcr.io/charlieshreck/mcp-<domain>` | `mcp-media` |
| K8s resources | `<domain>-mcp` | `media-mcp` |
| NodePort | 311XX (allocate next available) | 31123 |
| Ingress | `<domain>-mcp.agentic.kernow.io` | `media-mcp.agentic.kernow.io` |
| Tool prefix | `<service>_*` | `plex_*`, `sonarr_*` |

**CRITICAL**: Docker image is `mcp-<domain>`, NOT `<domain>-mcp`!

## Step-by-Step Workflow

### 1. Create Directory Structure

```bash
mkdir -p /home/mcp-servers/domains/<domain>/src/<domain>_mcp/tools
```

### 2. Create Package Files

**`src/<domain>_mcp/__init__.py`**:
```python
"""<Domain> MCP - <description>."""
__version__ = "1.0.0"
```

**`src/<domain>_mcp/tools/__init__.py`**:
```python
"""<Domain> MCP tools."""
```

### 3. Create Server (server.py)

Key components:
- FastMCP with `stateless_http=True`
- Starlette wrapper with `/health` and `/ready` endpoints
- Import and register tools from each service module
- Health check that verifies component connectivity

```python
import os
import logging
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn

from <domain>_mcp.tools import service1, service2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="<domain>-mcp",
    instructions="""<Domain> MCP for <description>.
    Tools: <list key tools>""",
    stateless_http=True
)

# Register tools from each service
service1.register_tools(mcp)
service2.register_tools(mcp)

async def health_check(request):
    components = {}
    # Check each service
    try:
        result = await service1.health_function()
        components["service1"] = "healthy" if "error" not in result else "unhealthy"
    except:
        components["service1"] = "unhealthy"

    healthy = sum(1 for v in components.values() if v == "healthy")
    status = "healthy" if healthy >= len(components) // 2 else "unhealthy"

    return JSONResponse({
        "status": status,
        "service": "<domain>-mcp",
        "version": "1.0.0",
        "components": components,
        "healthy_count": healthy,
        "total_count": len(components)
    })

async def ready_check(request):
    return JSONResponse({"ready": True})

app = Starlette(routes=[
    Route("/health", health_check),
    Route("/ready", ready_check),
    Mount("/", app=mcp.get_asgi_app()),
])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
```

### 4. Create Tool Modules

Each service gets its own file in `tools/`. Pattern:

```python
"""<Service> tools."""
import os
import logging
import httpx
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Configuration from environment
SERVICE_URL = os.environ.get("SERVICE_URL", "https://service.kernow.io")
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "")

async def service_request(endpoint: str) -> dict:
    """Make request to service API."""
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        headers = {"X-Api-Key": SERVICE_API_KEY}
        response = await client.get(f"{SERVICE_URL}/api/v3/{endpoint}", headers=headers)
        response.raise_for_status()
        return response.json()

# Health check function (exported for server.py)
async def get_status() -> dict:
    """Get service status for health checks."""
    try:
        return await service_request("system/status")
    except Exception as e:
        return {"error": str(e)}

def register_tools(mcp: FastMCP):
    """Register service tools with the MCP server."""

    @mcp.tool()
    async def service_get_status() -> dict:
        """Get service status."""
        return await get_status()

    @mcp.tool()
    async def service_list_items() -> list:
        """List items from service."""
        try:
            return await service_request("items")
        except Exception as e:
            return [{"error": str(e)}]
```

### 5. Create pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "<domain>-mcp"
version = "1.0.0"
description = "Consolidated <Domain> MCP - <services list>"
requires-python = ">=3.11"
dependencies = [
    "kernow-mcp-common",
    "fastmcp>=2.7.0",
    "httpx>=0.28.0",
    "uvicorn>=0.34.0",
    "starlette>=0.40.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]

[tool.setuptools.packages.find]
where = ["src"]
```

### 6. Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies (add as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared utilities
COPY shared/kernow_mcp_common /app/shared/kernow_mcp_common
COPY shared/pyproject.toml /app/shared/
COPY shared/README.md /app/shared/

# Copy domain code
COPY domains/<domain>/pyproject.toml .
COPY domains/<domain>/README.md .
COPY domains/<domain>/src ./src

# Install
RUN pip install --no-cache-dir /app/shared
RUN pip install --no-cache-dir .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Non-root user
RUN useradd -m -u 1000 mcp
USER mcp

EXPOSE 8000

ENV PORT=8000
ENV HOST=0.0.0.0

CMD ["python", "-m", "<domain>_mcp.server"]
```

### 7. Create Kubernetes Manifest

File: `kubernetes/domains/<domain>.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: <domain>-mcp
  namespace: ai-platform
  labels:
    app: <domain>-mcp
    component: mcp
    domain: <domain>
spec:
  replicas: 1
  selector:
    matchLabels:
      app: <domain>-mcp
  template:
    metadata:
      labels:
        app: <domain>-mcp
        component: mcp
        domain: <domain>
    spec:
      containers:
        - name: <domain>-mcp
          image: ghcr.io/charlieshreck/mcp-<domain>:latest  # NOTE: mcp-<domain>!
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
          env:
            - name: PORT
              value: "8000"
            # Add service-specific env vars and secrets
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "1000m"
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: <domain>-mcp
  namespace: ai-platform
spec:
  type: NodePort
  selector:
    app: <domain>-mcp
  ports:
    - port: 8000
      targetPort: 8000
      nodePort: 311XX  # Allocate next available
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: <domain>-mcp
  namespace: ai-platform
  annotations:
    traefik.ingress.kubernetes.io/router.entrypoints: websecure
    traefik.ingress.kubernetes.io/router.tls: "true"
spec:
  ingressClassName: traefik
  rules:
    - host: <domain>-mcp.agentic.kernow.io
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: <domain>-mcp
                port:
                  number: 8000
  tls:
    - hosts:
        - <domain>-mcp.agentic.kernow.io
      secretName: wildcard-agentic-kernow-io
```

### 8. Update CI Workflow

Edit `.github/workflows/build-domain-mcps.yaml`:

```yaml
strategy:
  matrix:
    domain:
      - observability
      - external
      - media
      - <new-domain>  # Add here
```

### 9. Mark Old MCPs as Deprecated

Add comment header to old MCP yaml files:

```yaml
# =============================================================================
# DEPRECATED: This MCP is now consolidated into <domain>-mcp
# New endpoint: <domain>-mcp.agentic.kernow.io (or 10.20.0.40:<nodeport>)
# Source: https://github.com/charlieshreck/mcp-servers/tree/main/domains/<domain>
# This file is kept for reference only - do not deploy new instances
# =============================================================================
```

### 10. Commit and Deploy

```bash
# Commit mcp-servers
git -C /home/mcp-servers add domains/<domain>/ kubernetes/domains/<domain>.yaml .github/workflows/
git -C /home/mcp-servers commit -m "feat: add <domain>-mcp domain consolidating <services>"
git -C /home/mcp-servers push origin main

# Commit agentic_lab (deprecation notices)
git -C /home/agentic_lab add kubernetes/applications/mcp-servers/
git -C /home/agentic_lab commit -m "chore: mark <services> as deprecated for <domain>-mcp"
git -C /home/agentic_lab push origin main

# Update parent submodules
git add mcp-servers agentic_lab
git commit -m "chore: update submodules for <domain>-mcp"
git push origin main

# Wait for CI, then deploy
KUBECONFIG=/home/agentic_lab/infrastructure/terraform/talos-cluster/generated/kubeconfig \
  kubectl apply -f /home/mcp-servers/kubernetes/domains/<domain>.yaml

# Verify
curl http://10.20.0.40:<nodeport>/health
```

## Common Pitfalls

1. **Image name**: CI builds `mcp-<domain>`, manifest must match
2. **Package name**: Use underscores (`media_mcp`), not hyphens
3. **Tool registration**: Each service module needs `register_tools(mcp)` function
4. **Health checks**: Export a simple status function from each tool module
5. **Secrets**: Reference existing K8s secrets, don't create new ones unless needed
6. **verify=False**: Use for internal HTTPS services with self-signed certs

## Existing Domains

| Domain | Port | Consolidates | Status |
|--------|------|--------------|--------|
| observability | 31120 | keep, coroot, monitoring, gatus | Deployed |
| external | 31121 | web-search, github, reddit, wikipedia, browser-automation | Deployed |
| media | 31123 | plex, arr-suite | Deployed |
| home | TBD | home-assistant, tasmota, unifi, adguard, homepage | Planned |
| knowledge | TBD | knowledge, neo4j, outline, vikunja | Planned |
| infrastructure | TBD | infrastructure, proxmox, truenas, cloudflare, opnsense, infisical | Planned |

## Secrets Reference

Existing K8s secrets in ai-platform namespace:
- `mcp-plex`, `mcp-sonarr`, `mcp-radarr`, `mcp-prowlarr`, `mcp-overseerr`, `mcp-tautulli`, `mcp-transmission`, `mcp-sabnzbd`
- `mcp-home-assistant`, `mcp-tasmota`, `mcp-unifi`, `mcp-adguard`
- `mcp-github`, `mcp-cloudflare`, `mcp-truenas-hdd`, `mcp-truenas-media`
- `mcp-proxmox`, `mcp-opnsense`, `mcp-infisical`
