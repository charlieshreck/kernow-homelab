# Kernow Homelab

Parent repository for all Kernow homelab cluster configurations.

## Repository Structure

This repository uses git submodules to organize the three cluster configurations:

```
/home/                              # kernow-homelab (this repo)
├── .mcp.json                       # MCP server configuration (24 servers)
├── CLAUDE.md                       # This file
├── agentic_lab/                    # Submodule: AI platform cluster (10.20.0.0/24)
├── prod_homelab/                   # Submodule: Production cluster (10.10.0.0/24)
├── monit_homelab/                  # Submodule: Monitoring cluster (10.30.0.0/24)
└── mcp-servers/                    # Submodule: Consolidated MCP server codebase
    ├── domains/                    # Domain-based MCP servers
    │   └── observability/          # Keep + Coroot + monitoring + Gatus
    ├── shared/                     # Common utilities (kernow_mcp_common)
    └── kubernetes/                 # K8s manifests for domain MCPs
```

## Cluster Overview

| Cluster | Submodule | Network | Purpose |
|---------|-----------|---------|---------|
| **prod** | `prod_homelab/` | 10.10.0.0/24 | Production apps, media, ArgoCD |
| **agentic** | `agentic_lab/` | 10.20.0.0/24 | AI platform, MCP servers |
| **monit** | `monit_homelab/` | 10.30.0.0/24 | Monitoring stack |

## Submodule Workflow

### Pull Latest
```bash
# Update parent repo and submodule pointers
git pull

# Update all submodules to their tracked commits
git submodule update --init --recursive

# Update submodules to latest remote HEAD (optional)
git submodule update --remote
```

### Work in a Submodule (Helper Script - PREFERRED)
```bash
# Single command: stages all, commits, pushes submodule, updates + pushes parent
/home/scripts/git-commit-submodule.sh agentic_lab "feat: add new feature"
/home/scripts/git-commit-submodule.sh monit_homelab "fix: update config"
/home/scripts/git-commit-submodule.sh prod_homelab "docs: update readme"
```

### Work in a Submodule (Manual with `git -C`)

**CRITICAL**: The working directory is `/home` (parent repo). `cd` does NOT persist
between shell commands. Always use `git -C <path>` to target the correct repo.

```bash
# Step 1: Stage and commit INSIDE the submodule
git -C /home/agentic_lab add -A
git -C /home/agentic_lab commit -m "feat: my change"
git -C /home/agentic_lab push origin main

# Step 2: Update parent repo's submodule pointer
git -C /home add agentic_lab
git -C /home commit -m "chore: update agentic_lab submodule"
git -C /home push origin main
```

**IMPORTANT for Claude**:
- ALWAYS use `git -C <path>` - never rely on `cd` for git operations
- ALWAYS use the helper script when possible (handles both steps automatically)
- NEVER run bare `git add -A && git commit` - this operates on `/home` (parent), not the submodule
- The two-step process is mandatory: commit in submodule first, then update parent reference

### Clone Fresh
```bash
git clone --recurse-submodules git@github.com:charlieshreck/kernow-homelab.git
```

## MCP Servers

6 consolidated domain MCP servers run in the agentic cluster (ai-platform namespace), accessible via DNS ingress and configured in `.mcp.json`:

| Domain MCP | Endpoint | Components |
|------------|----------|------------|
| observability | observability-mcp.agentic.kernow.io | Keep, Coroot, VictoriaMetrics, AlertManager, Grafana, Gatus |
| infrastructure | infrastructure-mcp.agentic.kernow.io | Kubernetes, Proxmox, TrueNAS, Cloudflare, OPNsense, Caddy, Infisical |
| knowledge | knowledge-mcp.agentic.kernow.io | Qdrant, Neo4j, Outline, Vikunja |
| home | home-mcp.agentic.kernow.io | Home Assistant, Tasmota, UniFi, AdGuard, Homepage |
| media | media-mcp.agentic.kernow.io | Plex, Sonarr, Radarr, Prowlarr, Overseerr, Tautulli, Transmission, SABnzbd |
| external | external-mcp.agentic.kernow.io | SearXNG web search, GitHub, Reddit, Wikipedia, Playwright browser |

### MCP Access

All domain MCPs use HTTP ingress endpoints:
```
http://<domain>-mcp.agentic.kernow.io/mcp    # MCP protocol endpoint
http://<domain>-mcp.agentic.kernow.io/health # Health check
```

### Architecture

- **Pre-built Docker images**: `ghcr.io/charlieshreck/mcp-<domain>:latest`
- **Source code**: `mcp-servers/domains/<domain>/`
- **Kubernetes manifests**: `mcp-servers/kubernetes/domains/<domain>.yaml`
- **Shared utilities**: `mcp-servers/shared/kernow_mcp_common/`

### Development
See `mcp-servers/README.md` for:
- Domain structure and tool organization
- Dockerfile build process
- Local development workflow

## Kubeconfig Paths

```bash
# Production cluster
export KUBECONFIG=/home/prod_homelab/infrastructure/terraform/generated/kubeconfig

# Agentic cluster
export KUBECONFIG=/home/agentic_lab/infrastructure/terraform/talos-cluster/generated/kubeconfig

# Monitoring cluster
export KUBECONFIG=/home/monit_homelab/kubeconfig
```

## Per-Cluster Documentation

Each submodule has its own CLAUDE.md with cluster-specific details:
- `agentic_lab/CLAUDE.md` - AI platform architecture, MCP development
- `prod_homelab/CLAUDE.md` - Production apps, ArgoCD patterns
- `monit_homelab/CLAUDE.md` - Monitoring stack configuration

See also: `~/.claude/CLAUDE.md` for global rules (GitOps, Infisical, forbidden actions).
