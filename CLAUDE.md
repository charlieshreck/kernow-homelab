# Kernow Homelab

Parent repository for all Kernow homelab cluster configurations.

## Repository Structure

```
/home/                              # kernow-homelab (this repo)
├── .mcp.json                       # MCP server configuration (6 domain MCPs)
├── CLAUDE.md                       # This file
├── agentic_lab/                    # Submodule: AI platform cluster (10.20.0.0/24)
├── prod_homelab/                   # Submodule: Production cluster (10.10.0.0/24)
├── monit_homelab/                  # Submodule: Monitoring cluster (10.30.0.0/24)
└── mcp-servers/                    # Submodule: Consolidated MCP server codebase
    ├── domains/                    # Domain-based MCP servers
    ├── shared/                     # Common utilities (kernow_mcp_common)
    └── kubernetes/                 # K8s manifests for domain MCPs
```

## MCP Servers

6 consolidated domain MCP servers in the agentic cluster (ai-platform namespace):

| Domain MCP | Endpoint | Components |
|------------|----------|------------|
| observability | observability-mcp.agentic.kernow.io | Keep, Coroot, VictoriaMetrics, AlertManager, Grafana, Gatus |
| infrastructure | infrastructure-mcp.agentic.kernow.io | Kubernetes, Proxmox, TrueNAS, Cloudflare, OPNsense, Caddy, Infisical |
| knowledge | knowledge-mcp.agentic.kernow.io | Qdrant, Neo4j, Outline, Vikunja |
| home | home-mcp.agentic.kernow.io | Home Assistant, Tasmota, UniFi, AdGuard, Homepage |
| media | media-mcp.agentic.kernow.io | Plex, Sonarr, Radarr, Prowlarr, Overseerr, Tautulli, Transmission, SABnzbd |
| external | external-mcp.agentic.kernow.io | SearXNG web search, GitHub, Reddit, Wikipedia, Playwright browser |

- **Docker images**: `ghcr.io/charlieshreck/mcp-<domain>:latest`
- **Source code**: `mcp-servers/domains/<domain>/`
- **K8s manifests**: `mcp-servers/kubernetes/domains/<domain>.yaml`
- **Development**: See `mcp-servers/README.md`

## Per-Cluster Documentation

Each submodule has its own CLAUDE.md:
- `agentic_lab/CLAUDE.md` - AI platform architecture, MCP development
- `prod_homelab/CLAUDE.md` - Production apps, ArgoCD patterns
- `monit_homelab/CLAUDE.md` - Monitoring stack configuration

## Submodule Workflow

```bash
# Pull latest
git submodule update --init --recursive

# Commit changes (preferred - handles both submodule and parent)
/home/scripts/git-commit-submodule.sh <submodule> "<commit message>"

# Clone fresh
git clone --recurse-submodules git@github.com:charlieshreck/kernow-homelab.git
```

See `~/.claude/CLAUDE.md` for full git workflow rules, operational patterns, and behavioral instructions.
