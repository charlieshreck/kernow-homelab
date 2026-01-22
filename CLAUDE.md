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

### Work in a Submodule (Manual)
```bash
cd agentic_lab/  # or prod_homelab/ or monit_homelab/
# ... make changes ...
git add . && git commit -m "message" && git push

# Return to parent and update reference
cd ..
git add agentic_lab
git commit -m "Update agentic_lab submodule"
git push
```

### Work in a Submodule (Helper Script - Preferred)
```bash
# Single command: commits submodule, pushes, updates parent
/home/scripts/git-commit-submodule.sh agentic_lab "feat: add new feature"
/home/scripts/git-commit-submodule.sh monit_homelab "fix: update config"
/home/scripts/git-commit-submodule.sh prod_homelab "docs: update readme"
```

**IMPORTANT for Claude**: Always use the helper script to avoid:
- Committing in wrong directory (parent vs submodule)
- Forgetting to update parent submodule reference
- Push failures from incorrect working directory

### Clone Fresh
```bash
git clone --recurse-submodules git@github.com:charlieshreck/kernow-homelab.git
```

## MCP Servers

All 24 MCP servers run in the agentic cluster (ai-platform namespace) and are configured in `.mcp.json`:

| Server | Port | Purpose |
|--------|------|---------|
| knowledge | 31084 | Qdrant knowledge base, runbooks, entities |
| infrastructure | 31083 | kubectl, talosctl, cluster operations |
| coroot | 31081 | Observability, metrics, anomalies |
| proxmox | 31082 | VM management |
| opnsense | 31085 | Firewall, DHCP, DNS |
| adguard | 31086 | DNS rewrites, filtering |
| cloudflare | 31087 | DNS records, tunnels |
| unifi | 31088 | Network clients, devices |
| truenas | 31089 | Storage management |
| home-assistant | 31090 | Smart home control |
| arr-suite | 31091 | Media management |
| plex | 31096 | Media server |
| vikunja | 31097 | Task management |
| web-search | 31093 | SearXNG search |
| browser-automation | 31094 | Playwright browser automation |
| infisical | 31080 | Secrets (read-only) |
| homepage | 31092 | Dashboard widgets |
| neo4j | 31098 | Knowledge graph queries |
| tasmota | 31100 | Tasmota smart device control (26 devices) |
| monitoring | 31101 | Monitoring stack (VictoriaMetrics, AlertManager, VictoriaLogs, Grafana, Gatus) |
| keep | 31107 | Alert aggregation, deduplication, correlation |
| github | 31111 | GitHub repos, issues, PRs, code search (requires token) |
| wikipedia | 31112 | Wikipedia articles, search, knowledge retrieval |
| reddit | 31104 | Reddit browsing, subreddit search, discussions |
| outline | 31114 | Outline wiki document management, collections |

## MCP Architecture Modernization

The MCP infrastructure is transitioning from ConfigMap-embedded Python to pre-built Docker images:

### Current State (Phase 1)
- 24 individual MCPs with `stateless_http=True` for Kubernetes stability
- Code embedded in ConfigMaps (slow startup due to pip install)

### Target State (Phase 2+)
- 6 consolidated domain-based MCPs with pre-built Docker images
- Domains: observability, infrastructure, knowledge, home, media, external

### Consolidated MCP Domains

| Domain | Port | Combines |
|--------|------|----------|
| observability-mcp | 31120 | Keep + Coroot + monitoring + Gatus |
| infrastructure-mcp | TBD | K8s + Proxmox + TrueNAS + Cloudflare + OPNsense |
| knowledge-mcp | TBD | Qdrant + Outline + Neo4j |
| home-mcp | TBD | Home Assistant + Tasmota + UniFi + AdGuard |
| media-mcp | TBD | Plex + Arr-Suite + Tautulli |
| external-mcp | TBD | Web Search + GitHub + Reddit + Wikipedia |

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
