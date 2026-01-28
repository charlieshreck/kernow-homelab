# Collaborate: Claude-Gemini Workflow

Orchestrate a collaborative planning workflow between Claude and Gemini.

## Arguments

- `$ARGUMENTS` - Either "start <project_name>" to begin new workflow, or "resume" to continue existing

## Overview

This skill manages a structured collaboration between Claude (you) and Gemini for project planning. **Each project is a single living document** in Outline that grows through the workflow phases.

### Planning Phases

1. **Draft** - You create initial project proposal (inline in master doc)
2. **Expand** - Both agents flesh out details (Gemini first, then you)
3. **Review** - Cross-review each other's expansions
4. **Synthesize** - You merge both perspectives into final plan
5. **Approve** - Gemini gives final verdict

Conflicts trigger an interview with the human to resolve disagreements.

### Completion Phases (Post-Approval)

6. **Implementing** - Track implementation progress with notes
7. **Validating** - Tripartite validation (automated, behavioral, peer review)
8. **Lessons Learned** - Structured interview to capture insights
9. **Graduating** - Archive knowledge to Qdrant with domain-aware decay
10. **Completed** - Final state, workflow archived

Or: **Abandoned** - Project cancelled at any phase

### Document Model

- **Master doc** = Overview with status, key decisions table, sub-doc links, deliverables
- **Sub-docs** = Deep analysis as child documents (real code, architecture, evidence)
- **No empty containers** - every document has substantial content
- **Summaries in master** - each sub-doc is summarized with a verdict line in the master
- **Decisions inline** - key decisions table updated in the master doc as they're made
- **Projects can split** - if scope grows, break into sub-projects with `split` command

## Workflow Commands

### Start New Workflow
```bash
/home/scripts/collaborate.py start "<project_name>" --collection "<collection_id>" --goal "Brief goal statement"
```

### List Available Collections
```bash
/home/scripts/collaborate.py collections
```

### Check Status
```bash
/home/scripts/collaborate.py status --checkpoint /home/collaborate/state/<workflow_id>.json
```

### Run Next Phase
```bash
/home/scripts/collaborate.py run /home/collaborate/state/<workflow_id>.json
```

### Update After Your Work
```bash
/home/scripts/collaborate.py update /home/collaborate/state/<workflow_id>.json \
  --draft "Your draft content in markdown" \
  --claude-expansion "Your expansion content in markdown" \
  --claude-review "Your review content in markdown" \
  --synthesis "Your synthesis content in markdown" \
  --claude-position '{"recommendation": "...", "confidence": 0.8, "rationale": "..."}'
```

All content is **appended as sections** to the single master document.

### Resolve Conflict
```bash
/home/scripts/collaborate.py resolve /home/collaborate/state/<workflow_id>.json \
  --decision <claude|gemini|hybrid> \
  --rationale "Explanation for decision"
```

## Your Responsibilities

When the orchestrator returns `action_required`, you must:

### `claude_draft`
1. Write your initial draft as markdown text
2. Update workflow: `--draft "Your markdown content here"`
3. The content replaces the placeholder in the master document

### `claude_expand`
1. Read Gemini's expansion sub-doc (doc ID in response or master doc's sub-docs table)
2. Write a **deep analysis** doc - real code, architecture diagrams, evidence, not just bullet points
3. Follow the depth of existing Agentic Knowledge System docs as reference
4. Update workflow: `--claude-expansion "Your full expansion markdown"`
5. This creates a child document under the master with your content

### `claude_review`
1. Read Gemini's review sub-doc (or full expansion)
2. Write a **substantive review** - evidence-based arguments, not just "agreed"
3. Include code examples, alternative approaches, or research backing your points
4. Update workflow: `--claude-review "Your full review markdown"`
5. Also set position: `--claude-position '{"recommendation": "...", "confidence": 0.85, "rationale": "..."}'`

### `claude_synthesize`
1. Read the master doc (all sections are there)
2. Write final synthesis incorporating all feedback
3. Update workflow: `--synthesis "Your synthesis content"`

### `claude_revise`
1. Read Gemini's feedback on synthesis (in the master doc)
2. Write revised synthesis
3. Run approval phase again

## Handling Conflicts

When status shows `"status": "conflict"`, an interview structure is provided.
Present this to the human using AskUserQuestion tool:
1. Show both positions clearly
2. Ask the targeted questions
3. Capture their decision and rationale
4. Apply resolution: `collaborate.py resolve ... --decision <their_choice> --rationale "..."`

## Kickoff Interview

Before starting, determine where the project goes:
1. List collections: `collaborate.py collections`
2. Ask the user which collection this belongs to (or if a new one is needed)
3. Ask for a one-sentence goal statement
4. Start the workflow with the collection ID and goal

## Example Flow

```bash
# 1. List collections to find where this project belongs
/home/scripts/collaborate.py collections

# 2. Start workflow (creates single master doc in the collection)
/home/scripts/collaborate.py start "API Redesign" \
  --collection "702740e6-c209-4dd7-ac90-a370fe480f8a" \
  --goal "Redesign the internal API to support real-time updates"
# Returns: action_required: claude_draft

# 3. You write draft content directly
/home/scripts/collaborate.py update .../workflow.json \
  --draft "## Problem\nThe current API doesn't support real-time..."
# Automatically runs next phase, invokes Gemini expand

# 4. Gemini expands (appended to master doc)
# Returns action_required: claude_expand

# 5. You write your expansion
/home/scripts/collaborate.py update .../workflow.json \
  --claude-expansion "## Additional Considerations\n..."

# 6. Continue until approved or conflict...
```

## MCP Tools You'll Use

- `mcp__knowledge__export_document` - Read the master doc content
- `mcp__knowledge__list_collections` - Find collections
- `mcp__knowledge__create_collection` - Create new collection if needed
- `mcp__knowledge__get_collection_structure` - See what's in a collection

## State Location

- Checkpoints: `/home/collaborate/state/<workflow_id>.json`
- Logs: `/home/collaborate/logs/<workflow_id>.log`

## Collections

| Collection | Purpose |
|------------|---------|
| Collaboration Workflow | The workflow tooling itself |
| Agentic Knowledge System | Knowledge architecture projects |
| Infrastructure Plans | Infra change plans |
| Kernow MCPs | MCP server docs |

Use `collaborate.py collections` for the full list with IDs.

## Splitting Projects

If a project gets too large, split it:
```bash
/home/scripts/collaborate.py split /home/collaborate/state/<workflow_id>.json \
  --name "Validation Engine" \
  --scope "Design and implement the tripartite validation strategy"
```
This creates a new child workflow linked from the parent master doc.

## Depth Expectations

Follow the standard set by the Agentic Knowledge System collection:
- **Expansions** should contain: architecture diagrams, code examples, API contracts, performance budgets
- **Reviews** should contain: evidence-based critiques, alternative approaches, research sources
- **Don't** write surface-level summaries - each sub-doc should stand alone as a useful reference

## Completion Flow Commands

After a plan is approved, use these commands to track implementation and graduate knowledge:

### Start Implementation
```bash
/home/scripts/collaborate.py implement /home/collaborate/state/<workflow_id>.json
```

### Add Implementation Notes
```bash
/home/scripts/collaborate.py note /home/collaborate/state/<workflow_id>.json \
  --text "Implemented the Phase enum changes and tested state transitions"
```

### Start Validation
```bash
/home/scripts/collaborate.py validate /home/collaborate/state/<workflow_id>.json
```

### Report Validation Results
```bash
# Automated tests
/home/scripts/collaborate.py validate-automated .../workflow.json --passed

# Behavioral verification
/home/scripts/collaborate.py validate-behavioral .../workflow.json --passed --reason "Side effects verified"

# Gemini peer review
/home/scripts/collaborate.py validate-review .../workflow.json
```

### Handle Validation Failure
```bash
/home/scripts/collaborate.py validation-resolve .../workflow.json \
  --action fix_and_retry \
  --reason "Test assertion was wrong, implementation is correct"
```

Actions: `fix_and_retry`, `revise_plan`, `override`, `abandon`

### Lessons Learned Interview
```bash
/home/scripts/collaborate.py lessons-learned .../workflow.json --responses '{
  "outcome": "success",
  "challenges": [
    {
      "symptom": "Gemini timeout during approval",
      "root_cause": "Default 300s timeout too short for complex reviews",
      "solution": "Extended timeout to 540s",
      "key_insight": "Complex multi-doc reviews need longer timeouts"
    }
  ],
  "patterns": [
    {
      "title": "Hybrid Document Model",
      "context": "When managing collaborative documents",
      "implementation": "Master doc for overview, sub-docs for depth",
      "domain": "patterns"
    }
  ],
  "workflow_feedback": "The interview mechanism worked well for conflict resolution"
}'
```

### Graduate to Knowledge Base
```bash
# Preview what will be graduated
/home/scripts/collaborate.py graduate .../workflow.json --dry-run

# Execute graduation
/home/scripts/collaborate.py graduate .../workflow.json
```

### Abandon Project
```bash
/home/scripts/collaborate.py abandon .../workflow.json \
  --reason "Requirements changed, project no longer needed"
```

## Knowledge Decay Rates

Graduated knowledge uses domain-aware decay rates:
| Domain | Quarterly Decay | Example |
|--------|-----------------|---------|
| Architecture | 2% | System design decisions |
| Patterns | 3% | Reusable implementation patterns |
| Decisions | 5% | Project-specific choices |
| Lessons | 8% | Troubleshooting insights |
| Implementation | 15% | Specific code/API details |

## Notes

- **Master doc** = overview linking to deep sub-docs
- **Sub-docs** = child documents with real analysis (code, diagrams, evidence)
- Always read the master doc AND relevant sub-docs before expanding or reviewing
- Set your position with confidence score before reviews
- Workflow survives session restarts - use `resume` to continue
- When Gemini expands, it creates a sub-doc and the summary appears in the master
- **Completion flow** is optional - projects can stay in APPROVED if not being implemented
