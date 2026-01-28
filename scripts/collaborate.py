#!/usr/bin/env python3
"""
Claude-Gemini Collaboration Workflow Orchestrator

Hybrid document model:
  - Master doc: Overview, status, key decisions, deliverables, sub-doc links
  - Sub-docs: Deep analysis created by each agent as child documents
  - Positions/decisions: Summarized in master doc as tables
  - No empty containers: Every doc has real content

Projects can be split into sub-projects when scope grows too large.

Planning Commands:
    collaborate.py start <project_name> --collection <id> [--goal <text>]
    collaborate.py resume [--checkpoint <path>]
    collaborate.py status [--checkpoint <path>]
    collaborate.py run <checkpoint>
    collaborate.py update <checkpoint> [--draft <text>] [--claude-position <json>]
    collaborate.py resolve <checkpoint> --decision <choice> [--rationale <text>]
    collaborate.py split <checkpoint> --name <sub_project> --scope <text>
    collaborate.py collections

Completion Flow Commands:
    collaborate.py implement <checkpoint>              - Start implementation
    collaborate.py note <checkpoint> --text "..."      - Add implementation note
    collaborate.py validate <checkpoint>               - Start validation phase
    collaborate.py validate-automated <checkpoint> --passed [--reason "..."]
    collaborate.py validate-behavioral <checkpoint> --passed [--reason "..."]
    collaborate.py validate-review <checkpoint>        - Gemini peer review
    collaborate.py validation-resolve <checkpoint> --action <fix_and_retry|revise_plan|override|abandon>
    collaborate.py lessons-learned <checkpoint> --responses <json>
    collaborate.py graduate <checkpoint> [--dry-run]
    collaborate.py abandon <checkpoint> --reason "..."

Maintenance Commands:
    collaborate.py list                                - List all workflows
    collaborate.py cleanup [--days 30] [--dry-run]     - Remove old completed/abandoned workflows
    collaborate.py archive <checkpoint>                - Archive workflow to archive/
"""

import json
import subprocess
import sys
import os
import hashlib
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, asdict, field
import argparse
import uuid

# Paths
STATE_DIR = Path("/home/collaborate/state")
LOGS_DIR = Path("/home/collaborate/logs")

# MCP Endpoints
KNOWLEDGE_MCP_URL = "http://knowledge-mcp.agentic.kernow.io"

# Domain-aware knowledge decay rates (quarterly decay percentage)
DOMAIN_DECAY_RATES = {
    "architecture": 0.02,   # ~2% per quarter - very stable
    "patterns": 0.03,       # ~3% per quarter - mostly stable
    "decisions": 0.05,      # ~5% per quarter - context-dependent
    "lessons": 0.08,        # ~8% per quarter - situation-specific
    "implementation": 0.15, # ~15% per quarter - highly volatile
}

# Lessons learned interview structure
LESSONS_LEARNED_QUESTIONS = [
    {
        "id": "outcome",
        "question": "What was the project outcome?",
        "type": "single_select",
        "options": [
            {"value": "success", "label": "Success - fully achieved goals"},
            {"value": "partial", "label": "Partial - some goals achieved"},
            {"value": "pivot", "label": "Pivot - changed direction"},
            {"value": "abandoned", "label": "Abandoned - stopped before completion"}
        ]
    },
    {
        "id": "challenges",
        "question": "What challenges were encountered?",
        "type": "structured_list",
        "schema": {
            "symptom": "What was observed?",
            "root_cause": "What was the underlying cause?",
            "solution": "How was it resolved?",
            "key_insight": "What should be remembered for next time?"
        }
    },
    {
        "id": "patterns",
        "question": "What reusable patterns emerged?",
        "type": "structured_list",
        "schema": {
            "title": "Pattern name",
            "context": "When to use this",
            "implementation": "How to implement",
            "domain": "architecture|patterns|decisions|implementation"
        }
    },
    {
        "id": "workflow_feedback",
        "question": "How did the collaboration workflow perform?",
        "type": "free_text"
    }
]


class Phase(Enum):
    INITIALIZING = "initializing"
    DRAFTING = "drafting"
    EXPANDING_GEMINI = "expanding_gemini"
    EXPANDING_CLAUDE = "expanding_claude"
    REVIEWING_GEMINI = "reviewing_gemini"
    REVIEWING_CLAUDE = "reviewing_claude"
    SYNTHESIZING = "synthesizing"
    APPROVING = "approving"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONFLICT = "conflict"
    PAUSED = "paused"
    # Completion phases
    IMPLEMENTING = "implementing"
    VALIDATING = "validating"
    VALIDATION_FAILED = "validation_failed"
    LESSONS_LEARNED = "lessons_learned"
    GRADUATING = "graduating"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ConflictType(Enum):
    EXPLICIT_REJECT = "explicit_reject"
    POSITION_MISMATCH = "position_mismatch"
    HIGH_CONFIDENCE_DISAGREEMENT = "high_confidence_disagreement"
    ITERATION_LIMIT = "iteration_limit"


@dataclass
class Position:
    agent: str
    recommendation: str
    confidence: float
    rationale: str


@dataclass
class Conflict:
    topic: str
    claude_position: Position
    gemini_position: Position
    conflict_type: ConflictType
    iteration: int
    resolved: bool = False
    resolution: Optional[str] = None
    decided_by: Optional[str] = None


@dataclass
class WorkflowState:
    workflow_id: str
    project_name: str
    collection_id: str
    phase: Phase
    iteration: int
    goal: str = ""

    # Master document - overview, status, key decisions
    master_doc_id: Optional[str] = None

    # Sub-documents - child docs with deep analysis
    # Maps role -> doc_id (e.g. "gemini-expansion-0" -> "abc123")
    sub_docs: Dict[str, str] = field(default_factory=dict)

    # Track what's been appended to the master doc (for idempotency)
    sections_appended: List[str] = field(default_factory=list)

    # Sub-projects (split from this project)
    sub_projects: List[Dict] = field(default_factory=list)

    # Conflict tracking
    conflicts: List[Dict] = field(default_factory=list)
    resolved_conflicts: List[Dict] = field(default_factory=list)

    # Positions (current state)
    claude_position: Optional[Dict] = None
    gemini_position: Optional[Dict] = None

    # Completion flow state
    implementation_notes: List[str] = field(default_factory=list)
    validation_results: Optional[Dict] = None
    lessons_learned: Optional[Dict] = None
    graduation_manifest: Optional[Dict] = None
    abandon_reason: Optional[str] = None

    # Metadata
    created_at: str = None
    updated_at: str = None
    context_loaded: bool = False

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


# --- Master Document Template ---

MASTER_DOC_TEMPLATE = """# {project_name}

**Status**: {phase}
**Workflow ID**: {workflow_id}
**Created**: {created_at}

---

## Goal

{goal}

---

## Key Decisions

| Decision | Choice | Decided By | Confidence |
|----------|--------|------------|------------|

---

## Sub-Documents

| Document | Author | Phase | Status |
|----------|--------|-------|--------|

---

## Draft

*Awaiting initial draft...*
"""


class CollaborationOrchestrator:
    """Main orchestrator for Claude-Gemini collaboration."""

    MAX_ITERATIONS = 3
    CONFIDENCE_THRESHOLD = 0.7
    CONFLICT_DELTA_THRESHOLD = 0.3

    def __init__(self, state: Optional[WorkflowState] = None):
        self.state = state
        self._ensure_dirs()

    def _ensure_dirs(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self) -> Path:
        return STATE_DIR / f"{self.state.workflow_id}.json"

    def save_checkpoint(self):
        """Save current state to checkpoint file."""
        self.state.updated_at = datetime.now().isoformat()
        with open(self._checkpoint_path(), 'w') as f:
            state_dict = asdict(self.state)
            state_dict['phase'] = self.state.phase.value
            json.dump(state_dict, f, indent=2, default=str)
        self._log(f"Checkpoint saved: {self._checkpoint_path()}")

    @classmethod
    def load_checkpoint(cls, path: Path) -> 'CollaborationOrchestrator':
        """Load state from checkpoint file."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Handle legacy state files - extract only recognized fields
        recognized_fields = {
            'workflow_id', 'project_name', 'collection_id', 'phase', 'iteration',
            'goal', 'master_doc_id', 'sub_docs', 'sections_appended', 'sub_projects',
            'conflicts', 'resolved_conflicts', 'claude_position', 'gemini_position',
            'implementation_notes', 'validation_results', 'lessons_learned',
            'graduation_manifest', 'abandon_reason', 'created_at', 'updated_at',
            'context_loaded'
        }
        filtered_data = {k: v for k, v in data.items() if k in recognized_fields}

        # Convert phase string to enum
        filtered_data['phase'] = Phase(filtered_data['phase'])

        # Ensure default values for new fields
        filtered_data.setdefault('sub_docs', {})
        filtered_data.setdefault('sections_appended', [])
        filtered_data.setdefault('sub_projects', [])
        filtered_data.setdefault('implementation_notes', [])

        state = WorkflowState(**filtered_data)
        return cls(state)

    def _log(self, message: str, level: str = "INFO"):
        """Log message to file and stdout."""
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)

        log_file = LOGS_DIR / f"{self.state.workflow_id}.log"
        with open(log_file, 'a') as f:
            f.write(log_line + "\n")

    # --- MCP Integration ---

    def _call_mcp(self, tool: str, arguments: Dict) -> Dict:
        """Call a knowledge MCP tool."""
        try:
            response = requests.post(
                f"{KNOWLEDGE_MCP_URL}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": tool,
                        "arguments": arguments
                    },
                    "id": str(uuid.uuid4())
                },
                headers={
                    "Accept": "application/json, text/event-stream"
                },
                timeout=30
            )

            text = response.text
            if text.startswith("event:"):
                for line in text.split("\n"):
                    if line.startswith("data:"):
                        json_str = line[5:].strip()
                        result = json.loads(json_str)
                        break
                else:
                    return {"error": "No data in SSE response"}
            else:
                result = response.json()

            if "result" in result:
                content = result["result"].get("content", [])
                if content and len(content) > 0:
                    text = content[0].get("text", "{}")
                    if text.startswith("{"):
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            pass
                    if "with ID:" in text:
                        doc_id = text.split("with ID:")[-1].strip()
                        return {"id": doc_id, "text": text}
                    return {"text": text}
            return result
        except Exception as e:
            self._log(f"MCP call failed: {e}", "ERROR")
            return {"error": str(e)}

    # --- Master Document Operations ---

    def create_master_doc(self) -> str:
        """Create the single master document for this project."""
        content = MASTER_DOC_TEMPLATE.format(
            project_name=self.state.project_name,
            phase=self.state.phase.value,
            workflow_id=self.state.workflow_id,
            created_at=self.state.created_at,
            goal=self.state.goal or "*No goal specified*"
        )

        result = self._call_mcp("create_document", {
            "title": self.state.project_name,
            "text": content,
            "collection_id": self.state.collection_id,
            "publish": True
        })

        doc_id = result.get("id")
        if doc_id:
            self.state.master_doc_id = doc_id
            self._log(f"Created master doc: {doc_id}")
        else:
            self._log(f"Failed to create master doc: {result}", "ERROR")

        return doc_id

    def append_section(self, section_key: str, section_content: str):
        """Append a section to the master document.

        section_key is used for idempotency - won't re-append if already added.
        """
        if section_key in self.state.sections_appended:
            self._log(f"Section already appended: {section_key}")
            return

        if not self.state.master_doc_id:
            self._log("No master doc to append to", "ERROR")
            return

        # Get current document content
        current = self._call_mcp("export_document", {
            "document_id": self.state.master_doc_id
        })
        current_text = current.get("text", current.get("data", ""))

        # Remove duplicate title headings before appending
        current_text = self._dedupe_title_headings(current_text)

        # Append the new section
        updated_text = current_text.rstrip() + "\n\n" + section_content

        self._call_mcp("update_document", {
            "document_id": self.state.master_doc_id,
            "text": updated_text
        })

        self.state.sections_appended.append(section_key)
        self._log(f"Appended section: {section_key}")

    def update_status_header(self):
        """Update the status line in the master doc header."""
        if not self.state.master_doc_id:
            return

        current = self._call_mcp("export_document", {
            "document_id": self.state.master_doc_id
        })
        current_text = current.get("text", current.get("data", ""))

        # Remove duplicate title headings (Outline adds title as H1 on export)
        current_text = self._dedupe_title_headings(current_text)

        # Replace the status line
        lines = current_text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("**Status**:"):
                lines[i] = f"**Status**: {self.state.phase.value}"
                break

        self._call_mcp("update_document", {
            "document_id": self.state.master_doc_id,
            "text": "\n".join(lines)
        })

    def _dedupe_title_headings(self, text: str) -> str:
        """Remove duplicate H1 title headings from document text."""
        lines = text.split("\n")
        if not lines:
            return text

        # Find all H1 lines that match the project name
        title_pattern = f"# {self.state.project_name}"
        first_title_idx = None
        indices_to_remove = []

        for i, line in enumerate(lines):
            if line.strip() == title_pattern:
                if first_title_idx is None:
                    first_title_idx = i
                else:
                    indices_to_remove.append(i)

        # Remove duplicates (keep the first one)
        for idx in reversed(indices_to_remove):
            lines.pop(idx)

        return "\n".join(lines)

    def create_sub_doc(self, title: str, content: str, key: str) -> Optional[str]:
        """Create a child document under the master doc with real content."""
        if not self.state.master_doc_id:
            self._log("No master doc to parent to", "ERROR")
            return None

        result = self._call_mcp("create_document", {
            "title": title,
            "text": content,
            "collection_id": self.state.collection_id,
            "parent_document_id": self.state.master_doc_id,
            "publish": True
        })

        doc_id = result.get("id")
        if doc_id:
            self.state.sub_docs[key] = doc_id
            self._log(f"Created sub-doc '{title}': {doc_id}")
        else:
            self._log(f"Failed to create sub-doc: {result}", "ERROR")

        return doc_id

    def _update_sub_docs_table(self):
        """Update the Sub-Documents table in the master doc."""
        if not self.state.master_doc_id or not self.state.sub_docs:
            return

        current = self._call_mcp("export_document", {
            "document_id": self.state.master_doc_id
        })
        current_text = current.get("text", current.get("data", ""))

        # Build new sub-docs table
        rows = ""
        for key, doc_id in self.state.sub_docs.items():
            parts = key.split("-")
            author = parts[0] if parts else "unknown"
            phase = parts[1] if len(parts) > 1 else "unknown"
            iteration = parts[2] if len(parts) > 2 else "0"
            rows += f"| [{key}](/doc/{doc_id}) | {author.title()} | {phase.title()} | Iteration {iteration} |\n"

        new_table = f"""## Sub-Documents

| Document | Author | Phase | Status |
|----------|--------|-------|--------|
{rows}"""

        # Replace old sub-docs table
        old_pattern = "## Sub-Documents\n\n| Document | Author | Phase | Status |\n|----------|--------|-------|--------|"
        if old_pattern in current_text:
            # Find end of the old table
            start = current_text.index(old_pattern)
            # Find next ## heading or --- after the table
            rest = current_text[start + len(old_pattern):]
            end_markers = ["\n---\n", "\n## "]
            end_offset = len(current_text)
            for marker in end_markers:
                idx = rest.find(marker)
                if idx >= 0:
                    end_offset = start + len(old_pattern) + idx
                    break

            current_text = current_text[:start] + new_table + current_text[end_offset:]

            self._call_mcp("update_document", {
                "document_id": self.state.master_doc_id,
                "text": current_text
            })

    def _update_decisions_table(self, decision: str, choice: str, agent: str, confidence: float):
        """Add a row to the Key Decisions table in the master doc."""
        if not self.state.master_doc_id:
            return

        current = self._call_mcp("export_document", {
            "document_id": self.state.master_doc_id
        })
        current_text = current.get("text", current.get("data", ""))

        new_row = f"| {decision} | {choice} | {agent.title()} | {confidence} |\n"

        # Find the end of the decisions table header and append the row
        table_marker = "| Decision | Choice | Decided By | Confidence |\n|----------|--------|------------|------------|"
        if table_marker in current_text:
            insert_pos = current_text.index(table_marker) + len(table_marker)
            # Check if there are existing rows
            rest = current_text[insert_pos:]
            if rest.startswith("\n|"):
                # Find end of existing rows
                lines = rest.split("\n")
                last_row = 0
                for i, line in enumerate(lines):
                    if line.startswith("|"):
                        last_row = i
                    else:
                        break
                insert_pos += sum(len(lines[j]) + 1 for j in range(last_row + 1))
            current_text = current_text[:insert_pos] + "\n" + new_row + current_text[insert_pos:]

            self._call_mcp("update_document", {
                "document_id": self.state.master_doc_id,
                "text": current_text
            })

    # --- Section Formatting ---

    def _format_position_section(self, agent: str, phase: str, position: Dict) -> str:
        """Format a position as a section to append."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""### {agent.title()} Position ({phase.replace('_', ' ').title()}, Iteration {self.state.iteration})

*{timestamp}* | **Confidence**: {position.get('confidence', 'N/A')}

**Recommendation**: {position.get('recommendation', 'N/A')}

{position.get('rationale', '')}

{f"> **Notes for other agent**: {position.get('notes_for_claude', position.get('notes_for_gemini', ''))}" if position.get('notes_for_claude') or position.get('notes_for_gemini') else ''}
"""

    def _format_review_section(self, agent: str, review: Dict) -> str:
        """Format a review as a section to append."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        verdict = review.get("verdict", "N/A")

        major_issues = review.get("major_issues", [])
        issues_md = ""
        if major_issues:
            issues_md = "\n**Major Issues:**\n"
            for issue in major_issues:
                if isinstance(issue, dict):
                    issues_md += f"- **{issue.get('issue', 'Unknown')}**: {issue.get('description', '')}\n"
                else:
                    issues_md += f"- {issue}\n"

        minor_issues = review.get("minor_issues", [])
        if minor_issues:
            issues_md += "\n**Minor Issues:**\n"
            for issue in minor_issues:
                if isinstance(issue, dict):
                    issues_md += f"- {issue.get('issue', issue.get('recommendation', str(issue)))}\n"
                else:
                    issues_md += f"- {issue}\n"

        approved = review.get("approved_aspects", [])
        approved_md = ""
        if approved:
            approved_md = "\n**Approved Aspects:**\n"
            for a in approved:
                approved_md += f"- {a}\n"

        return f"""### {agent.title()} Review (Iteration {self.state.iteration})

*{timestamp}* | **Verdict**: {verdict} | **Confidence**: {review.get('confidence', 'N/A')}
{issues_md}{approved_md}
{f"> **Notes**: {review.get('notes_for_claude', review.get('notes_for_gemini', ''))}" if review.get('notes_for_claude') or review.get('notes_for_gemini') else ''}
"""

    def _format_decision_entry(self, entry: Dict) -> str:
        """Format a decision log entry."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"| {entry.get('phase', 'N/A')} | {entry.get('agent', 'N/A')} | {entry.get('summary', 'N/A')} | {entry.get('confidence', 'N/A')} |\n"

    def _format_conflict_section(self, conflict: Dict) -> str:
        """Format a conflict as a section."""
        claude_pos = conflict.get('claude_position', {})
        gemini_pos = conflict.get('gemini_position', {})
        return f"""### Conflict: {conflict.get('topic', 'Unknown')} (Iteration {conflict.get('iteration', self.state.iteration)})

**Type**: {conflict.get('conflict_type', 'Unknown')}

| Agent | Position | Confidence |
|-------|----------|------------|
| Claude | {claude_pos.get('recommendation', 'N/A')} | {claude_pos.get('confidence', 'N/A')} |
| Gemini | {gemini_pos.get('recommendation', 'N/A')} | {gemini_pos.get('confidence', 'N/A')} |

**Status**: PENDING RESOLUTION
"""

    # --- Gemini Invocation ---

    def invoke_gemini(self, action: str, payload: Dict) -> Dict:
        """Invoke Gemini with a collaboration skill."""
        prompt = json.dumps(payload, indent=2)

        self._log(f"Invoking Gemini: {action}")

        cmd = [
            "gemini",
            "-p", f"Execute the collaborate-{action} skill with this input:\n\n{prompt}",
            "-o", "json",
            "--yolo"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                self._log(f"Gemini error: {result.stderr}", "ERROR")
                return {"status": "error", "error": result.stderr}

            output = result.stdout

            # Log raw response for debugging (truncated)
            self._log(f"Gemini raw response (first 500 chars): {output[:500]}")

            response = self._parse_gemini_response(output)
            self._log(f"Gemini response parsed, keys: {list(response.keys())}")
            return response

        except subprocess.TimeoutExpired:
            self._log("Gemini invocation timed out", "ERROR")
            return {"status": "error", "error": "timeout"}
        except Exception as e:
            self._log(f"Gemini invocation failed: {e}", "ERROR")
            return {"status": "error", "error": str(e)}

    def _parse_gemini_response(self, output: str) -> Dict:
        """Parse Gemini response, handling various formats including JSON in markdown."""
        import re

        # First, strip markdown code blocks to avoid picking up JSON examples
        # This removes ```json ... ``` blocks that might contain example JSON
        clean_output = re.sub(r'```(?:json|python|cypher)?\n.*?```', '', output, flags=re.DOTALL)

        # Try to find JSON in cleaned output first
        try:
            return json.loads(clean_output.strip())
        except json.JSONDecodeError:
            pass

        # Find JSON object(s) in the cleaned output
        json_objects = []
        brace_count = 0
        start_idx = None

        for i, char in enumerate(clean_output):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    try:
                        obj = json.loads(clean_output[start_idx:i+1])
                        json_objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start_idx = None

        # If nothing found in cleaned output, try original (in case response IS the JSON)
        if not json_objects:
            try:
                return json.loads(output.strip())
            except json.JSONDecodeError:
                pass

            # Last resort: find JSON in original
            brace_count = 0
            start_idx = None
            for i, char in enumerate(output):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx is not None:
                        try:
                            obj = json.loads(output[start_idx:i+1])
                            json_objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start_idx = None

        if not json_objects:
            self._log("No valid JSON found in Gemini response", "WARNING")
            return {"status": "complete", "raw_response": output}

        # If multiple JSON objects, find the one with expected workflow fields
        expected_fields = {'verdict', 'major_issues', 'confidence', 'sections_added',
                          'checklist', 'status', 'action', 'output_document_id'}
        for obj in json_objects:
            if any(field in obj for field in expected_fields):
                self._log(f"Found response JSON with fields: {[f for f in expected_fields if f in obj]}")
                return obj

        # Return the largest object if no expected fields found
        largest = max(json_objects, key=lambda x: len(str(x)))
        self._log(f"Using largest JSON object with keys: {list(largest.keys())[:5]}")
        return largest

    # --- Conflict Detection ---

    def detect_conflict(self, gemini_response: Dict) -> Optional[Conflict]:
        """Detect if there's a conflict between Claude and Gemini."""
        verdict = gemini_response.get("verdict", "").upper()
        if verdict == "REJECTED":
            return Conflict(
                topic="Overall approach",
                claude_position=Position(
                    agent="claude",
                    recommendation=self.state.claude_position.get("recommendation", "current proposal") if self.state.claude_position else "current proposal",
                    confidence=self.state.claude_position.get("confidence", 0.75) if self.state.claude_position else 0.75,
                    rationale=self.state.claude_position.get("rationale", "") if self.state.claude_position else ""
                ),
                gemini_position=Position(
                    agent="gemini",
                    recommendation="Rejected - needs rethink",
                    confidence=gemini_response.get("confidence", 0.8),
                    rationale=gemini_response.get("notes_for_claude", "")
                ),
                conflict_type=ConflictType.EXPLICIT_REJECT,
                iteration=self.state.iteration
            )

        blockers = gemini_response.get("blockers", [])
        for blocker in blockers:
            if blocker.get("severity") in ["blocker", "major"]:
                if self.state.iteration >= self.MAX_ITERATIONS:
                    return Conflict(
                        topic=blocker.get("issue", "Unknown issue"),
                        claude_position=Position(
                            agent="claude",
                            recommendation=self.state.claude_position.get("recommendation", "") if self.state.claude_position else "",
                            confidence=self.state.claude_position.get("confidence", 0.75) if self.state.claude_position else 0.75,
                            rationale=self.state.claude_position.get("rationale", "") if self.state.claude_position else ""
                        ),
                        gemini_position=Position(
                            agent="gemini",
                            recommendation=blocker.get("my_position", ""),
                            confidence=gemini_response.get("confidence", 0.8),
                            rationale=blocker.get("description", "")
                        ),
                        conflict_type=ConflictType.ITERATION_LIMIT,
                        iteration=self.state.iteration
                    )

        gemini_conf = gemini_response.get("confidence", 0.5)
        claude_conf = self.state.claude_position.get("confidence", 0.5) if self.state.claude_position else 0.5

        if (gemini_conf > self.CONFIDENCE_THRESHOLD and
            claude_conf > self.CONFIDENCE_THRESHOLD and
            not gemini_response.get("agrees_with_proposal", True)):

            conf_delta = abs(gemini_conf - claude_conf)
            if conf_delta > self.CONFLICT_DELTA_THRESHOLD or self.state.iteration >= self.MAX_ITERATIONS:
                return Conflict(
                    topic="High confidence disagreement",
                    claude_position=Position(
                        agent="claude",
                        recommendation=self.state.claude_position.get("recommendation", "") if self.state.claude_position else "",
                        confidence=claude_conf,
                        rationale=self.state.claude_position.get("rationale", "") if self.state.claude_position else ""
                    ),
                    gemini_position=Position(
                        agent="gemini",
                        recommendation=gemini_response.get("notes_for_claude", "alternative approach"),
                        confidence=gemini_conf,
                        rationale=str(gemini_response.get("blockers", []))
                    ),
                    conflict_type=ConflictType.HIGH_CONFIDENCE_DISAGREEMENT,
                    iteration=self.state.iteration
                )

        return None

    def generate_interview(self, conflict: Conflict) -> Dict:
        """Generate interview questions for human conflict resolution."""
        return {
            "conflict_id": str(uuid.uuid4()),
            "topic": conflict.topic,
            "conflict_type": conflict.conflict_type.value,
            "iteration": conflict.iteration,
            "positions": {
                "claude": {
                    "recommendation": conflict.claude_position.recommendation,
                    "confidence": conflict.claude_position.confidence,
                    "rationale": conflict.claude_position.rationale
                },
                "gemini": {
                    "recommendation": conflict.gemini_position.recommendation,
                    "confidence": conflict.gemini_position.confidence,
                    "rationale": conflict.gemini_position.rationale
                }
            },
            "questions": [
                {
                    "id": "q1",
                    "question": f"On '{conflict.topic}', which approach?",
                    "options": [
                        {"value": "claude", "label": f"Claude: {conflict.claude_position.recommendation}"},
                        {"value": "gemini", "label": f"Gemini: {conflict.gemini_position.recommendation}"},
                        {"value": "hybrid", "label": "Hybrid of both"},
                        {"value": "other", "label": "Different approach"}
                    ]
                },
                {
                    "id": "q2",
                    "question": "What factors matter most?",
                    "options": [
                        {"value": "speed", "label": "Ship fast"},
                        {"value": "quality", "label": "Code quality"},
                        {"value": "scalability", "label": "Scalability"},
                        {"value": "simplicity", "label": "Simplicity"}
                    ],
                    "multi_select": True
                }
            ],
            "workflow_state": {
                "workflow_id": self.state.workflow_id,
                "checkpoint_path": str(self._checkpoint_path()),
                "phase": self.state.phase.value,
                "project_name": self.state.project_name
            }
        }

    def apply_resolution(self, resolution: Dict):
        """Apply human resolution to conflict."""
        conflict_data = {
            "topic": resolution.get("topic"),
            "resolution": resolution.get("decision"),
            "decided_by": resolution.get("decided_by", "human"),
            "rationale": resolution.get("rationale", ""),
            "resolved_at": datetime.now().isoformat()
        }

        # Append resolution to master doc
        resolution_md = f"""### Resolution: {resolution.get('topic', 'Unknown')}

**Decision**: {resolution.get('decision', 'N/A')}
**Decided By**: {resolution.get('decided_by', 'human')}
**Rationale**: {resolution.get('rationale', 'N/A')}
"""
        self.append_section(
            f"resolution-{self.state.iteration}-{resolution.get('topic', 'unknown')}",
            resolution_md
        )

        self.state.resolved_conflicts.append(conflict_data)
        self.state.conflicts = [c for c in self.state.conflicts
                                if c.get("topic") != resolution.get("topic")]
        self.state.phase = Phase.SYNTHESIZING
        self.state.iteration = 0
        self.save_checkpoint()
        self._log(f"Resolution applied: {conflict_data}")

    # --- Phase Handlers ---

    def run_phase(self) -> Dict:
        """Execute the current phase and return result."""
        phase_handlers = {
            Phase.INITIALIZING: self._phase_initialize,
            Phase.DRAFTING: self._phase_draft,
            Phase.EXPANDING_GEMINI: self._phase_expand_gemini,
            Phase.EXPANDING_CLAUDE: self._phase_expand_claude,
            Phase.REVIEWING_GEMINI: self._phase_review_gemini,
            Phase.REVIEWING_CLAUDE: self._phase_review_claude,
            Phase.SYNTHESIZING: self._phase_synthesize,
            Phase.APPROVING: self._phase_approve,
            Phase.CONFLICT: self._phase_conflict,
            # Completion phases
            Phase.APPROVED: self._phase_approved,
            Phase.IMPLEMENTING: self._phase_implementing,
            Phase.VALIDATING: self._phase_validating,
            Phase.VALIDATION_FAILED: self._phase_validation_failed,
            Phase.LESSONS_LEARNED: self._phase_lessons_learned,
            Phase.GRADUATING: self._phase_graduating,
            Phase.COMPLETED: self._phase_completed,
            Phase.ABANDONED: self._phase_abandoned,
        }

        handler = phase_handlers.get(self.state.phase)
        if handler:
            return handler()
        else:
            return {"status": "error", "message": f"No handler for phase: {self.state.phase}"}

    def _phase_initialize(self) -> Dict:
        self._log("Phase: Initializing")
        self.state.context_loaded = True
        self.state.phase = Phase.DRAFTING
        self.save_checkpoint()
        return {
            "status": "continue",
            "next_phase": "drafting",
            "action_required": "claude_draft",
            "message": "Context loaded. Claude should create initial draft."
        }

    def _phase_draft(self) -> Dict:
        self._log("Phase: Drafting")
        # Check if draft section has been appended
        if "draft" in self.state.sections_appended:
            self.state.phase = Phase.EXPANDING_GEMINI
            self.update_status_header()
            self.save_checkpoint()
            return {
                "status": "continue",
                "next_phase": "expanding_gemini",
                "message": "Draft received. Moving to Gemini expansion."
            }
        return {
            "status": "waiting",
            "action_required": "claude_draft",
            "message": "Waiting for Claude to provide draft content via --draft flag."
        }

    def _phase_expand_gemini(self) -> Dict:
        self._log("Phase: Gemini Expansion")

        payload = {
            "action": "expand",
            "document_id": self.state.master_doc_id,
            "collection_id": self.state.collection_id,
            "context": {
                "project_name": self.state.project_name,
                "iteration": self.state.iteration,
                "previous_docs": []
            }
        }

        response = self.invoke_gemini("expand", payload)

        if response.get("status") == "error":
            return {"status": "error", "error": response.get("error")}

        # Extract Gemini's position
        self.state.gemini_position = {
            "recommendation": response.get("summary", ""),
            "confidence": response.get("confidence", 0.75),
            "rationale": str(response.get("gaps_identified", [])),
            "notes_for_claude": response.get("notes_for_claude", ""),
            "sections_added": response.get("sections_added", []),
            "research_sources": response.get("research_sources", [])
        }

        # Track Gemini's expansion sub-doc
        gemini_doc_id = response.get("output_document_id")
        sub_doc_key = f"gemini-expansion-{self.state.iteration}"
        if gemini_doc_id:
            self.state.sub_docs[sub_doc_key] = gemini_doc_id

        # Update master doc with sub-doc link and summary
        sections_list = "\n".join(f"- {s}" for s in response.get("sections_added", []))
        gaps_list = "\n".join(f"- {g}" for g in response.get("gaps_identified", []))

        expansion_summary = f"""## Gemini Expansion (Iteration {self.state.iteration})

**Confidence**: {self.state.gemini_position.get('confidence', 'N/A')}

**Summary**: {response.get('summary', 'N/A')}

**Sections Added:**
{sections_list or "*None specified*"}

**Gaps Identified:**
{gaps_list or "*None*"}

> **Notes for Claude**: {response.get('notes_for_claude', 'None')}
"""
        self.append_section(f"gemini-expansion-{self.state.iteration}", expansion_summary)

        # Update sub-documents table in master doc
        self._update_sub_docs_table()

        self.state.phase = Phase.EXPANDING_CLAUDE
        self.update_status_header()
        self.save_checkpoint()

        return {
            "status": "continue",
            "next_phase": "expanding_claude",
            "gemini_response": response,
            "action_required": "claude_expand",
            "message": "Gemini expansion complete. Claude should expand now."
        }

    def _phase_expand_claude(self) -> Dict:
        self._log("Phase: Claude Expansion")
        if "claude-expansion" in [s.split("-")[0] + "-" + s.split("-")[1] for s in self.state.sections_appended if s.startswith("claude-expansion")]:
            self.state.phase = Phase.REVIEWING_GEMINI
            self.state.iteration += 1
            self.update_status_header()
            self.save_checkpoint()
            return {
                "status": "continue",
                "next_phase": "reviewing_gemini",
                "message": "Claude expansion received. Moving to cross-review."
            }
        return {
            "status": "waiting",
            "action_required": "claude_expand",
            "gemini_position": self.state.gemini_position,
            "message": "Waiting for Claude to provide expansion via --claude-expansion flag."
        }

    def _phase_review_gemini(self) -> Dict:
        self._log("Phase: Gemini Review")

        payload = {
            "action": "review",
            "document_id": self.state.master_doc_id,
            "author": "claude",
            "collection_id": self.state.collection_id,
            "context": {
                "project_name": self.state.project_name,
                "iteration": self.state.iteration,
                "my_previous_position": self.state.gemini_position
            }
        }

        response = self.invoke_gemini("review", payload)

        if response.get("status") == "error":
            return {"status": "error", "error": response.get("error")}

        self.state.gemini_position = {
            "recommendation": response.get("notes_for_claude", ""),
            "confidence": response.get("confidence", 0.75),
            "rationale": str(response.get("blockers", [])),
            "verdict": response.get("verdict", ""),
            "blockers": response.get("blockers", []),
            "major_issues": response.get("major_issues", []),
            "minor_issues": response.get("minor_issues", []),
            "approved_aspects": response.get("approved_aspects", [])
        }

        # Create Gemini review as sub-doc with full depth
        review_key = f"gemini-review-{self.state.iteration}"
        review_content = self._format_review_section("gemini", response)
        self.create_sub_doc(
            f"{self.state.project_name} - Gemini Review (Iteration {self.state.iteration})",
            review_content,
            review_key
        )
        self._update_sub_docs_table()

        # Add verdict summary to master doc
        verdict_summary = f"""## Review Cycle {self.state.iteration}

**Gemini Verdict**: {response.get('verdict', 'N/A')} ({response.get('confidence', 'N/A')} confidence)

**Major Issues**: {len(response.get('major_issues', []))} | **Minor Issues**: {len(response.get('minor_issues', []))}

> See sub-document for full review with evidence and arguments.
"""
        self.append_section(f"gemini-review-summary-{self.state.iteration}", verdict_summary)

        # Check for conflict
        conflict = self.detect_conflict(response)
        if conflict:
            self.state.conflicts.append(asdict(conflict))
            conflict_md = self._format_conflict_section(asdict(conflict))
            self.append_section(f"conflict-{self.state.iteration}", conflict_md)
            self.state.phase = Phase.CONFLICT
            self.update_status_header()
            self.save_checkpoint()
            return {
                "status": "conflict",
                "conflict": asdict(conflict),
                "interview": self.generate_interview(conflict),
                "message": "Conflict detected. Human interview required."
            }

        self.state.phase = Phase.REVIEWING_CLAUDE
        self.update_status_header()
        self.save_checkpoint()

        return {
            "status": "continue",
            "next_phase": "reviewing_claude",
            "gemini_response": response,
            "action_required": "claude_review",
            "message": "Gemini review complete. Claude should review Gemini's expansion."
        }

    def _phase_review_claude(self) -> Dict:
        self._log("Phase: Claude Review")
        if any(s.startswith("claude-review") for s in self.state.sections_appended):
            self.state.phase = Phase.SYNTHESIZING
            self.update_status_header()
            self.save_checkpoint()
            return {
                "status": "continue",
                "next_phase": "synthesizing",
                "message": "Reviews complete. Moving to synthesis."
            }
        return {
            "status": "waiting",
            "action_required": "claude_review",
            "gemini_position": self.state.gemini_position,
            "message": "Waiting for Claude review via --claude-review flag."
        }

    def _phase_synthesize(self) -> Dict:
        self._log("Phase: Synthesis")
        if any(s.startswith("synthesis") for s in self.state.sections_appended):
            self.state.phase = Phase.APPROVING
            self.update_status_header()
            self.save_checkpoint()
            return {
                "status": "continue",
                "next_phase": "approving",
                "message": "Synthesis complete. Moving to final approval."
            }
        return {
            "status": "waiting",
            "action_required": "claude_synthesize",
            "resolved_conflicts": self.state.resolved_conflicts,
            "message": "Waiting for Claude to synthesize final plan."
        }

    def _phase_approve(self) -> Dict:
        self._log("Phase: Final Approval")

        payload = {
            "action": "approve",
            "document_id": self.state.master_doc_id,
            "collection_id": self.state.collection_id,
            "context": {
                "project_name": self.state.project_name,
                "iteration": self.state.iteration,
                "resolved_conflicts": self.state.resolved_conflicts,
            }
        }

        response = self.invoke_gemini("approve", payload)

        if response.get("status") == "error":
            return {"status": "error", "error": response.get("error")}

        verdict = response.get("verdict", "").upper()

        self.state.gemini_position = {
            "recommendation": f"Final verdict: {verdict}",
            "confidence": response.get("confidence", 0.75),
            "rationale": str(response.get("remaining_concerns", [])),
            "verdict": verdict,
        }

        # Append approval to master doc
        conditions = response.get("sign_off", {}).get("conditions", [])
        conditions_md = "\n".join(f"- {c}" for c in conditions) if conditions else "*None*"

        concerns = response.get("remaining_concerns", [])
        concerns_md = ""
        if concerns:
            concerns_md = "\n**Remaining Concerns:**\n"
            for c in concerns:
                if isinstance(c, dict):
                    concerns_md += f"- {c.get('concern', str(c))}\n"
                else:
                    concerns_md += f"- {c}\n"

        approval_md = f"""## Approval

**Verdict**: {verdict}
**Confidence**: {response.get('confidence', 'N/A')}
**Approved By**: Gemini

**Conditions:**
{conditions_md}
{concerns_md}
"""
        self.append_section(f"approval-{self.state.iteration}", approval_md)

        # Append decision log
        decision_md = f"""## Decision Log

| Phase | Agent | Decision | Confidence |
|-------|-------|----------|------------|
"""
        self.append_section("decision-log-header", decision_md)

        if verdict == "APPROVED":
            self.state.phase = Phase.APPROVED
            self.update_status_header()
            self.save_checkpoint()
            return {
                "status": "complete",
                "verdict": "APPROVED",
                "gemini_response": response,
                "message": "Plan approved! Ready for implementation."
            }
        elif verdict == "CHANGES_REQUESTED":
            conflict = self.detect_conflict(response)
            if conflict:
                self.state.conflicts.append(asdict(conflict))
                self.state.phase = Phase.CONFLICT
                self.save_checkpoint()
                return {
                    "status": "conflict",
                    "conflict": asdict(conflict),
                    "interview": self.generate_interview(conflict),
                    "message": "Conflict in final approval. Human interview required."
                }

            self.state.phase = Phase.SYNTHESIZING
            self.state.iteration += 1
            self.save_checkpoint()
            return {
                "status": "continue",
                "next_phase": "synthesizing",
                "gemini_response": response,
                "action_required": "claude_revise",
                "message": "Changes requested. Claude should revise synthesis."
            }
        else:
            self.state.phase = Phase.REJECTED
            self.update_status_header()
            self.save_checkpoint()
            return {
                "status": "complete",
                "verdict": "REJECTED",
                "gemini_response": response,
                "message": "Plan rejected. Needs fundamental rethink."
            }

    def _phase_conflict(self) -> Dict:
        self._log("Phase: Conflict Resolution")
        if not self.state.conflicts:
            self.state.phase = Phase.SYNTHESIZING
            self.save_checkpoint()
            return {
                "status": "continue",
                "next_phase": "synthesizing",
                "message": "Conflicts resolved. Continuing to synthesis."
            }

        conflict = self.state.conflicts[0]
        interview = self.generate_interview(Conflict(
            topic=conflict.get("topic", "Unknown"),
            claude_position=Position(**conflict.get("claude_position", {})),
            gemini_position=Position(**conflict.get("gemini_position", {})),
            conflict_type=ConflictType(conflict.get("conflict_type", "position_mismatch")),
            iteration=conflict.get("iteration", 0)
        ))

        return {
            "status": "conflict",
            "interview": interview,
            "message": "Waiting for human resolution."
        }

    # --- Completion Phase Handlers ---

    def _phase_approved(self) -> Dict:
        """Plan approved - ready for implementation."""
        self._log("Phase: Approved (ready for implementation)")
        return {
            "status": "waiting",
            "phase": "approved",
            "action_required": "start_implementation",
            "commands": [
                "collaborate.py implement <checkpoint> - Start implementation tracking",
                "collaborate.py abandon <checkpoint> --reason '...' - Abandon project"
            ],
            "message": "Plan approved! Use 'implement' command to begin implementation or 'abandon' to cancel."
        }

    def _phase_implementing(self) -> Dict:
        """Implementation in progress."""
        self._log("Phase: Implementing")
        notes_count = len(self.state.implementation_notes)
        return {
            "status": "waiting",
            "phase": "implementing",
            "action_required": "continue_implementation",
            "implementation_notes": self.state.implementation_notes,
            "commands": [
                f"collaborate.py note <checkpoint> --text '...' - Add implementation note ({notes_count} notes so far)",
                "collaborate.py validate <checkpoint> - Move to validation when ready",
                "collaborate.py abandon <checkpoint> --reason '...' - Abandon project"
            ],
            "message": f"Implementation in progress. {notes_count} notes recorded. Use 'validate' when ready."
        }

    def _phase_validating(self) -> Dict:
        """Validation phase - Claude and Gemini verify implementation."""
        self._log("Phase: Validating")

        # Check if validation has been completed
        if self.state.validation_results:
            results = self.state.validation_results
            all_passed = all([
                results.get("automated", {}).get("passed", False),
                results.get("behavioral", {}).get("passed", False),
                results.get("peer_review", {}).get("passed", False)
            ])

            if all_passed:
                self.state.phase = Phase.LESSONS_LEARNED
                self.update_status_header()
                self.save_checkpoint()
                return {
                    "status": "continue",
                    "next_phase": "lessons_learned",
                    "validation_results": results,
                    "message": "Validation passed! Moving to lessons learned."
                }
            else:
                self.state.phase = Phase.VALIDATION_FAILED
                self.update_status_header()
                self.save_checkpoint()
                return {
                    "status": "continue",
                    "next_phase": "validation_failed",
                    "validation_results": results,
                    "message": "Validation failed. Human interview required."
                }

        # Run validation
        return {
            "status": "waiting",
            "phase": "validating",
            "action_required": "run_validation",
            "validation_strategy": {
                "automated": "Tests, linting, type checks",
                "behavioral": "Verify side-effects match plan",
                "peer_review": "Gemini reviews against synthesis"
            },
            "commands": [
                "collaborate.py validate-automated <checkpoint> --results <json>",
                "collaborate.py validate-behavioral <checkpoint> --results <json>",
                "collaborate.py validate-review <checkpoint>"
            ],
            "message": "Waiting for validation. Run validation checks and report results."
        }

    def _phase_validation_failed(self) -> Dict:
        """Validation failed - need human interview."""
        self._log("Phase: Validation Failed")

        failed_checks = []
        if self.state.validation_results:
            for check, result in self.state.validation_results.items():
                if isinstance(result, dict) and not result.get("passed", True):
                    failed_checks.append({
                        "check": check,
                        "reason": result.get("reason", "Unknown"),
                        "details": result.get("details", {})
                    })

        interview = {
            "phase": "validation_failed",
            "failed_checks": failed_checks,
            "questions": [
                {
                    "id": "q1",
                    "question": "What caused the validation failure?",
                    "options": [
                        {"value": "implementation_bug", "label": "Implementation bug - needs fix"},
                        {"value": "plan_incomplete", "label": "Plan was incomplete - needs revision"},
                        {"value": "test_issue", "label": "Test/validation issue - not implementation"},
                        {"value": "scope_change", "label": "Scope changed - plan needs update"}
                    ]
                },
                {
                    "id": "q2",
                    "question": "What action should be taken?",
                    "options": [
                        {"value": "fix_and_retry", "label": "Fix issues and re-validate"},
                        {"value": "revise_plan", "label": "Go back to planning phase"},
                        {"value": "override", "label": "Override - accept as-is with known issues"},
                        {"value": "abandon", "label": "Abandon project"}
                    ]
                }
            ],
            "workflow_state": {
                "workflow_id": self.state.workflow_id,
                "checkpoint_path": str(self._checkpoint_path()),
                "phase": self.state.phase.value,
                "project_name": self.state.project_name
            }
        }

        return {
            "status": "interview_required",
            "phase": "validation_failed",
            "interview": interview,
            "message": "Validation failed. Human interview required to determine next steps."
        }

    def _phase_lessons_learned(self) -> Dict:
        """Lessons learned interview phase."""
        self._log("Phase: Lessons Learned")

        if self.state.lessons_learned:
            # Lessons already captured, move to graduation
            self.state.phase = Phase.GRADUATING
            self.update_status_header()
            self.save_checkpoint()
            return {
                "status": "continue",
                "next_phase": "graduating",
                "lessons_learned": self.state.lessons_learned,
                "message": "Lessons captured. Moving to graduation."
            }

        return {
            "status": "interview_required",
            "phase": "lessons_learned",
            "interview": {
                "questions": LESSONS_LEARNED_QUESTIONS,
                "workflow_state": {
                    "workflow_id": self.state.workflow_id,
                    "checkpoint_path": str(self._checkpoint_path()),
                    "project_name": self.state.project_name
                }
            },
            "commands": [
                "collaborate.py lessons-learned <checkpoint> --responses <json>"
            ],
            "message": "Lessons learned interview required. Capture insights for knowledge base."
        }

    def _phase_graduating(self) -> Dict:
        """Graduate knowledge to Qdrant."""
        self._log("Phase: Graduating")

        if self.state.graduation_manifest:
            # Already graduated
            self.state.phase = Phase.COMPLETED
            self.update_status_header()
            self.save_checkpoint()
            return {
                "status": "continue",
                "next_phase": "completed",
                "graduation_manifest": self.state.graduation_manifest,
                "message": "Knowledge graduated. Project complete!"
            }

        # Build graduation manifest
        manifest = self._build_graduation_manifest()

        return {
            "status": "waiting",
            "phase": "graduating",
            "action_required": "graduate_knowledge",
            "manifest": manifest,
            "commands": [
                "collaborate.py graduate <checkpoint> - Execute graduation",
                "collaborate.py graduate <checkpoint> --dry-run - Preview graduation"
            ],
            "message": "Ready to graduate knowledge to Qdrant. Use 'graduate' command."
        }

    def _phase_completed(self) -> Dict:
        """Project completed."""
        self._log("Phase: Completed")
        return {
            "status": "complete",
            "phase": "completed",
            "project_name": self.state.project_name,
            "workflow_id": self.state.workflow_id,
            "graduation_manifest": self.state.graduation_manifest,
            "lessons_learned": self.state.lessons_learned,
            "message": "Project completed! Knowledge has been graduated to the knowledge base."
        }

    def _phase_abandoned(self) -> Dict:
        """Project abandoned."""
        self._log("Phase: Abandoned")
        return {
            "status": "complete",
            "phase": "abandoned",
            "project_name": self.state.project_name,
            "workflow_id": self.state.workflow_id,
            "abandon_reason": self.state.abandon_reason,
            "message": f"Project abandoned. Reason: {self.state.abandon_reason or 'Not specified'}"
        }

    # --- Graduation Helpers ---

    def _build_graduation_manifest(self) -> Dict:
        """Build manifest of knowledge to graduate to Qdrant."""
        manifest = {
            "project_id": self.state.workflow_id,
            "project_name": self.state.project_name,
            "collection_id": self.state.collection_id,
            "created_at": self.state.created_at,
            "completed_at": datetime.now().isoformat(),
            "decisions": [],
            "patterns": [],
            "lessons": []
        }

        # Extract decisions from resolved conflicts
        for conflict in self.state.resolved_conflicts:
            content_hash = hashlib.sha256(
                f"{conflict.get('topic', '')}-{conflict.get('resolution', '')}".encode()
            ).hexdigest()[:16]

            manifest["decisions"].append({
                "id": f"decision-{content_hash}",
                "topic": conflict.get("topic", "Unknown"),
                "resolution": conflict.get("resolution", ""),
                "decided_by": conflict.get("decided_by", "unknown"),
                "rationale": conflict.get("rationale", ""),
                "domain": "decisions",
                "decay_rate": DOMAIN_DECAY_RATES["decisions"],
                "tags": [self.state.project_name.lower().replace(" ", "-")]
            })

        # Extract patterns from lessons learned
        if self.state.lessons_learned:
            patterns = self.state.lessons_learned.get("patterns", [])
            for pattern in patterns:
                content_hash = hashlib.sha256(
                    f"{pattern.get('title', '')}-{pattern.get('implementation', '')}".encode()
                ).hexdigest()[:16]

                domain = pattern.get("domain", "patterns")
                manifest["patterns"].append({
                    "id": f"pattern-{content_hash}",
                    "title": pattern.get("title", ""),
                    "context": pattern.get("context", ""),
                    "implementation": pattern.get("implementation", ""),
                    "domain": domain,
                    "decay_rate": DOMAIN_DECAY_RATES.get(domain, 0.05),
                    "tags": [self.state.project_name.lower().replace(" ", "-")]
                })

            # Extract lessons/challenges
            challenges = self.state.lessons_learned.get("challenges", [])
            for challenge in challenges:
                content_hash = hashlib.sha256(
                    f"{challenge.get('symptom', '')}-{challenge.get('root_cause', '')}".encode()
                ).hexdigest()[:16]

                manifest["lessons"].append({
                    "id": f"lesson-{content_hash}",
                    "symptom": challenge.get("symptom", ""),
                    "root_cause": challenge.get("root_cause", ""),
                    "solution": challenge.get("solution", ""),
                    "key_insight": challenge.get("key_insight", ""),
                    "domain": "lessons",
                    "decay_rate": DOMAIN_DECAY_RATES["lessons"],
                    "tags": [self.state.project_name.lower().replace(" ", "-")]
                })

        return manifest

    def graduate_to_qdrant(self, dry_run: bool = False) -> Dict:
        """Graduate knowledge to Qdrant collections."""
        manifest = self._build_graduation_manifest()
        results = {
            "decisions_graduated": 0,
            "patterns_graduated": 0,
            "lessons_graduated": 0,
            "errors": []
        }

        if dry_run:
            self._log("Graduation dry run - no changes made")
            return {
                "status": "dry_run",
                "manifest": manifest,
                "would_graduate": {
                    "decisions": len(manifest["decisions"]),
                    "patterns": len(manifest["patterns"]),
                    "lessons": len(manifest["lessons"])
                }
            }

        # Graduate decisions
        for decision in manifest["decisions"]:
            try:
                result = self._call_mcp("add_decision", {
                    "topic": decision["topic"],
                    "decision": decision["resolution"],
                    "rationale": decision["rationale"],
                    "decided_by": decision["decided_by"],
                    "tags": decision["tags"]
                })
                if "error" not in result:
                    results["decisions_graduated"] += 1
                else:
                    results["errors"].append(f"Decision '{decision['topic']}': {result['error']}")
            except Exception as e:
                results["errors"].append(f"Decision '{decision['topic']}': {str(e)}")

        # Graduate patterns to documentation
        for pattern in manifest["patterns"]:
            try:
                content = f"""# {pattern['title']}

## When to Use
{pattern['context']}

## Implementation
{pattern['implementation']}

## Metadata
- **Domain**: {pattern['domain']}
- **Decay Rate**: {pattern['decay_rate']}
- **Tags**: {', '.join(pattern['tags'])}
"""
                result = self._call_mcp("add_documentation", {
                    "title": pattern["title"],
                    "content": content,
                    "tags": pattern["tags"]
                })
                if "error" not in result:
                    results["patterns_graduated"] += 1
                else:
                    results["errors"].append(f"Pattern '{pattern['title']}': {result['error']}")
            except Exception as e:
                results["errors"].append(f"Pattern '{pattern['title']}': {str(e)}")

        # Graduate lessons
        for lesson in manifest["lessons"]:
            try:
                content = f"""## Symptom
{lesson['symptom']}

## Root Cause
{lesson['root_cause']}

## Solution
{lesson['solution']}

## Key Insight
{lesson['key_insight']}
"""
                # Store as documentation with lessons tag
                result = self._call_mcp("add_documentation", {
                    "title": f"Lesson: {lesson['symptom'][:50]}...",
                    "content": content,
                    "tags": ["lessons-learned"] + lesson["tags"]
                })
                if "error" not in result:
                    results["lessons_graduated"] += 1
                else:
                    results["errors"].append(f"Lesson: {result['error']}")
            except Exception as e:
                results["errors"].append(f"Lesson: {str(e)}")

        # Store manifest
        self.state.graduation_manifest = manifest
        self.state.graduation_manifest["results"] = results

        # Append graduation summary to master doc
        graduation_md = f"""## Graduation Summary

**Graduated at**: {datetime.now().isoformat()}

| Category | Count |
|----------|-------|
| Decisions | {results['decisions_graduated']} |
| Patterns | {results['patterns_graduated']} |
| Lessons | {results['lessons_graduated']} |

{f"**Errors**: {len(results['errors'])}" if results['errors'] else ""}
"""
        self.append_section("graduation-summary", graduation_md)

        self.state.phase = Phase.COMPLETED
        self.update_status_header()
        self.save_checkpoint()

        return {
            "status": "graduated",
            "results": results,
            "manifest": manifest
        }

    def abandon_project(self, reason: str) -> Dict:
        """Abandon the project with a reason."""
        self.state.abandon_reason = reason
        self.state.phase = Phase.ABANDONED
        self.update_status_header()

        # Append abandon section to master doc
        abandon_md = f"""## Project Abandoned

**Abandoned at**: {datetime.now().isoformat()}
**Reason**: {reason}
**Phase at abandonment**: {self.state.phase.value}
"""
        self.append_section("abandonment", abandon_md)

        self.save_checkpoint()
        self._log(f"Project abandoned: {reason}")

        return {
            "status": "abandoned",
            "reason": reason,
            "workflow_id": self.state.workflow_id
        }

    # --- Status ---

    def get_status(self) -> Dict:
        status = {
            "workflow_id": self.state.workflow_id,
            "project_name": self.state.project_name,
            "collection_id": self.state.collection_id,
            "phase": self.state.phase.value,
            "iteration": self.state.iteration,
            "master_doc_id": self.state.master_doc_id,
            "has_conflicts": len(self.state.conflicts) > 0,
            "sections_appended": self.state.sections_appended,
            "checkpoint": str(self._checkpoint_path()),
            "updated_at": self.state.updated_at
        }

        # Add completion flow status if applicable
        if self.state.phase in [Phase.IMPLEMENTING, Phase.VALIDATING, Phase.VALIDATION_FAILED,
                                Phase.LESSONS_LEARNED, Phase.GRADUATING, Phase.COMPLETED, Phase.ABANDONED]:
            status["implementation_notes_count"] = len(self.state.implementation_notes)
            status["validation_results"] = self.state.validation_results
            status["has_lessons_learned"] = self.state.lessons_learned is not None
            status["has_graduation_manifest"] = self.state.graduation_manifest is not None
            if self.state.abandon_reason:
                status["abandon_reason"] = self.state.abandon_reason

        return status

    # --- Start Workflow ---

    @classmethod
    def start(cls, project_name: str, collection_id: str,
              goal: str = "") -> 'CollaborationOrchestrator':
        """Start a new collaboration workflow."""
        workflow_id = f"{project_name.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        state = WorkflowState(
            workflow_id=workflow_id,
            project_name=project_name,
            collection_id=collection_id,
            phase=Phase.INITIALIZING,
            iteration=0,
            goal=goal,
        )

        orchestrator = cls(state)
        orchestrator._log(f"Started new workflow: {workflow_id}")

        # Create the single master doc
        orchestrator.create_master_doc()

        orchestrator.save_checkpoint()
        return orchestrator


def list_collections():
    """List available Outline collections."""
    try:
        response = requests.post(
            f"{KNOWLEDGE_MCP_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "list_collections",
                    "arguments": {}
                },
                "id": str(uuid.uuid4())
            },
            headers={"Accept": "application/json, text/event-stream"},
            timeout=30
        )

        text = response.text
        if text.startswith("event:"):
            for line in text.split("\n"):
                if line.startswith("data:"):
                    json_str = line[5:].strip()
                    result = json.loads(json_str)
                    break
        else:
            result = response.json()

        if "result" in result:
            content = result["result"].get("content", [])
            if content:
                print(content[0].get("text", "No collections found"))
        else:
            print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Claude-Gemini Collaboration Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start new workflow")
    start_parser.add_argument("project_name", help="Project name")
    start_parser.add_argument("--collection", required=True, help="Outline collection ID")
    start_parser.add_argument("--goal", default="", help="Project goal statement")

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume workflow")
    resume_parser.add_argument("--checkpoint", help="Checkpoint file path")

    # Status command
    status_parser = subparsers.add_parser("status", help="Get workflow status")
    status_parser.add_argument("--checkpoint", help="Checkpoint file path")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run current phase")
    run_parser.add_argument("checkpoint", help="Checkpoint file path")

    # Resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Apply conflict resolution")
    resolve_parser.add_argument("checkpoint", help="Checkpoint file path")
    resolve_parser.add_argument("--decision", required=True, help="Decision (claude/gemini/hybrid)")
    resolve_parser.add_argument("--rationale", default="", help="Rationale for decision")

    # Update command (for Claude to update state)
    update_parser = subparsers.add_parser("update", help="Update workflow state")
    update_parser.add_argument("checkpoint", help="Checkpoint file path")
    update_parser.add_argument("--draft", help="Draft content (markdown text)")
    update_parser.add_argument("--claude-expansion", help="Claude expansion content (markdown text)")
    update_parser.add_argument("--claude-review", help="Claude review content (markdown text)")
    update_parser.add_argument("--synthesis", help="Synthesis content (markdown text)")
    update_parser.add_argument("--claude-position", help="Claude position JSON")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split project into sub-project")
    split_parser.add_argument("checkpoint", help="Checkpoint file path")
    split_parser.add_argument("--name", required=True, help="Sub-project name")
    split_parser.add_argument("--scope", required=True, help="Sub-project scope/goal")

    # Collections command
    subparsers.add_parser("collections", help="List Outline collections")

    # Completion flow commands
    implement_parser = subparsers.add_parser("implement", help="Start implementation phase")
    implement_parser.add_argument("checkpoint", help="Checkpoint file path")

    note_parser = subparsers.add_parser("note", help="Add implementation note")
    note_parser.add_argument("checkpoint", help="Checkpoint file path")
    note_parser.add_argument("--text", required=True, help="Note text")

    validate_parser = subparsers.add_parser("validate", help="Start validation phase")
    validate_parser.add_argument("checkpoint", help="Checkpoint file path")

    validate_auto_parser = subparsers.add_parser("validate-automated", help="Report automated validation results")
    validate_auto_parser.add_argument("checkpoint", help="Checkpoint file path")
    validate_auto_parser.add_argument("--passed", action="store_true", help="Validation passed")
    validate_auto_parser.add_argument("--reason", default="", help="Reason if failed")
    validate_auto_parser.add_argument("--details", default="{}", help="Details JSON")

    validate_behav_parser = subparsers.add_parser("validate-behavioral", help="Report behavioral validation results")
    validate_behav_parser.add_argument("checkpoint", help="Checkpoint file path")
    validate_behav_parser.add_argument("--passed", action="store_true", help="Validation passed")
    validate_behav_parser.add_argument("--reason", default="", help="Reason if failed")
    validate_behav_parser.add_argument("--details", default="{}", help="Details JSON")

    validate_review_parser = subparsers.add_parser("validate-review", help="Run Gemini peer review")
    validate_review_parser.add_argument("checkpoint", help="Checkpoint file path")

    validation_resolve_parser = subparsers.add_parser("validation-resolve", help="Resolve validation failure")
    validation_resolve_parser.add_argument("checkpoint", help="Checkpoint file path")
    validation_resolve_parser.add_argument("--action", required=True, choices=["fix_and_retry", "revise_plan", "override", "abandon"], help="Action to take")
    validation_resolve_parser.add_argument("--reason", default="", help="Reason for action")

    lessons_parser = subparsers.add_parser("lessons-learned", help="Submit lessons learned")
    lessons_parser.add_argument("checkpoint", help="Checkpoint file path")
    lessons_parser.add_argument("--responses", required=True, help="Responses JSON")

    graduate_parser = subparsers.add_parser("graduate", help="Graduate knowledge to Qdrant")
    graduate_parser.add_argument("checkpoint", help="Checkpoint file path")
    graduate_parser.add_argument("--dry-run", action="store_true", help="Preview without writing")

    abandon_parser = subparsers.add_parser("abandon", help="Abandon project")
    abandon_parser.add_argument("checkpoint", help="Checkpoint file path")
    abandon_parser.add_argument("--reason", required=True, help="Reason for abandonment")

    # Maintenance commands
    subparsers.add_parser("list", help="List all workflows")

    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old completed/abandoned workflows")
    cleanup_parser.add_argument("--days", type=int, default=30, help="Remove workflows older than N days (default: 30)")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="Show what would be removed")

    archive_parser = subparsers.add_parser("archive", help="Archive a workflow")
    archive_parser.add_argument("checkpoint", help="Checkpoint file path")

    args = parser.parse_args()

    if args.command == "start":
        orchestrator = CollaborationOrchestrator.start(
            args.project_name,
            args.collection,
            args.goal,
        )
        result = orchestrator.run_phase()
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "resume":
        checkpoint = Path(args.checkpoint) if args.checkpoint else max(STATE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        orchestrator = CollaborationOrchestrator.load_checkpoint(checkpoint)
        result = orchestrator.run_phase()
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "status":
        checkpoint = Path(args.checkpoint) if args.checkpoint else max(STATE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        orchestrator = CollaborationOrchestrator.load_checkpoint(checkpoint)
        print(json.dumps(orchestrator.get_status(), indent=2, default=str))

    elif args.command == "run":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        result = orchestrator.run_phase()
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "resolve":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        conflict = orchestrator.state.conflicts[0] if orchestrator.state.conflicts else None
        if conflict:
            resolution = {
                "topic": conflict.get("topic"),
                "decision": args.decision,
                "rationale": args.rationale,
                "decided_by": "human"
            }
            orchestrator.apply_resolution(resolution)
            print(json.dumps({"status": "resolved", "resolution": resolution}, indent=2))
        else:
            print(json.dumps({"status": "error", "message": "No active conflict"}, indent=2))

    elif args.command == "update":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))

        if args.draft:
            # Replace the placeholder in the master doc (don't append - just replace placeholder)
            if orchestrator.state.master_doc_id:
                current = orchestrator._call_mcp("export_document", {
                    "document_id": orchestrator.state.master_doc_id
                })
                current_text = current.get("text", current.get("data", ""))
                current_text = current_text.replace(
                    "*Awaiting initial draft...*",
                    args.draft
                )
                orchestrator._call_mcp("update_document", {
                    "document_id": orchestrator.state.master_doc_id,
                    "text": current_text
                })
            # Track that draft was added (for phase progression)
            if "draft" not in orchestrator.state.sections_appended:
                orchestrator.state.sections_appended.append("draft")

        if args.claude_expansion:
            # Create a sub-doc for Claude's expansion (real depth)
            key = f"claude-expansion-{orchestrator.state.iteration}"
            orchestrator.create_sub_doc(
                f"{orchestrator.state.project_name} - Claude Expansion (Iteration {orchestrator.state.iteration})",
                args.claude_expansion,
                key
            )
            # Add summary to master doc
            orchestrator.append_section(key,
                f"""## Claude Expansion (Iteration {orchestrator.state.iteration})

> See sub-document for full analysis.

""")
            orchestrator._update_sub_docs_table()

        if args.claude_review:
            # Create a sub-doc for Claude's review (real depth)
            key = f"claude-review-{orchestrator.state.iteration}"
            orchestrator.create_sub_doc(
                f"{orchestrator.state.project_name} - Claude Review (Iteration {orchestrator.state.iteration})",
                args.claude_review,
                key
            )
            # Append summary to master doc (required for phase progression check)
            orchestrator.append_section(key,
                f"""## Claude Review (Iteration {orchestrator.state.iteration})

> See sub-document for full review.

""")
            orchestrator._update_sub_docs_table()

        if args.synthesis:
            orchestrator.append_section(f"synthesis-{orchestrator.state.iteration}",
                f"""## Synthesis

{args.synthesis}
""")

        if args.claude_position:
            orchestrator.state.claude_position = json.loads(args.claude_position)
            # Append position inline
            pos = orchestrator.state.claude_position
            phase_name = orchestrator.state.phase.value
            orchestrator.append_section(
                f"claude-position-{phase_name}-{orchestrator.state.iteration}",
                orchestrator._format_position_section("claude", phase_name, pos)
            )

        orchestrator.save_checkpoint()
        result = orchestrator.run_phase()
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "split":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        # Create sub-project as new workflow in the same collection
        sub_orchestrator = CollaborationOrchestrator.start(
            f"{orchestrator.state.project_name} - {args.name}",
            orchestrator.state.collection_id,
            args.scope,
        )
        # Link sub-project from parent
        orchestrator.state.sub_projects.append({
            "name": args.name,
            "workflow_id": sub_orchestrator.state.workflow_id,
            "checkpoint": str(sub_orchestrator._checkpoint_path()),
            "scope": args.scope,
            "created_at": datetime.now().isoformat()
        })
        # Add to parent master doc
        orchestrator.append_section(
            f"sub-project-{args.name.lower().replace(' ', '-')}",
            f"""## Sub-Project: {args.name}

**Scope**: {args.scope}
**Workflow ID**: {sub_orchestrator.state.workflow_id}
**Status**: {sub_orchestrator.state.phase.value}
"""
        )
        orchestrator.save_checkpoint()
        result = sub_orchestrator.run_phase()
        print(json.dumps({
            "status": "split",
            "parent_workflow": orchestrator.state.workflow_id,
            "sub_project": {
                "workflow_id": sub_orchestrator.state.workflow_id,
                "checkpoint": str(sub_orchestrator._checkpoint_path()),
                "master_doc_id": sub_orchestrator.state.master_doc_id
            },
            "sub_project_result": result
        }, indent=2, default=str))

    elif args.command == "collections":
        list_collections()

    # Completion flow command handlers
    elif args.command == "implement":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        if orchestrator.state.phase != Phase.APPROVED:
            print(json.dumps({"status": "error", "message": f"Cannot start implementation from phase: {orchestrator.state.phase.value}. Must be in APPROVED phase."}, indent=2))
        else:
            orchestrator.state.phase = Phase.IMPLEMENTING
            orchestrator.update_status_header()
            # Add implementation section to master doc
            orchestrator.append_section("implementation-start", f"""## Implementation

**Started**: {datetime.now().isoformat()}

### Implementation Notes

*Notes will be added as implementation progresses...*
""")
            orchestrator.save_checkpoint()
            result = orchestrator.run_phase()
            print(json.dumps(result, indent=2, default=str))

    elif args.command == "note":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        if orchestrator.state.phase != Phase.IMPLEMENTING:
            print(json.dumps({"status": "error", "message": "Can only add notes during IMPLEMENTING phase"}, indent=2))
        else:
            note = {
                "timestamp": datetime.now().isoformat(),
                "text": args.text
            }
            orchestrator.state.implementation_notes.append(note)
            # Append to master doc
            orchestrator.append_section(
                f"impl-note-{len(orchestrator.state.implementation_notes)}",
                f"- **{note['timestamp'][:10]}**: {args.text}\n"
            )
            orchestrator.save_checkpoint()
            print(json.dumps({"status": "note_added", "note": note, "total_notes": len(orchestrator.state.implementation_notes)}, indent=2))

    elif args.command == "validate":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        if orchestrator.state.phase != Phase.IMPLEMENTING:
            print(json.dumps({"status": "error", "message": "Can only start validation from IMPLEMENTING phase"}, indent=2))
        else:
            orchestrator.state.phase = Phase.VALIDATING
            orchestrator.state.validation_results = {}  # Initialize validation results
            orchestrator.update_status_header()
            orchestrator.append_section("validation-start", f"""## Validation

**Started**: {datetime.now().isoformat()}

| Check | Status | Details |
|-------|--------|---------|
| Automated | Pending | Tests, linting, type checks |
| Behavioral | Pending | Side-effects verification |
| Peer Review | Pending | Gemini review |
""")
            orchestrator.save_checkpoint()
            result = orchestrator.run_phase()
            print(json.dumps(result, indent=2, default=str))

    elif args.command == "validate-automated":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        if orchestrator.state.phase != Phase.VALIDATING:
            print(json.dumps({"status": "error", "message": "Must be in VALIDATING phase"}, indent=2))
        else:
            if orchestrator.state.validation_results is None:
                orchestrator.state.validation_results = {}
            orchestrator.state.validation_results["automated"] = {
                "passed": args.passed,
                "reason": args.reason,
                "details": json.loads(args.details),
                "timestamp": datetime.now().isoformat()
            }
            orchestrator.save_checkpoint()
            print(json.dumps({"status": "recorded", "check": "automated", "passed": args.passed}, indent=2))

    elif args.command == "validate-behavioral":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        if orchestrator.state.phase != Phase.VALIDATING:
            print(json.dumps({"status": "error", "message": "Must be in VALIDATING phase"}, indent=2))
        else:
            if orchestrator.state.validation_results is None:
                orchestrator.state.validation_results = {}
            orchestrator.state.validation_results["behavioral"] = {
                "passed": args.passed,
                "reason": args.reason,
                "details": json.loads(args.details),
                "timestamp": datetime.now().isoformat()
            }
            orchestrator.save_checkpoint()
            print(json.dumps({"status": "recorded", "check": "behavioral", "passed": args.passed}, indent=2))

    elif args.command == "validate-review":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        if orchestrator.state.phase != Phase.VALIDATING:
            print(json.dumps({"status": "error", "message": "Must be in VALIDATING phase"}, indent=2))
        else:
            # Invoke Gemini for peer review
            payload = {
                "action": "review",
                "document_id": orchestrator.state.master_doc_id,
                "author": "implementation",
                "collection_id": orchestrator.state.collection_id,
                "context": {
                    "project_name": orchestrator.state.project_name,
                    "iteration": orchestrator.state.iteration,
                    "phase": "validation",
                    "implementation_notes": orchestrator.state.implementation_notes
                }
            }
            response = orchestrator.invoke_gemini("review", payload)

            passed = response.get("verdict", "").upper() in ["APPROVED", "CHANGES_REQUESTED"]
            if orchestrator.state.validation_results is None:
                orchestrator.state.validation_results = {}
            orchestrator.state.validation_results["peer_review"] = {
                "passed": passed and not response.get("blockers", []),
                "verdict": response.get("verdict", ""),
                "confidence": response.get("confidence", 0),
                "issues": response.get("major_issues", []) + response.get("minor_issues", []),
                "timestamp": datetime.now().isoformat()
            }
            orchestrator.save_checkpoint()

            # Check if all validations are complete
            results = orchestrator.state.validation_results
            if all(k in results for k in ["automated", "behavioral", "peer_review"]):
                result = orchestrator.run_phase()
                print(json.dumps(result, indent=2, default=str))
            else:
                print(json.dumps({"status": "recorded", "check": "peer_review", "response": response}, indent=2, default=str))

    elif args.command == "validation-resolve":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        if orchestrator.state.phase != Phase.VALIDATION_FAILED:
            print(json.dumps({"status": "error", "message": "Must be in VALIDATION_FAILED phase"}, indent=2))
        else:
            action = args.action
            if action == "fix_and_retry":
                orchestrator.state.phase = Phase.IMPLEMENTING
                orchestrator.state.validation_results = None  # Clear old results
                orchestrator.append_section(
                    f"validation-retry-{orchestrator.state.iteration}",
                    f"### Validation Retry\n\n**Action**: Fix and retry\n**Reason**: {args.reason}\n**Time**: {datetime.now().isoformat()}\n"
                )
            elif action == "revise_plan":
                orchestrator.state.phase = Phase.SYNTHESIZING
                orchestrator.state.iteration += 1
                orchestrator.append_section(
                    f"validation-revise-{orchestrator.state.iteration}",
                    f"### Plan Revision Required\n\n**Reason**: {args.reason}\n**Time**: {datetime.now().isoformat()}\n"
                )
            elif action == "override":
                orchestrator.state.phase = Phase.LESSONS_LEARNED
                orchestrator.append_section(
                    f"validation-override",
                    f"### Validation Override\n\n**Reason**: {args.reason}\n**Time**: {datetime.now().isoformat()}\n⚠️ Proceeding with known issues.\n"
                )
            elif action == "abandon":
                result = orchestrator.abandon_project(args.reason)
                print(json.dumps(result, indent=2, default=str))
                sys.exit(0)

            orchestrator.update_status_header()
            orchestrator.save_checkpoint()
            result = orchestrator.run_phase()
            print(json.dumps(result, indent=2, default=str))

    elif args.command == "lessons-learned":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        if orchestrator.state.phase != Phase.LESSONS_LEARNED:
            print(json.dumps({"status": "error", "message": "Must be in LESSONS_LEARNED phase"}, indent=2))
        else:
            responses = json.loads(args.responses)
            orchestrator.state.lessons_learned = responses

            # Format lessons for master doc
            lessons_md = f"""## Lessons Learned

**Captured**: {datetime.now().isoformat()}

### Outcome
{responses.get('outcome', 'Not specified')}

### Challenges
"""
            for challenge in responses.get('challenges', []):
                lessons_md += f"""
#### {challenge.get('symptom', 'Unknown')}
- **Root Cause**: {challenge.get('root_cause', 'Unknown')}
- **Solution**: {challenge.get('solution', 'Unknown')}
- **Key Insight**: {challenge.get('key_insight', 'None')}
"""

            lessons_md += "\n### Patterns Identified\n"
            for pattern in responses.get('patterns', []):
                lessons_md += f"""
#### {pattern.get('title', 'Unknown')}
- **Context**: {pattern.get('context', '')}
- **Domain**: {pattern.get('domain', 'patterns')}
"""

            lessons_md += f"\n### Workflow Feedback\n{responses.get('workflow_feedback', 'None provided')}\n"

            orchestrator.append_section("lessons-learned", lessons_md)
            orchestrator.save_checkpoint()
            result = orchestrator.run_phase()
            print(json.dumps(result, indent=2, default=str))

    elif args.command == "graduate":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        if orchestrator.state.phase != Phase.GRADUATING:
            print(json.dumps({"status": "error", "message": f"Must be in GRADUATING phase, currently in {orchestrator.state.phase.value}"}, indent=2))
        else:
            result = orchestrator.graduate_to_qdrant(dry_run=args.dry_run)
            print(json.dumps(result, indent=2, default=str))

    elif args.command == "abandon":
        orchestrator = CollaborationOrchestrator.load_checkpoint(Path(args.checkpoint))
        result = orchestrator.abandon_project(args.reason)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "list":
        # List all workflows
        workflows = []
        for checkpoint_file in STATE_DIR.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                workflows.append({
                    "workflow_id": data.get("workflow_id", "unknown"),
                    "project_name": data.get("project_name", "unknown"),
                    "phase": data.get("phase", "unknown"),
                    "updated_at": data.get("updated_at", "unknown"),
                    "checkpoint": str(checkpoint_file)
                })
            except Exception as e:
                workflows.append({
                    "workflow_id": checkpoint_file.stem,
                    "error": str(e),
                    "checkpoint": str(checkpoint_file)
                })

        # Sort by updated_at descending
        workflows.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        print(f"\n{'Workflow ID':<50} {'Phase':<20} {'Updated':<25}")
        print("-" * 95)
        for w in workflows:
            print(f"{w.get('project_name', w.get('workflow_id', 'unknown')):<50} {w.get('phase', 'error'):<20} {w.get('updated_at', '')[:19]:<25}")
        print(f"\nTotal: {len(workflows)} workflows")
        print(f"State dir: {STATE_DIR}")

    elif args.command == "cleanup":
        # Remove old completed/abandoned workflows
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=args.days)
        terminal_phases = ['completed', 'abandoned', 'rejected']

        to_remove = []
        for checkpoint_file in STATE_DIR.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                phase = data.get("phase", "")
                updated_at = data.get("updated_at", "")
                if phase in terminal_phases and updated_at:
                    updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00").split("+")[0])
                    if updated_dt < cutoff:
                        to_remove.append({
                            "file": checkpoint_file,
                            "workflow_id": data.get("workflow_id"),
                            "project_name": data.get("project_name"),
                            "phase": phase,
                            "updated_at": updated_at
                        })
            except Exception:
                pass

        if not to_remove:
            print(f"No workflows older than {args.days} days in terminal state (completed/abandoned/rejected)")
        else:
            print(f"Found {len(to_remove)} workflow(s) to remove:\n")
            for item in to_remove:
                print(f"  - {item['project_name']} ({item['phase']}, updated {item['updated_at'][:10]})")

            if args.dry_run:
                print(f"\n[DRY RUN] Would remove {len(to_remove)} state file(s) and their logs")
            else:
                for item in to_remove:
                    # Remove state file
                    item['file'].unlink()
                    # Remove corresponding log file
                    log_file = LOGS_DIR / f"{item['workflow_id']}.log"
                    if log_file.exists():
                        log_file.unlink()
                print(f"\nRemoved {len(to_remove)} workflow(s)")

    elif args.command == "archive":
        # Archive a workflow to archive/ subdirectory
        archive_dir = STATE_DIR / "archive"
        archive_dir.mkdir(exist_ok=True)

        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(json.dumps({"status": "error", "message": f"Checkpoint not found: {args.checkpoint}"}, indent=2))
        else:
            # Move state file
            archive_path = archive_dir / checkpoint_path.name
            checkpoint_path.rename(archive_path)

            # Move log file if exists
            with open(archive_path, 'r') as f:
                data = json.load(f)
            workflow_id = data.get("workflow_id", checkpoint_path.stem)
            log_file = LOGS_DIR / f"{workflow_id}.log"
            if log_file.exists():
                log_archive = archive_dir / f"{workflow_id}.log"
                log_file.rename(log_archive)

            print(json.dumps({
                "status": "archived",
                "workflow_id": workflow_id,
                "archive_path": str(archive_path)
            }, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
