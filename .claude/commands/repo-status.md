# Repository Status

Check git status across all Kernow homelab repositories.

## Workflow

1. **Check parent repo status**:
   ```bash
   cd /home && git status --short
   ```

2. **Check each submodule**:
   ```bash
   cd /home/agentic_lab && git status --short
   cd /home/prod_homelab && git status --short
   cd /home/monit_homelab && git status --short
   ```

3. **Check for unpushed commits** in each repo:
   ```bash
   git log origin/main..HEAD --oneline
   ```

4. **Summarize findings**:
   - List any uncommitted changes per repo
   - List any unpushed commits per repo
   - Highlight if submodule references are out of sync

## Output Format

```
## Repository Status

### /home (kernow-homelab)
- Status: Clean / X files modified
- Unpushed: None / X commits

### /home/agentic_lab
- Status: Clean / X files modified
- Unpushed: None / X commits

### /home/prod_homelab
- Status: Clean / X files modified
- Unpushed: None / X commits

### /home/monit_homelab
- Status: Clean / X files modified
- Unpushed: None / X commits
```

## Notes
- Use this before starting work to check current state
- Use this after work to verify all changes are committed/pushed
- Replaces running `git status` in each directory manually
