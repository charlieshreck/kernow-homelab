# Commit Submodule Changes

Commit changes in a submodule and update the parent repository reference.

## Arguments
- `$ARGUMENTS` - Should contain: `<submodule_name> <commit_message>`
- Valid submodules: `agentic_lab`, `prod_homelab`, `monit_homelab`

## Workflow

1. **Parse arguments**: Extract submodule name and commit message from `$ARGUMENTS`
   - If no arguments provided, check all submodules for changes and ask which to commit

2. **Validate submodule**: Ensure the submodule exists at `/home/<submodule_name>`

3. **Check for changes**: Run `git status --porcelain` in the submodule
   - If no changes, report "No changes to commit" and exit

4. **Stage and commit in submodule**:
   ```bash
   cd /home/<submodule_name>
   git add -A
   git commit -m "<message>"
   ```
   Include the standard co-author trailer in the commit.

5. **Push submodule**: `git push origin main`

6. **Update parent reference**:
   ```bash
   cd /home
   git add <submodule_name>
   git commit -m "chore: update <submodule_name> submodule"
   git push origin main
   ```

7. **Report success**: Confirm the commit was made and pushed

## Example Usage
```
/commit-submodule agentic_lab "feat: add new MCP server"
/commit-submodule prod_homelab "fix: update ingress config"
/commit-submodule monit_homelab "chore: bump prometheus version"
```

## Notes
- This replaces the manual workflow of cd'ing into submodules
- Always pushes to `main` branch
- Automatically updates parent repo's submodule reference
- Use this instead of `/commit` when working in submodules
