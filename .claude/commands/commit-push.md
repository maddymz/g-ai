---
name: commit-push
description: Commit and push changes following repository branch rules
---

You are tasked with committing and pushing changes to the repository following the established branch protection rules.

## Workflow Steps

**CRITICAL: Always follow these steps in order:**

1. **Checkout main branch**
   - First, switch to the main branch
   - Command: `git checkout main`

2. **Pull latest changes**
   - Ensure main is up to date
   - Command: `git pull`

3. **Create feature branch**
   - Create and checkout a new feature branch with a descriptive name
   - Use kebab-case naming based on change type:
     - `add-<feature-name>` for new features
     - `fix-<bug-description>` for bug fixes
     - `update-<component-name>` for updates/modifications
     - `refactor-<area>` for refactoring
   - Command: `git checkout -b <branch-name>`

4. **Check git status**
   - Review what files have changed
   - Command: `git status`

5. **Stage changes**
   - Add files to staging area
   - Prefer adding specific files by name rather than `git add .` or `git add -A`
   - Never commit sensitive files (.env, credentials, API keys)
   - Command: `git add <specific-files>`

6. **Review changes**
   - Show diff and recent commit messages to understand context
   - Commands: `git diff --staged` and `git log --oneline -5`

7. **Create commit**
   - Draft a concise commit message following the repository's style
   - Include Co-Authored-By line
   - Command: Use heredoc format for multi-line messages
   ```bash
   git commit -m "$(cat <<'EOF'
   Commit message here

   Co-Authored-By: Claude Sonnet 4.5
   EOF
   )"
   ```

8. **Push feature branch to remote**
   - Push the feature branch with upstream tracking
   - Command: `git push -u origin <branch-name>`

9. **Create pull request**
   - Create a PR using the gh CLI tool
   - Use `gh pr create` with appropriate title and body
   - Include summary, changes, and test plan in PR description

## Pull Request Format

Always create a pull request using the gh CLI tool with the following format:

```bash
gh pr create --title "PR title here" --body "$(cat <<'EOF'
## Summary
- Bullet points describing the changes

## Changes
- **New/Modified files**: List with descriptions

## Test Plan
- [ ] Verification steps

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

PR Format Guidelines:
- Clear title summarizing the changes
- Summary section with bullet points
- Changes section listing modified/new files with descriptions
- Test plan showing verification steps as a checklist
- Include the Claude Code footer

## Important Notes

- ALWAYS create a feature branch - never commit directly to main
- NEVER force push unless explicitly requested
- NEVER commit .env files or credentials
- ALWAYS review changes before committing
- ALWAYS follow the commit message style from recent commits
- ALWAYS create a pull request - the repository enforces branch protection rules
- If uncertain about what to commit, ask the user first
