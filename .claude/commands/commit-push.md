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

3. **Check git status**
   - Review what files have changed
   - Command: `git status`

4. **Stage changes**
   - Add files to staging area
   - Prefer adding specific files by name rather than `git add .` or `git add -A`
   - Never commit sensitive files (.env, credentials, API keys)
   - Command: `git add <specific-files>`

5. **Review changes**
   - Show diff and recent commit messages to understand context
   - Commands: `git diff --staged` and `git log --oneline -5`

6. **Create commit**
   - Draft a concise commit message following the repository's style
   - Include Co-Authored-By line
   - Command: Use heredoc format for multi-line messages
   ```bash
   git commit -m "$(cat <<'EOF'
   Commit message here

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
   EOF
   )"
   ```

7. **Push to remote**
   - Attempt to push to remote
   - Command: `git push`

8. **Handle branch protection**
   - If push is rejected due to branch protection rules (requires PR), create a pull request instead
   - Use `gh pr create` with appropriate title and body
   - Include summary, changes, and test plan in PR description

## Branch Protection Handling

If the push is rejected with "Changes must be made through a pull request":
- This means branch protection rules require a PR
- Create a PR using the gh CLI tool
- Format the PR with:
  - Clear title summarizing the changes
  - Summary section with bullet points
  - Changes section listing modified/new files
  - Test plan showing what was tested
  - Co-authored footer

## Important Notes

- NEVER skip the checkout main step
- NEVER force push unless explicitly requested
- NEVER commit .env files or credentials
- ALWAYS review changes before committing
- ALWAYS follow the commit message style from recent commits
- If uncertain about what to commit, ask the user first
