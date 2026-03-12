# AI-Assisted Development Workflow — Cheat Sheet

A practical guide for working with AI coding assistants (Claude Code, etc.)
on new projects. Based on lessons learned.

## Phase 1: Before You Start Coding

### Write a short design doc first (not 1200 lines)

The AI works dramatically better with context. But keep it focused:

```markdown
# Project Name — Design Brief

## Problem (2-3 sentences)
## Key decisions (bulleted list, <10 items)
## Constraints (what NOT to do)
## Dependencies (what libraries, why)
```

Save the deep analysis for when you've validated the core idea with code.

### Start with a code spike, not infrastructure

```
# GOOD order:
1. Write a 50-line prototype of the core logic
2. Get it working
3. Then add tests, CI, pre-commit

# BAD order:
1. Set up 5 CI jobs, 3 issue templates, branch protection
2. Still have zero application code
```

The spike tells you what you actually need. Infrastructure built on
assumptions will need to be revised.

## Phase 2: Setting Up a Project with AI

### Be directive, not open-ended

```
# Slower — AI will over-engineer:
"What do you suggest for development environment?"

# Faster — AI does exactly what you need:
"Set up pixi with a dev environment. Just ruff + pytest for now."
```

### Challenge every recommendation with "do I need this now?"

| AI suggests | Ask yourself |
|---|---|
| 5 pixi environments | Do I have more than 1 test file? |
| Conventional commits | Do I have collaborators or need changelogs? |
| Issue templates | Has anyone other than me filed an issue? |
| detect-secrets | Am I handling credentials yet? |
| Multiple CI matrix versions | Does my code use version-specific features? |

Add infrastructure when pain justifies it, not prophylactically.

### Minimum viable engineering setup

For a solo dev starting a new project, this is enough:

```
- ruff (lint + format)
- pytest
- 1 CI job (lint + test)
- Branch protection (require PR + CI pass)
- .gitignore
```

Add more when you need it:
- First collaborator → CONTRIBUTING.md, issue templates
- First secret scare → detect-secrets
- First "works on my machine" → lockfile, pixi/uv
- First release → conventional commits, changelog

## Phase 3: Working Session Patterns

### The issue-branch-PR cycle

```bash
# 1. Tell the AI what to build (be specific)
"Create issue #N, then implement X on a feature branch"

# 2. AI creates issue, branch, writes code
# 3. Run checks LOCALLY before committing
pixi run -e dev check    # or: ruff check && pytest

# 4. AI commits, pushes, creates PR
# 5. Review CI + Copilot feedback
# 6. Fix issues, squash-merge, clean up
```

### Always run checks locally before commit

This avoids CI round-trip delays:

```bash
pixi run -e dev check    # runs lint + format-check + typecheck + test
```

### How to give good prompts

```
# Include WHAT + WHY + CONSTRAINTS:
"Implement the config parser (src/hydro_param/config.py).
 It should validate YAML configs using pydantic.
 Keep it simple — just the top-level schema for now,
 not the full dataset definitions."

# Not:
"Let's work on the config system"
```

### When to plan vs. when to just do it

| Situation | Action |
|---|---|
| < 3 files, clear scope | Just do it |
| Architectural decision | Ask AI to propose options, you pick |
| > 5 files or unclear scope | Ask AI to plan first, review, then execute |
| Refactoring existing code | Always read first, plan second, execute third |

### Review AI output critically

The AI will:
- Add things you didn't ask for (extra error handling, docstrings, abstractions)
- Choose the "comprehensive" option over the simple one
- Not push back on premature complexity

Your job is to say: "That's more than I need. Just do X."

## Phase 4: Ongoing Development

### Keep the AI context-aware

- Maintain `CLAUDE.md` with current architectural decisions
- Update it when decisions change
- The AI reads this every session — keep it accurate and concise

### Cadence for adding infrastructure

| Milestone | Add |
|---|---|
| First working feature | Tests for that feature |
| 3+ modules | Type checking (mypy) |
| First collaborator | CONTRIBUTING.md, PR template |
| First deployment | CI/CD pipeline, branch protection |
| First release | Versioning, changelog |
| Recurring CI failures | Pre-commit hooks |

### Post-session checklist

- [ ] Is main clean? (`git status` on main)
- [ ] Did we delete the feature branch?
- [ ] Are all CI checks green?
- [ ] Does `CLAUDE.md` reflect any new decisions?
- [ ] Did we actually write application code, not just infrastructure?
