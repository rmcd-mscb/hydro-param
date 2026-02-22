---
name: issue-branch
description: Create a GitHub issue and feature branch in one step
---

# /issue-branch — Issue-First Workflow

Create a GitHub issue and corresponding feature branch following project conventions.

## Arguments

The user provides a short description of the work. For example:
- `/issue-branch add soils derivation step`
- `/issue-branch fix temporal chunking for SNODAS`

## Steps

1. **Determine the commit type** from the description: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, or `ci`.

2. **Create a GitHub issue** using `gh issue create`:
   - Title: `<type>: <description>`
   - Body: A brief summary of the planned work
   - Assign to the current user

3. **Extract the issue number** from the returned URL.

4. **Create and checkout a feature branch** named `<type>/<issue-number>-<short-description>`:
   - Use lowercase, hyphens for spaces
   - Keep the short description to 3-5 words max
   - Branch from `main` (pull latest first with `git pull origin main`)

5. **Report** the issue URL and branch name to the user.

## Example

User: `/issue-branch add soils derivation step`

Result:
- Issue: `feat: add soils derivation step` → `https://github.com/rmcd-mscb/hydro-param/issues/59`
- Branch: `feat/59-add-soils-derivation-step`

## Rules

- NEVER skip the issue creation — every code change starts with a GitHub issue
- NEVER branch from anything other than `main`
- ALWAYS use conventional commit type prefixes in the issue title
- ALWAYS pull latest main before branching
