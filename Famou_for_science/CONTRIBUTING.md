# Contributing

Contributions are welcome, but keep changes scoped and reproducible.

## Before opening a PR

- Read `README.md`
- Reproduce the example locally
- Keep paths repository-relative
- Do not add machine-specific files, caches, or local environment snapshots

## Contribution scope

Good candidates:

- bug fixes in evaluation or plotting scripts
- documentation improvements
- reproducibility improvements
- cleaner public-facing naming and structure

Avoid in this public snapshot:

- adding private services or credentials
- introducing absolute local paths
- mixing internal workflow assets into the public tree

## Pull request checklist

- The example still runs
- No absolute filesystem paths were introduced
- No secrets or private endpoints were added
- New files use consistent English naming where practical
