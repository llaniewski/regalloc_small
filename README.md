# Register Allocation minimal example

This repo contains a simplification of the pattern of execution used in the [TCLB Solver](https://github/CFD-GO/TCLB).

Additionally it contains:
- an R script `get_stats.R` which pulls about register usage from the assembler (`.s`) file.
- configuration for `devcontainer`, which can be used with GitHub Codespaces<br/>[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/llaniewski/regalloc_small)

## Usage:
If you have HIP installed, just do:
```bash
make
```
