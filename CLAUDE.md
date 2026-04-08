# CLAUDE.md

This file provides guidance for Claude Code when working on this project.

## Project Overview

Privacy-preserving federated learning research project.

## Running Experiments

- **All experiments must be reproducible via configs** — No hardcoded hyperparameters in code
- **All experiments run via `main.py`** — Single entry point: `python main.py --config <path>`
- **Notebooks are for exploration only** — Never run production experiments from notebooks
- **Seed everything** — Set random seeds (numpy, torch, python) in config for true reproducibility
- **Log metrics automatically** — Every run dumps metrics to structured format (JSON/CSV) without manual intervention
- **Checkpoint early, checkpoint often** — Save model state at intervals; FL training can be unstable
- **Version your data** — Track dataset versions/splits in config
- **Fail fast on config errors** — Validate configs at startup, not mid-experiment

## Code Quality

- **Monolithic architecture** — Keep related code together; avoid over-engineering into microservices
- **Simple code over bloated classes** — Prefer functions and simple data structures over complex class hierarchies
- **Flat is better than nested** — Avoid deep inheritance; prefer composition
- **Type hints on public interfaces** — Makes code self-documenting without docstring bloat
- **No silent failures** — Raise exceptions rather than returning None or logging warnings
- **Keep torch ops in one place** — Centralize `.to(device)` calls; don't scatter device management
- **Tests for math** — Unit test aggregation algorithms and privacy mechanisms; these are easy to get subtly wrong

## FL/Privacy-Specific

- **Separate client logic from server logic** — Even in simulation, keep the boundary clean
- **Make privacy parameters explicit** — Epsilon, delta, clipping norms must be in config, never hardcoded

## Project Structure

```
configs/          # Experiment configurations
notebooks/        # Exploration and visualization
src/              # Source code
tests/            # Unit tests
main.py           # Single entry point for experiments
```
