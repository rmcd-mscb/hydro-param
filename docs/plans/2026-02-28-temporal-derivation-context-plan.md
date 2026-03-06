# Wire Temporal SIR Data into DerivationContext — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pass temporal SIR data through `DerivationContext` so steps 10 (PET), 11 (transpiration), and 7 (forcing) use climate-derived parameters instead of scalar defaults.

**Architecture:** Move temporal loading before `DerivationContext` construction in `pws_run_cmd`, pass it as `temporal=...`. Remove the redundant `merge_temporal_into_derived()` workaround since `_derive_forcing()` already handles forcing when `ctx.temporal` is populated.

**Tech Stack:** Python, xarray, pytest

---

### Task 1: Create issue and feature branch

**Step 1: Create feature branch**

```bash
git checkout -b feat/120-temporal-derivation-context main
```

**Step 2: Verify branch**

```bash
git branch --show-current
```

Expected: `feat/120-temporal-derivation-context`

---

### Task 2: Wire temporal data into DerivationContext in CLI

**Files:**
- Modify: `src/hydro_param/cli.py:625-767`

**Step 1: Update the import block**

In `src/hydro_param/cli.py`, change lines 627-630 from:

```python
    from hydro_param.derivations.pywatershed import (
        PywatershedDerivation,
        merge_temporal_into_derived,
    )
```

to:

```python
    from hydro_param.derivations.pywatershed import PywatershedDerivation
```

**Step 2: Load temporal data before DerivationContext**

Replace the block at lines 723-767 (from `# ── Derive parameters ──` through the temporal merge) with:

```python
    # ── Load temporal data from SIR ──
    temporal: dict[str, xr.Dataset] = {}
    for name in sir.available_temporal():
        try:
            temporal[name] = sir.load_temporal(name)
        except (OSError, KeyError) as exc:
            logger.error("Failed to load temporal SIR data '%s': %s", name, exc)
            logger.error(
                "Re-run 'hydro-param run pipeline.yml' to regenerate SIR output."
            )
            raise SystemExit(1) from exc

    if temporal:
        logger.info("Loaded %d temporal datasets: %s", len(temporal), list(temporal.keys()))
    else:
        logger.info("No temporal data in SIR; PET/transpiration will use defaults.")

    # ── Derive parameters ──
    logger.info("Deriving pywatershed parameters from SIR")

    derivation_config: dict = {}
    if pws_config.parameter_overrides.values:
        derivation_config["parameter_overrides"] = {
            "values": pws_config.parameter_overrides.values,
        }

    ctx = DerivationContext(
        sir=sir,
        fabric=fabric,
        segments=segments,
        waterbodies=waterbodies,
        fabric_id_field=pws_config.domain.id_field,
        segment_id_field=pws_config.domain.segment_id_field,
        config=derivation_config,
        temporal=temporal or None,
    )

    try:
        plugin = PywatershedDerivation()
        derived = plugin.derive(ctx)
    except Exception as exc:
        logger.exception("Parameter derivation failed.")
        raise SystemExit(1) from exc
    finally:
        for ds in temporal.values():
            ds.close()
```

Key changes:
- Temporal loading moved **before** `DerivationContext`
- `temporal=temporal or None` — pass `None` when empty (matches existing guards)
- `finally` block closes all temporal datasets after derivation
- The entire `merge_temporal_into_derived` block (old lines 749-767) is removed

**Step 3: Run tests**

```bash
pixi run -e dev pytest tests/test_cli.py -v -x 2>&1 | tail -20
```

**Step 4: Commit**

```bash
git add src/hydro_param/cli.py
git commit -m "feat: wire temporal SIR data into DerivationContext (#120)

Load temporal datasets from SIR before constructing DerivationContext
and pass them as temporal=... so steps 10 (PET), 11 (transpiration),
and 7 (forcing) compute climate-derived parameters instead of falling
back to scalar defaults."
```

---

### Task 3: Remove merge_temporal_into_derived function

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:167-249`

**Step 1: Delete `merge_temporal_into_derived` function**

Remove lines 167-249 (the entire `merge_temporal_into_derived` function) from `src/hydro_param/derivations/pywatershed.py`.

**Step 2: Remove the See Also reference in `_derive_forcing`**

Find the `See Also` reference to `merge_temporal_into_derived` (around line 2485) and remove it.

**Step 3: Run tests**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py -v -x 2>&1 | tail -20
```

Expected: `TestMergeTemporalIntoDerived` tests will fail (function removed). Other tests should pass.

**Step 4: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "refactor: remove merge_temporal_into_derived workaround (#120)

_derive_forcing() handles forcing generation when ctx.temporal is
populated. The standalone merge function was a workaround for when
the CLI didn't pass temporal data through the context."
```

---

### Task 4: Remove merge_temporal_into_derived tests, add temporal wiring tests

**Files:**
- Modify: `tests/test_pywatershed_derivation.py:1452-1581`
- Modify: `tests/test_cli.py` (add new test)

**Step 1: Delete `TestMergeTemporalIntoDerived` class**

Remove lines 1452-1580 from `tests/test_pywatershed_derivation.py` (the entire class and its section comment).

**Step 2: Add CLI test for temporal wiring**

Find the existing `pws_run_cmd` test area in `tests/test_cli.py` and add a test that verifies temporal data is passed to `DerivationContext`. The exact test structure depends on the existing test patterns in that file — match them. The test should mock `SIRAccessor.available_temporal()` to return temporal keys and verify `DerivationContext` is constructed with `temporal` populated.

**Step 3: Run all tests**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py tests/test_cli.py -v -x 2>&1 | tail -30
```

Expected: All pass.

**Step 4: Commit**

```bash
git add tests/test_pywatershed_derivation.py tests/test_cli.py
git commit -m "test: update tests for temporal DerivationContext wiring (#120)

Remove TestMergeTemporalIntoDerived (function deleted). Add test
verifying pws_run_cmd passes temporal data to DerivationContext."
```

---

### Task 5: Run full check suite and verify

**Step 1: Run full checks**

```bash
pixi run -e dev check
```

Expected: All pass (lint, format, typecheck, tests).

**Step 2: Run pre-commit**

```bash
pixi run -e dev pre-commit
```

Expected: All hooks pass.

**Step 3: Commit any fixups if needed**

If pre-commit hooks made formatting changes, commit them.
