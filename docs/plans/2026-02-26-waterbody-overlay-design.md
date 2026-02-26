# Design: Derivation Step 6 — Waterbody Overlay

**Date:** 2026-02-26
**Issue:** TBD
**Status:** Approved

## Scope

Implement pywatershed derivation step 6: polygon-on-polygon overlay of NHDPlus
waterbody polygons against HRU polygons to derive depression storage parameters.

| Step | Name | Type | Dependencies |
|------|------|------|--------------|
| 6 | `waterbody_overlay` | GIS overlay | hru_area (step 1), NHDPlus waterbodies |

## Parameters Produced

| Parameter | Shape | Units | Description |
|-----------|-------|-------|-------------|
| `dprst_frac` | `(nhru,)` | decimal 0–1 | Fraction of HRU area covered by waterbodies |
| `dprst_area_max` | `(nhru,)` | acres | Total waterbody area clipped to each HRU |
| `hru_type` | `(nhru,)` | int32 | 1=land (default), 2=lake (>50% waterbody coverage) |

## Data Input

Waterbody polygons are provided via a new `waterbodies` field on
`DerivationContext`, following the same pattern as `fabric` and `segments`.

```python
@dataclass(frozen=True)
class DerivationContext:
    sir: xr.Dataset
    temporal: dict[str, xr.Dataset] | None = None
    fabric: gpd.GeoDataFrame | None = None
    segments: gpd.GeoDataFrame | None = None
    waterbodies: gpd.GeoDataFrame | None = None  # NEW
    ...
```

When `waterbodies is None`, step 6 assigns defaults and logs a warning.

## Feature Type Filter

Only `ftype` values `"LakePond"` and `"Reservoir"` are included. `SwampMarsh`
is excluded — it represents wetland vegetation, not open-water depression
storage in the PRMS sense.

## Algorithm

```
_derive_waterbody(self, ds: xr.Dataset, ctx: DerivationContext) -> xr.Dataset:

1. If ctx.waterbodies is None → log warning, assign defaults, return
2. Filter waterbodies to ftype in {"LakePond", "Reservoir"}
3. If no waterbodies remain after filter → log info, assign defaults, return
4. Ensure CRS match: reproject waterbodies to fabric CRS if needed
5. gpd.overlay(fabric, waterbodies, how="intersection")
6. Compute intersection polygon areas (native CRS units, m² for EPSG:5070)
7. Groupby fabric_id_field → sum areas per HRU
8. Convert m² → acres (÷ 4046.8564224)
9. Reuse hru_area from ds["hru_area"] (already in acres from step 1)
10. dprst_frac = clipped_area_acres / hru_area_acres, clipped to [0, 1]
11. dprst_area_max = clipped_area_acres
12. hru_type = np.where(dprst_frac > 0.5, 2, 1).astype(np.int32)
13. Assign all three to dataset, return
```

## HRU Type Classification

- Default: `hru_type = 1` (land)
- Set to `2` (lake) when `dprst_frac > 0.5` (waterbody covers more than 50%
  of HRU area)
- `0` (inactive) and `3` (swale) are not assigned by this step

## Fallback Behavior

| Condition | Action |
|-----------|--------|
| `ctx.waterbodies is None` | Warning log, defaults: frac=0, area=0, type=1 |
| No LakePond/Reservoir after filter | Info log, same defaults |
| Overlay produces no intersections | Same defaults (no waterbodies overlap any HRU) |
| HRU has no waterbody intersection | That HRU gets frac=0, area=0, type=1 |

## Pipeline Ordering

```
... → Step 5 (soils) → Step 6 (waterbody) → Step 7 (forcing) → ...
```

Step 6 depends on step 1 (`hru_area`) which runs earlier. No downstream step
depends on step 6 output — `dprst_depth_avg` is a constant default already
handled in step 13.

## File Changes

| Change | File | Notes |
|--------|------|-------|
| Modified | `src/hydro_param/plugins.py` | Add `waterbodies` field to `DerivationContext` |
| Modified | `src/hydro_param/derivations/pywatershed.py` | Add `_derive_waterbody()`, wire into `derive()` |
| Modified | `tests/test_pywatershed_derivation.py` | Tests for step 6 |

No new files, no new YAML configs.

## Testing Strategy

| Test | Description |
|------|-------------|
| Overlay correctness | Synthetic square HRU + known waterbody polygon → verify exact fraction and area |
| hru_type threshold | HRU with 60% coverage → type=2; HRU with 40% → type=1 |
| No waterbodies | `ctx.waterbodies=None` → defaults with warning |
| Empty after filter | Only SwampMarsh waterbodies → defaults |
| Partial overlap | Waterbody extends beyond HRU → only clipped area counted |
| Multiple waterbodies per HRU | Two waterbodies in one HRU → areas summed |
| CRS mismatch | Waterbodies in EPSG:4326, fabric in EPSG:5070 → auto-reproject |
| Integration | Full `derive()` with waterbodies → all three params in output |

## Out of Scope

- Steps 12 (routing) — separate design
- GRanD/NID supplementary waterbody sources
- Configurable ftype filter or coverage threshold
- Waterbody-specific depth parameters (dprst_depth_avg handled by step 13 defaults)
