# Design: Add Missing Topology/Identity Parameters (#158)

## Problem

The DRB reference parameter file contains four parameters that hydro-param
does not currently produce: `hru_lon`, `nhm_id`, `nhm_seg`, and
`hru_segment_nhm`. All four are derivable from the geospatial fabric — no
new datasets are needed.

## Design

### Parameter sources

| Output parameter   | Source                          | Configured by        |
|--------------------|---------------------------------|----------------------|
| `hru_lon`          | WGS84 centroid longitude        | fabric geometry      |
| `nhm_id`           | `fabric[id_field]`              | `id_field` in config |
| `nhm_seg`          | `segments[segment_id_field]`    | `segment_id_field`   |
| `hru_segment_nhm`  | `hru_segment` → segment ID join | both fields above    |

### Key decision: config-driven column names

The config declares `id_field` and `segment_id_field` to identify which
fabric columns hold the HRU and segment identifiers. The output parameter
names are always the pywatershed convention (`nhm_id`, `nhm_seg`) regardless
of the source column name. This means:

- `id_field: "nhru_v1_1"` → reads `fabric["nhru_v1_1"]`, writes param `nhm_id`
- `segment_id_field: "nhm_segment"` → reads `segments["nhm_segment"]`, writes param `nhm_seg`

### Changes by file

**`_derive_geometry()` — add `hru_lon`**
- WGS84 centroids are already computed for `hru_lat`. Add `.x.values` for longitude.
- SIR fallback mirrors `hru_lat`: `if "hru_lon" in sir`.

**`_derive_topology()` — add `nhm_id`, `nhm_seg`, `hru_segment_nhm`**
- `nhm_id`: emit `fabric[id_field]` as DataArray on `nhru` dim. Already validated
  to exist (used as coordinate). Always present when fabric is provided.
- `nhm_seg`: emit `segments[segment_id_field]` on `nsegment` dim. Already used
  for the `nsegment` coordinate — this writes it as a parameter too.
- `hru_segment_nhm`: for each HRU, map its `hru_segment` (1-based positional
  index into segments) to the corresponding `segment_id_field` value. Uses the
  same index→ID mapping already established for `nsegment` coords. HRUs with
  `hru_segment == 0` (no segment) get value 0.

All three topology params skip gracefully when fabric/segments are None
(existing guard at top of `_derive_topology`).

**`parameter_metadata.yml`**
- `hru_lon`: dimension nhru, units decimal_degrees, range [-180, 180]
- `nhm_id`: dimension nhru, type integer, optional true
- `nhm_seg`: dimension nsegment, type integer, optional true
- `hru_segment_nhm`: dimension nhru, type integer, optional true

**Tests**
- Geometry: verify `hru_lon` appears alongside `hru_lat`, values in valid range
- Topology: verify `nhm_id`, `nhm_seg`, `hru_segment_nhm` present and correct
- Graceful skip: verify absent columns don't cause errors
