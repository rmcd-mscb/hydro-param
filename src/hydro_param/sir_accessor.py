"""Lazy accessor for SIR files backed by a pipeline manifest.

Provide per-variable on-demand loading from the SIR output directory.
No data is held in memory between calls -- each ``load_variable()`` reads
from disk and the caller is responsible for releasing.

Support a Dataset-compatible API (``__contains__``, ``__getitem__``,
``data_vars``) so derivation steps can check variable availability and
load data with minimal code changes.  ``__contains__`` checks both static
and temporal variables.

A valid manifest with a SIR section is required.  If no manifest is
found, ``SIRAccessor`` raises ``FileNotFoundError`` at construction time.

See Also
--------
hydro_param.manifest : Manifest schema that tracks SIR output.
hydro_param.plugins.DerivationContext : Consumer of SIRAccessor.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import xarray as xr

from hydro_param.manifest import SIRSchemaEntry, load_manifest

logger = logging.getLogger(__name__)


class SIRAccessor:
    """Lazy accessor for SIR files backed by a pipeline manifest.

    Do not hold data in memory.  Each call to ``load_variable()``
    reads from disk; the caller releases when done.

    Parameters
    ----------
    output_dir : Path
        Pipeline output directory containing ``.manifest.yml`` and
        ``sir/`` subdirectory.

    Raises
    ------
    FileNotFoundError
        If no valid manifest with a SIR section is found, or if a
        file referenced in the manifest does not exist on disk.

    Notes
    -----
    Implement ``__contains__`` and ``__getitem__`` for compatibility
    with derivation steps that check ``"var_name" in sir`` and access
    ``sir["var_name"]``.  ``__contains__`` checks both static and
    temporal variables.

    Examples
    --------
    >>> sir = SIRAccessor(Path("output"))
    >>> "elevation_m_mean" in sir
    True
    >>> elev = sir["elevation_m_mean"]
    >>> sir.available_temporal()
    ['pr_mm_mean_2020', 'tmmx_C_mean_2020']
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._sir_schema: list[SIRSchemaEntry] = []

        manifest = load_manifest(self._output_dir)
        if manifest is None or manifest.sir is None:
            raise FileNotFoundError(
                f"No valid manifest with SIR section at {self._output_dir}. "
                f"Run 'hydro-param run pipeline.yml' to produce SIR output "
                f"before running Phase 2."
            )
        self._static = dict(manifest.sir.static_files)
        self._temporal = dict(manifest.sir.temporal_files)
        self._sir_schema = list(manifest.sir.sir_schema)

        self._canonical_to_prefixed = _build_canonical_index(self._static)
        self._temporal_canonical_to_prefixed = _build_canonical_index(self._temporal)
        self._validate_files()

    def _validate_files(self) -> None:
        """Verify all referenced files exist on disk.

        Check all static and temporal file references and report all
        missing files in a single error, not just the first.  This is
        performed eagerly at construction time so that all
        ``FileNotFoundError`` exceptions surface at instantiation, not
        during later ``load_variable()`` calls.

        Raises
        ------
        FileNotFoundError
            If any referenced file is missing.
        """
        missing: list[str] = []
        for name, rel_path in {**self._static, **self._temporal}.items():
            full = self._output_dir / rel_path
            if not full.exists():
                missing.append(f"  '{name}' -> {full}")
        if missing:
            raise FileNotFoundError(
                "Missing SIR files:\n"
                + "\n".join(missing)
                + "\nRe-run 'hydro-param run pipeline.yml' to regenerate."
            )

    def available_variables(self) -> list[str]:
        """List all static SIR variable names.

        Returns
        -------
        list[str]
            Variable names available for ``load_variable()``.
        """
        return list(self._static.keys())

    def available_temporal(self) -> list[str]:
        """List all temporal SIR dataset keys (canonical names).

        Strip dataset prefixes (``"dataset__variable"`` → ``"variable"``)
        so consumer code can match against canonical SIR variable names.

        Returns
        -------
        list[str]
            Canonical temporal dataset keys available for
            ``load_temporal()``.
        """
        return [_parse_canonical_name(k) for k in self._temporal]

    @property
    def data_vars(self) -> list[str]:
        """Return canonical static variable names as a convenience list.

        Strip dataset prefixes (``"dataset__variable"`` → ``"variable"``)
        so derivation code can use ``startswith()`` patterns with
        canonical names.  Use ``available_variables()`` to get the raw
        (possibly prefixed) keys.

        Unlike ``xr.Dataset.data_vars``, this returns a plain
        ``list[str]``, not a mapping.

        Returns
        -------
        list[str]
            Canonical variable names (prefixes stripped).
        """
        return [_parse_canonical_name(k) for k in self._static]

    @property
    def sir_schema(self) -> list[SIRSchemaEntry]:
        """Return a copy of SIR variable schema metadata.

        Returns
        -------
        list[SIRSchemaEntry]
            Schema entries from the manifest.  Returns a copy to
            prevent external mutation of internal state.
        """
        return list(self._sir_schema)

    def load_variable(self, name: str) -> xr.DataArray:
        """Load a single SIR variable from disk.

        Parameters
        ----------
        name : str
            SIR variable name (e.g., ``"elevation_m_mean"``).

        Returns
        -------
        xr.DataArray
            Variable data with the CSV index column as dimension.
            By convention, SIR CSV files use the fabric id field as
            the index column (produced by stage 5).

        Raises
        ------
        KeyError
            If the variable name is not in the SIR.
        OSError
            If the CSV file is corrupt, truncated, or unreadable.

        Notes
        -----
        For single-column CSVs where the column name differs from the
        registry key (e.g., a renamed variable), the sole data variable
        is returned regardless of its actual name.
        """
        resolved = self._resolve_static(name)
        if resolved is None:
            raise KeyError(
                f"SIR variable '{name}' not found. Available: {sorted(self._static.keys())}"
            )
        path = self._output_dir / self._static[resolved]
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception as exc:
            raise OSError(
                f"Failed to read SIR file for variable '{name}' at {path}: {exc}. "
                f"The file may be corrupt or truncated. "
                f"Re-run 'hydro-param run pipeline.yml' to regenerate."
            ) from exc
        ds = xr.Dataset.from_dataframe(df)
        if name in ds:
            return ds[name]
        # Try the canonical part of the resolved key
        canonical = _parse_canonical_name(resolved)
        if canonical and canonical in ds:
            return ds[canonical]
        if not ds.data_vars:
            raise ValueError(
                f"SIR file for '{name}' at {path} contains no data columns. "
                f"Re-run 'hydro-param run pipeline.yml' to regenerate."
            )
        fallback = next(iter(ds.data_vars.values()))
        if len(ds.data_vars) > 1:
            logger.warning(
                "SIR variable '%s' not found as a column in %s. "
                "File contains %d columns: %s. Returning first column '%s'. "
                "This may indicate a column naming mismatch.",
                name,
                path.name,
                len(ds.data_vars),
                sorted(str(v) for v in ds.data_vars),
                fallback.name,
            )
        return fallback

    def load_temporal(self, name: str) -> xr.Dataset:
        """Load a single temporal SIR file from disk.

        Parameters
        ----------
        name : str
            Temporal dataset key (e.g., ``"gridmet_2020"``).

        Returns
        -------
        xr.Dataset
            Temporal dataset.  Caller should close when done.

        Raises
        ------
        KeyError
            If the temporal key is not in the SIR.
        OSError
            If the NetCDF file is corrupt, truncated, or unreadable.
        """
        resolved = self._resolve_temporal(name)
        if resolved is None:
            raise KeyError(
                f"SIR temporal dataset '{name}' not found. "
                f"Available: {sorted(self._temporal.keys())}"
            )
        path = self._output_dir / self._temporal[resolved]
        try:
            return xr.open_dataset(path)
        except Exception as exc:
            raise OSError(
                f"Failed to read SIR temporal file for '{name}' at {path}: {exc}. "
                f"The file may be corrupt. "
                f"Re-run 'hydro-param run pipeline.yml' to regenerate."
            ) from exc

    def _resolve_static(self, name: str) -> str | None:
        """Resolve a variable name to a key in ``_static``.

        Return the name as-is if it exists as a prefixed key.
        Otherwise check the canonical-to-prefixed index for a match.
        Return ``None`` if no match is found.
        """
        if name in self._static:
            return name
        return self._canonical_to_prefixed.get(name)

    def _resolve_temporal(self, name: str) -> str | None:
        """Resolve a temporal key to a key in ``_temporal``.

        Return the name as-is if it exists as a prefixed key.
        Otherwise check the canonical-to-prefixed index for a match.
        Return ``None`` if no match is found.
        """
        if name in self._temporal:
            return name
        return self._temporal_canonical_to_prefixed.get(name)

    def source_for(self, name: str) -> str | None:
        """Return the source dataset name for a SIR variable.

        Parse the dataset prefix from the prefixed key
        (``"dataset__variable"``).  Return ``None`` if the variable
        is not found or has no prefix.

        Parameters
        ----------
        name : str
            SIR variable name (canonical or prefixed).

        Returns
        -------
        str or None
            Source dataset name, or ``None`` if not found or unprefixed.
        """
        resolved = self._resolve_static(name)
        if resolved is None:
            resolved = self._resolve_temporal(name)
        if resolved is None:
            return None
        return _parse_dataset_prefix(resolved)

    def __contains__(self, name: object) -> bool:
        """Check if a variable name is available (Dataset-compatible API).

        Check both static and temporal variable maps, including
        canonical (unprefixed) lookups via the canonical-to-prefixed
        index.  Use ``available_variables()`` or ``available_temporal()``
        to query each map separately.

        Notes
        -----
        ``__contains__`` checks both static and temporal keys, but
        ``__getitem__`` only loads static variables.  Use
        ``load_temporal()`` explicitly for temporal datasets.
        """
        if not isinstance(name, str):
            return False
        return self._resolve_static(name) is not None or self._resolve_temporal(name) is not None

    def __getitem__(self, name: str) -> xr.DataArray:
        """Load a static variable by name (Dataset-compatible API).

        Equivalent to ``load_variable(name)``.  For temporal datasets,
        use ``load_temporal()`` instead.
        """
        return self.load_variable(name)

    def load_dataset(self, name: str) -> xr.Dataset:
        """Load a static SIR file as a full Dataset (all columns).

        Unlike ``load_variable()`` which returns a single DataArray,
        this method returns the complete ``xr.Dataset`` with all columns
        from a multi-column CSV.  Useful for categorical fraction files
        (e.g., ``lndcov_frac_2021``) where each column represents a
        class fraction.

        Parameters
        ----------
        name : str
            SIR variable name (must match a static file key).

        Returns
        -------
        xr.Dataset
            Full dataset with all columns from the CSV.

        Raises
        ------
        KeyError
            If the variable name is not in the SIR.
        OSError
            If the CSV file is corrupt, truncated, or unreadable.
        """
        resolved = self._resolve_static(name)
        if resolved is None:
            raise KeyError(
                f"SIR variable '{name}' not found. Available: {sorted(self._static.keys())}"
            )
        path = self._output_dir / self._static[resolved]
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception as exc:
            raise OSError(
                f"Failed to read SIR file for '{name}' at {path}: {exc}. "
                f"The file may be corrupt or truncated. "
                f"Re-run 'hydro-param run pipeline.yml' to regenerate."
            ) from exc
        return xr.Dataset.from_dataframe(df)

    def find_variable(self, base_name: str) -> str | None:
        """Find a static variable by base name with fuzzy resolution.

        Resolution order:

        1. **Exact match** — ``base_name`` exists as a prefixed key or
           canonical name in the index.
        2. **Year-suffix match** — canonical name matches
           ``{base_name}_{YYYY}``.  Returns the most recent year when
           multiple years exist.
        3. **Prefix match** — canonical name starts with
           ``{base_name}_``.  This handles SIR names that include units
           and statistic suffixes (e.g., ``"canopy_pct"`` matches
           ``"canopy_pct_pct_mean"``).  Returns the unique match, or
           ``None`` with a warning if ambiguous.

        The returned key is always the actual (prefixed) key stored in
        ``_static``, so callers can pass it directly to ``load_variable()``.

        Parameters
        ----------
        base_name : str
            Variable base name (e.g., ``"fctimp_pct_mean"``,
            ``"canopy_pct"``).

        Returns
        -------
        str or None
            The actual SIR variable name (prefixed key), or ``None``
            if not found.

        Notes
        -----
        Steps are evaluated in strict order; a match at an earlier step
        short-circuits later ones.  In particular, year-suffixed names
        like ``canopy_pct_2021`` are always resolved by step 2 and never
        reach the prefix-match step, avoiding false positives.
        """
        # Exact match (prefixed key or canonical via index)
        resolved = self._resolve_static(base_name)
        if resolved is not None:
            return resolved
        # Year-suffix search across canonical parts
        pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{4}})$")
        matches: list[str] = []
        for prefixed_key in self._static:
            canonical = _parse_canonical_name(prefixed_key)
            if pattern.match(canonical):
                matches.append(prefixed_key)
        if matches:

            def _year_key(prefixed: str) -> int:
                m = pattern.search(_parse_canonical_name(prefixed))
                return int(m.group(1)) if m else 0

            resolved = max(matches, key=_year_key)
            logger.debug(
                "Resolved '%s' to year-suffixed variant '%s' (%d candidate(s))",
                base_name,
                resolved,
                len(matches),
            )
            return resolved

        # Prefix match: base_name is a prefix of the canonical name.
        # Handles SIR names with units/statistic suffixes, e.g.
        # "canopy_pct" → "canopy_pct_pct_mean".
        prefix = base_name + "_"
        prefix_matches: list[str] = []
        for prefixed_key in self._static:
            canonical = _parse_canonical_name(prefixed_key)
            if canonical.startswith(prefix):
                prefix_matches.append(prefixed_key)
        if len(prefix_matches) == 1:
            resolved = prefix_matches[0]
            logger.debug(
                "Resolved '%s' to prefix-matched variant '%s'",
                base_name,
                resolved,
            )
            return resolved
        if len(prefix_matches) > 1:
            logger.warning(
                "Ambiguous prefix match for '%s': %s; returning None. "
                "Use a more specific name to disambiguate.",
                base_name,
                sorted(_parse_canonical_name(k) for k in prefix_matches),
            )
        return None


def _parse_dataset_prefix(name: str) -> str | None:
    """Extract the dataset prefix from a prefixed SIR key.

    Parameters
    ----------
    name : str
        SIR key, possibly prefixed (e.g., ``"dem_3dep_10m__elevation_m_mean"``).

    Returns
    -------
    str or None
        Dataset name before ``__``, or ``None`` if no prefix.
    """
    if "__" in name:
        return name.split("__", 1)[0]
    return None


def _parse_canonical_name(name: str) -> str:
    """Extract the canonical variable name from a possibly prefixed SIR key.

    Parameters
    ----------
    name : str
        SIR key, possibly prefixed (e.g., ``"dem_3dep_10m__elevation_m_mean"``).

    Returns
    -------
    str
        The canonical part after ``__``, or the full name if no prefix.
    """
    if "__" in name:
        return name.split("__", 1)[1]
    return name


def _build_canonical_index(mapping: dict[str, str]) -> dict[str, str]:
    """Build a canonical-name-to-prefixed-key index.

    Map canonical (unprefixed) names to their prefixed keys for
    unprefixed lookups.  If two prefixed keys share the same
    canonical name, the last one wins (alphabetically) and a warning
    is logged.

    Parameters
    ----------
    mapping : dict[str, str]
        The ``_static`` or ``_temporal`` dict (prefixed key → rel path).

    Returns
    -------
    dict[str, str]
        Canonical name → prefixed key.
    """
    index: dict[str, str] = {}
    for prefixed_key in sorted(mapping.keys()):
        canonical = _parse_canonical_name(prefixed_key)
        if canonical != prefixed_key:
            if canonical in index:
                logger.warning(
                    "Canonical name '%s' maps to multiple prefixed keys: "
                    "'%s' and '%s'; using '%s'. Use prefixed names for "
                    "explicit access.",
                    canonical,
                    index[canonical],
                    prefixed_key,
                    prefixed_key,
                )
            index[canonical] = prefixed_key
    return index
