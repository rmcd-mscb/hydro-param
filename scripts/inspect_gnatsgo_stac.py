#!/usr/bin/env python
"""Diagnostic script to inspect gNATSGO STAC items and their assets.

Usage:
    pixi run python scripts/inspect_gnatsgo_stac.py

Queries the Planetary Computer STAC for gnatsgo-rasters items covering
the DRB (Delaware River Basin) bbox and prints the available asset keys,
their types, and hrefs.
"""

from __future__ import annotations

import planetary_computer
import pystac_client

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "gnatsgo-rasters"

# DRB batch-1 bbox (from pipeline log)
BBOX = [-75.81, 39.60, -75.23, 41.02]


def main() -> None:
    client = pystac_client.Client.open(CATALOG_URL, modifier=planetary_computer.sign_inplace)

    search = client.search(collections=[COLLECTION], bbox=BBOX)
    items = list(search.item_collection())

    print(f"Found {len(items)} items for collection={COLLECTION} bbox={BBOX}\n")

    for item in items:
        print(f"--- Item: {item.id} ---")
        print(f"  Properties: gsd={item.properties.get('gsd')}, datetime={item.datetime}")
        print(f"  Bbox: {item.bbox}")
        print(f"  Asset keys: {sorted(item.assets.keys())}")
        for key, asset in sorted(item.assets.items()):
            media = asset.media_type or "unknown"
            roles = asset.roles or []
            href_short = asset.href[:120] + "..." if len(asset.href) > 120 else asset.href
            print(f"    {key:20s}  type={media:40s}  roles={roles}")
            print(f"    {'':20s}  href={href_short}")
        print()

    # Summary: which asset keys are available across all items?
    all_keys: set[str] = set()
    for item in items:
        all_keys.update(item.assets.keys())
    print(f"All asset keys across {len(items)} items: {sorted(all_keys)}")

    # Check if 'data' exists
    has_data = any("data" in item.assets for item in items)
    print(f"\n'data' asset present: {has_data}")

    # Check for the keys we expect
    expected = ["aws0_100", "rootznemc", "rootznaws", "mukey"]
    for key in expected:
        present = sum(1 for item in items if key in item.assets)
        print(f"  '{key}' present in {present}/{len(items)} items")


if __name__ == "__main__":
    main()
