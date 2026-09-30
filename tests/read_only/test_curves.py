"""Tests for the Curves effect's curves read from a project (AE 2026).

`curves.aep` holds one 1920x1080 comp with two solids, each with a Curves
effect: `pencil`, whose master curve was drawn in AE 2026 with the pencil,
and `untouched`, added and left as is.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import get_comp_from_json_by_name, load_expected, parse_project

from py_aep.models.properties.curves import Curves
from py_aep.models.properties.property import Property

SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples" / "models" / "property"


def _curves_property(layer_name: str) -> Property:
    project = parse_project(SAMPLES_DIR / "curves.aep")
    layer = next(ly for ly in project.compositions[0].layers if ly.name == layer_name)
    prop = layer["ADBE Effect Parade"][0]["ADBE CurvesCustom-0001"]
    assert isinstance(prop, Property)
    return prop


def _expected_is_modified(layer_name: str) -> bool:
    comp = get_comp_from_json_by_name(load_expected(SAMPLES_DIR, "curves"), "Comp 1")
    layer = next(ly for ly in comp["layers"] if ly["name"] == layer_name)
    parade = next(
        p for p in layer["properties"] if p["matchName"] == "ADBE Effect Parade"
    )
    effect = parade["properties"][0]
    prop = next(
        p for p in effect["properties"] if p["matchName"] == "ADBE CurvesCustom-0001"
    )
    return bool(prop["isModified"])


@pytest.mark.parametrize("layer_name", ["pencil", "untouched"])
def test_is_modified_matches_json(layer_name: str) -> None:
    prop = _curves_property(layer_name)
    assert prop.is_modified == _expected_is_modified(layer_name)


def test_untouched_curves_are_identity() -> None:
    curves = _curves_property("untouched").value
    assert isinstance(curves, Curves)
    assert curves.uses_points
    assert curves.is_identity


def test_pencil_curve_reads_its_map() -> None:
    curves = _curves_property("pencil").value
    assert isinstance(curves, Curves)
    assert not curves.uses_points
    assert [name for name, ch in curves.channels.items() if not ch.is_identity] == [
        "rgb"
    ]
    rgb = curves.channels["rgb"]
    # Pencil mode renders the map, read linearly between levels.
    for level in (0, 64, 128, 200, 255):
        assert curves.evaluate("rgb", level / 255.0) == pytest.approx(
            rgb.map[level] / 255.0
        )


def test_curves_are_read_only() -> None:
    prop = _curves_property("pencil")
    with pytest.raises(ValueError, match="read-only"):
        prop.value = prop.value
    with pytest.raises(ValueError, match="read-only"):
        prop.set_value_at_time(1.0, prop.value)
