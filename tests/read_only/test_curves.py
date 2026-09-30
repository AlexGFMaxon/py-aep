"""Tests for the Curves effect's curves read from a project (AE 2026).

`curves.aep` holds one 1920x1080 comp with four solids, each with a Curves
effect drawn in AE 2026:

- `pencil`: the master curve drawn with the pencil;
- `untouched`: added and left as is;
- `points`: points dragged on the master (5 points) and red (3) curves;
- `keyed`: the master curve keyed at 0 s, 0.96 s and 2.04 s (linear keys),
  a different point curve at each.
"""

from __future__ import annotations

import math
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


@pytest.mark.parametrize("layer_name", ["pencil", "untouched", "points", "keyed"])
def test_is_modified_matches_json(layer_name: str) -> None:
    prop = _curves_property(layer_name)
    assert prop.is_modified == _expected_is_modified(layer_name)


def test_untouched_curves_are_identity() -> None:
    curves = _curves_property("untouched").value
    assert isinstance(curves, Curves)
    assert curves.uses_points
    assert curves.is_identity


def test_pencil_curve_decodes_its_map() -> None:
    curves = _curves_property("pencil").value
    assert isinstance(curves, Curves)
    assert not curves.uses_points
    assert [name for name, ch in curves.channels.items() if not ch.is_identity] == [
        "rgb"
    ]
    rgb = curves.channels["rgb"]
    # The first levels of the master map as drawn in AE 2026.
    assert rgb.map[:12] == [1, 7, 12, 18, 24, 29, 35, 40, 45, 50, 55, 60]
    # Read linearly between levels: halfway from level 0 (1) to 1 (7).
    assert curves.evaluate("rgb", 0.5 / 255.0) == pytest.approx(4.0 / 255.0)


def test_untouched_curves_equal_the_parT_default() -> None:
    prop = _curves_property("untouched")
    assert prop._default_arbp is not None
    assert prop._arbp is not None
    assert prop._arbp.data == prop._default_arbp.data


def test_curves_are_read_only() -> None:
    prop = _curves_property("pencil")
    with pytest.raises(ValueError, match="read-only"):
        prop.value = prop.value
    with pytest.raises(ValueError, match="read-only"):
        prop.set_value_at_time(1.0, prop.value)


def _rounded_spline_is_the_map(curves: Curves) -> list[str]:
    """The edited channels whose map the rounded spline reproduces."""
    matching = []
    for name, channel in curves.channels.items():
        if channel.is_identity:
            continue
        spline = channel.spline()
        levels = [
            math.floor(min(max(spline(level), 0.0), 255.0) + 0.5)
            for level in range(256)
        ]
        if levels == channel.map:
            matching.append(name)
    return matching


def test_point_curves_reproduce_their_maps() -> None:
    curves = _curves_property("points").value
    assert isinstance(curves, Curves)
    assert curves.uses_points
    assert curves.channels["rgb"].points == [
        (0, 0),
        (90, 36),
        (107, 180),
        (178, 204),
        (255, 255),
    ]
    assert curves.channels["red"].points == [(0, 0), (131, 184), (255, 255)]
    # No half-level tie on these curves: every level is exact.
    assert _rounded_spline_is_the_map(curves) == ["rgb", "red"]


def _keyed_json() -> list[dict]:
    comp = get_comp_from_json_by_name(load_expected(SAMPLES_DIR, "curves"), "Comp 1")
    layer = next(ly for ly in comp["layers"] if ly["name"] == "keyed")
    parade = next(
        p for p in layer["properties"] if p["matchName"] == "ADBE Effect Parade"
    )
    prop = next(
        p
        for p in parade["properties"][0]["properties"]
        if p["matchName"] == "ADBE CurvesCustom-0001"
    )
    return list(prop["keyframes"])


def test_keyed_curves_decode_each_key() -> None:
    prop = _curves_property("keyed")
    keys_json = _keyed_json()
    assert [kf.time for kf in prop.keyframes] == pytest.approx(
        [k["time"] for k in keys_json]
    )
    middles = []
    for kf in prop.keyframes:
        curves = kf.value
        assert isinstance(curves, Curves)
        assert _rounded_spline_is_the_map(curves) == ["rgb"]
        middles.append(curves.channels["rgb"].points[1])
    assert middles == [(103, 164), (211, 166), (173, 89)]


def test_keyed_curves_at_a_time() -> None:
    prop = _curves_property("keyed")
    first, second, last = (kf.value for kf in prop.keyframes)
    assert isinstance(first, Curves)
    assert isinstance(second, Curves)
    assert isinstance(last, Curves)

    def points(time: float) -> object:
        value = prop.value_at_time(time)
        return None if value is None else value.channels["rgb"].points

    assert points(0.0) == first.channels["rgb"].points
    assert points(prop.keyframes[1].time) == second.channels["rgb"].points
    assert points(5.0) == last.channels["rgb"].points
    # Between two linear keys AE blends the curves, which cannot be read.
    assert points(0.5) is None


def test_keyed_curves_keys_are_read_only() -> None:
    prop = _curves_property("keyed")
    with pytest.raises(ValueError, match="read-only"):
        prop.remove_key(0)
    with pytest.raises(ValueError, match="read-only"):
        prop.remove_all_keys()
    with pytest.raises(ValueError, match="read-only"):
        prop.keyframes[0].value = prop.keyframes[1].value
    assert len(prop.keyframes) == len(prop._arbps) == 3
