"""Writes around the Curves effect's curves (AE 2026 ground truth).

`curves.aep` holds four solids with a Curves effect each: `untouched`, and
`keyed`, whose curves are keyed at 0 s, 0.96 s and 2.04 s. The project's
stored definition of the effect mirrors the keyed instance.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from helpers import parse_project_fresh

from py_aep import parse as parse_aep
from py_aep.binary.chunk import Chunk, write_chunk
from py_aep.models.layers import Layer
from py_aep.models.project import Project
from py_aep.models.properties.curves import Curves
from py_aep.models.properties.property import Property

SAMPLE = (
    Path(__file__).parent.parent.parent
    / "samples"
    / "models"
    / "property"
    / "curves.aep"
)


def _layer(project: Project, name: str) -> Layer:
    return next(ly for ly in project.compositions[0].layers if ly.name == name)


def _curves(layer: Layer, index: int = 0) -> Property:
    prop = layer["ADBE Effect Parade"].properties[index]["ADBE CurvesCustom-0001"]
    assert isinstance(prop, Property)
    return prop


def _bytes(*chunks: Chunk | None) -> list[bytes]:
    out = []
    for chunk in chunks:
        assert chunk is not None
        buf = BytesIO()
        write_chunk(buf, chunk)
        out.append(buf.getvalue())
    return out


def _stream(prop: Property) -> list[bytes]:
    return _bytes(prop._tdsb, prop._tdb4, prop._cdat, prop._arbs)


def test_add_property_writes_an_untouched_curves(tmp_path: Path) -> None:
    # AE 2026's addProperty("ADBE CurvesCustom") writes an untouched
    # instance's bytes here, although the stored definition is keyed.
    project = parse_project_fresh(SAMPLE)
    _layer(project, "keyed")["ADBE Effect Parade"].add_property("ADBE CurvesCustom")
    out = tmp_path / "out.aep"
    project.save(out)

    reopened = parse_aep(out).project
    added = _curves(_layer(reopened, "keyed"), 1)
    untouched = _curves(_layer(reopened, "untouched"))

    assert not added.keyframes
    assert not added.is_modified
    curves = added.value
    assert isinstance(curves, Curves)
    assert curves.is_identity
    assert _stream(added) == _stream(untouched)


def test_moving_a_key_moves_its_curves(tmp_path: Path) -> None:
    # AE pairs a keyed Curves' aRbps with its keys by position: rendered in
    # AE 2026, the key moved last shows its own curves only when its aRbp
    # moved with it.
    project = parse_project_fresh(SAMPLE)
    prop = _curves(_layer(project, "keyed"))
    first = prop.keyframes[0].value
    assert isinstance(first, Curves)
    prop.keyframes[0].time = 3.0
    out = tmp_path / "out.aep"
    project.save(out)

    keys = _curves(_layer(parse_aep(out).project, "keyed")).keyframes
    moved = keys[-1].value
    assert keys[-1].time == 3.0
    assert isinstance(moved, Curves)
    assert moved.channels["rgb"].points == first.channels["rgb"].points
