"""Tests for the Curves effect's value (ADBE CurvesCustom-0001)."""

from __future__ import annotations

import math
import struct

import pytest

from py_aep.binary.chunk import Chunk
from py_aep.binary.property_chunks import CurvesArbpChunk
from py_aep.cli.validate import to_dict
from py_aep.models.properties.curves import CURVES_CHANNELS, Curves


def _blob(
    points: dict[str, list[tuple[int, int]]],
    mode: int = 1,
    stale: bool = False,
    selected: int = -1,
) -> bytes:
    """A Curves aRbp as After Effects writes it, the maps left identity."""
    out = bytearray(4 + 5 * 256 + 5 * 72)
    struct.pack_into(">HH", out, 0, 1, mode)
    for c, name in enumerate(CURVES_CHANNELS):
        pts = points.get(name, [(0, 0), (255, 255)])
        out[4 + 256 * c : 4 + 256 * (c + 1)] = bytes(range(256))
        offset = 4 + 5 * 256 + 72 * c
        for i, (x, y) in enumerate(pts):
            struct.pack_into(">hh", out, offset + 4 * i, x, y)
        if stale:
            # After Effects leaves old points past the count.
            struct.pack_into(">hh", out, offset + 4 * len(pts), 5440, 8193)
        struct.pack_into(">Ii", out, offset + 64, len(pts), selected)
    return bytes(out)


def _curves(data: bytes) -> Curves:
    arbp = Chunk(chunk_type="aRbp", data=data)
    return Curves._from_binary(CurvesArbpChunk.from_arbp(arbp))


def test_decodes_points_in_channel_order() -> None:
    curves = _curves(_blob({"rgb": [(0, 0), (168, 93), (255, 246)]}, stale=True))
    assert curves.uses_points
    assert curves.channels["rgb"].points == [(0, 0), (168, 93), (255, 246)]
    assert curves.channels["red"].points == [(0, 0), (255, 255)]
    assert curves.channels["red"].is_identity
    assert not curves.channels["rgb"].is_identity
    assert not curves.is_identity


def test_untouched_curves_are_identity() -> None:
    assert _curves(_blob({})).is_identity


def test_natural_spline_goes_through_the_points() -> None:
    points = [(0, 0), (41, 30), (129, 149), (191, 211), (255, 255)]
    curves = _curves(_blob({"rgb": points}))
    for x, y in curves.channels["rgb"].points:
        assert math.isclose(curves.evaluate("rgb", x / 255.0), y / 255.0, abs_tol=1e-9)


def test_two_points_are_a_straight_line() -> None:
    curves = _curves(_blob({"green": [(0, 51), (255, 204)]}))
    assert math.isclose(curves.evaluate("green", 0.5), 0.5, abs_tol=1e-9)


def test_duplicate_inputs_keep_the_last_output() -> None:
    curves = _curves(_blob({"rgb": [(0, 0), (128, 64), (128, 192), (255, 255)]}))
    assert math.isclose(curves.evaluate("rgb", 128 / 255.0), 192 / 255.0)


def test_overshoot_is_clamped_unless_asked() -> None:
    # Grayscale 4: the spline dips below 0 near black.
    curves = _curves(_blob({"rgb": [(0, 0), (62, 18), (193, 238), (255, 255)]}))
    assert curves.evaluate("rgb", 0.05) == 0.0
    assert curves.evaluate("rgb", 0.05, clamp=False) < 0.0


def test_pencil_mode_reads_the_maps() -> None:
    blob = bytearray(_blob({}, mode=0))
    blob[4 : 4 + 256] = bytes(255 - i for i in range(256))
    curves = _curves(bytes(blob))
    assert not curves.uses_points
    assert math.isclose(curves.evaluate("rgb", 0.0), 1.0)
    assert math.isclose(curves.evaluate("rgb", 1.0), 0.0)


def test_to_dict_emits_the_curves() -> None:
    data = to_dict(_curves(_blob({"rgb": [(0, 0), (168, 93), (255, 246)]})))
    assert data["mode"] == 1
    assert data["version"] == 1
    assert data["is_identity"] is False
    rgb = data["channels"]["rgb"]
    assert rgb["points"] == [(0, 0), (168, 93), (255, 246)]
    assert rgb["map"] == list(range(256))
    assert rgb["selected"] == -1


def test_junk_selected_point_reads_as_none() -> None:
    # AE leaves junk here at times: 0x9200FF with 3 points.
    curves = _curves(_blob({"rgb": [(0, 0), (128, 90), (255, 255)]}, selected=0x9200FF))
    assert curves.channels["rgb"].selected == -1
    curves = _curves(_blob({"rgb": [(0, 0), (128, 90), (255, 255)]}, selected=1))
    assert curves.channels["rgb"].selected == 1


def test_pencil_identity_map_with_stale_points_is_identity() -> None:
    # Drawn with the pencil, only the map counts; the points are stale.
    curves = _curves(_blob({"rgb": [(0, 0), (128, 40), (255, 255)]}, mode=0))
    assert curves.channels["rgb"].is_identity
    assert curves.is_identity


def test_same_curves_compares_points_not_bytes() -> None:
    edited = {"rgb": [(0, 0), (128, 90), (255, 255)]}
    stale = _curves(_blob(edited, stale=True))
    assert stale._same_curves(_curves(_blob(edited)))
    assert not stale._same_curves(_curves(_blob({})))


@pytest.mark.parametrize(
    ("channel", "x"), [("luma", 0.5), ("rgb", float("nan")), ("rgb", float("inf"))]
)
def test_evaluate_rejects_bad_input(channel: str, x: float) -> None:
    with pytest.raises(ValueError):
        _curves(_blob({})).evaluate(channel, x)


def test_rejects_other_sizes() -> None:
    with pytest.raises(ValueError):
        CurvesArbpChunk.from_arbp(Chunk(chunk_type="aRbp", data=b"\x00" * 16))
