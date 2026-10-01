"""Tests for name containers as older projects write them (AE CC 12.0).

Newer projects wrap a `tdsn`, `fnam`, `pdnm` or `RCom` string in a `Utf8`
child; older ones write the NUL-terminated string itself, in a buffer of
its own size (a `tdsn` of a lone zero byte, a 48-byte `fnam`).
"""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from py_aep import Project, parse
from py_aep.binary.chunk import ContainerChunk, read_chunks, write_chunk
from py_aep.binary.item_chunks import NhedChunk
from py_aep.binary.misc_chunks import DwgaChunk
from py_aep.binary.property_chunks import TDSN_SENTINEL, TdsnChunk
from py_aep.binary.scalar_chunks import Utf8Chunk
from py_aep.enums import BitsPerChannel, GpuAccelType
from py_aep.parsers.project import _nnhd_from_nhed


def _chunk(chunk_type: str, body: bytes) -> bytes:
    pad = b"\x00" if len(body) & 1 else b""
    return chunk_type.encode("ASCII") + struct.pack(">I", len(body)) + body + pad


def _read(data: bytes) -> list:
    return read_chunks(BytesIO(data), len(data))


def _write(chunks: list) -> bytes:
    out = BytesIO()
    for chunk in chunks:
        write_chunk(out, chunk)
    return out.getvalue()


def test_a_one_byte_tdsn_does_not_swallow_the_next_chunk() -> None:
    data = _chunk("tdsn", b"\x00") + _chunk(
        "tdmn", b"ADBE Transform Group".ljust(40, b"\x00")
    )
    chunks = _read(data)
    assert [c.chunk_type for c in chunks] == ["tdsn", "tdmn"]
    tdsn = chunks[0]
    assert isinstance(tdsn, TdsnChunk)
    # An empty legacy name is an unnamed property.
    assert tdsn.utf8.value == TDSN_SENTINEL
    assert tdsn.utf8.synthetic
    assert _write(chunks) == data


def test_a_legacy_fnam_reads_its_string() -> None:
    body = b"Color Control\x00ol\x00\x00" + bytes(range(1, 32))
    body = body[:48]
    data = _chunk("fnam", body)
    (fnam,) = _read(data)
    assert isinstance(fnam, ContainerChunk)
    (utf8,) = fnam.chunks
    assert isinstance(utf8, Utf8Chunk)
    assert utf8.value == "Color Control"
    # Written back byte for byte, stale bytes included.
    assert _write([fnam]) == data


def test_a_renamed_legacy_fnam_keeps_its_buffer() -> None:
    body = b"Color Control".ljust(48, b"\x00")
    (fnam,) = _read(_chunk("fnam", body))
    fnam.chunks[0].value = "Text Color"
    written = _write([fnam])
    assert written == _chunk("fnam", b"Text Color".ljust(48, b"\x00"))


def test_a_modern_fnam_is_unchanged() -> None:
    utf8 = _chunk("Utf8", b"Glow")
    data = _chunk("fnam", utf8)
    (fnam,) = _read(data)
    (child,) = fnam.chunks
    assert child.value == "Glow"
    assert not child.synthetic
    assert fnam.data == b""
    assert _write([fnam]) == data


def test_nnhd_stands_in_from_nhed() -> None:
    nhed = NhedChunk()
    nhed.bits_per_channel = 1
    nhed.frames_count_type = 1
    nhed.timecode_default_base = 25
    nnhd = _nnhd_from_nhed(nhed)
    assert nnhd.synthetic
    assert (
        nnhd.bits_per_channel,
        nnhd.frames_count_type,
        nnhd.timecode_default_base,
    ) == (1, 1, 25)


def test_a_legacy_fnam_with_a_four_letter_name_reads_its_string() -> None:
    # "Fill" then NUL padding also reads as a `Fill` header of length 0; the
    # NUL headers after it are not chunks.
    for name in (b"Fill", b"Glow", b"Tint", b"Blur"):
        data = _chunk("fnam", name.ljust(48, b"\x00"))
        (fnam,) = _read(data)
        (utf8,) = fnam.chunks
        assert utf8.value == name.decode()
        assert utf8.synthetic
        assert _write([fnam]) == data


def test_a_renamed_one_byte_tdsn_grows_to_its_name() -> None:
    (tdsn,) = _read(_chunk("tdsn", b"\x00"))
    tdsn.utf8.value = "My Name"
    assert _write([tdsn]) == _chunk("tdsn", b"My Name\x00")


def test_a_renamed_legacy_tdsn_keeps_whole_characters() -> None:
    (tdsn,) = _read(_chunk("tdsn", b"abc\x00"))
    tdsn.utf8.value = "abcdef\u00e9"
    assert _write([tdsn]) == _chunk("tdsn", "abcdef\u00e9".encode() + b"\x00")


def test_a_legacy_tdsn_reset_to_unnamed_writes_the_empty_name() -> None:
    (tdsn,) = _read(_chunk("tdsn", b"Mask 1\x00"))
    tdsn.utf8.value = TDSN_SENTINEL
    assert _write([tdsn]) == _chunk("tdsn", b"\x00")


def test_a_renamed_legacy_pdnm_grows_to_its_items() -> None:
    (pdnm,) = _read(_chunk("pdnm", b"On|Off\x00"))
    pdnm.chunks[0].value = "On|Off|Auto"
    assert _write([pdnm]) == _chunk("pdnm", b"On|Off|Auto\x00")


def test_a_legacy_fnam_too_long_for_its_buffer_raises() -> None:
    (fnam,) = _read(_chunk("fnam", b"Glow".ljust(48, b"\x00")))
    fnam.chunks[0].value = "x" * 48
    with pytest.raises(ValueError, match="48-byte fnam"):
        _write([fnam])


SAMPLE = (
    Path(__file__).parent.parent.parent
    / "samples"
    / "versions"
    / "ae2026"
    / "complete.aep"
)


def _as_cc12_project() -> Project:
    """A project with the stand-ins a CC 12.0 one parses with: no nnhd or
    dwga, a head of AE 12."""
    project = parse(SAMPLE).project
    project._nnhd = _nnhd_from_nhed(project._nhed)
    project._dwga = DwgaChunk(synthetic=True)
    project._head = SimpleNamespace(ae_version_major=12)
    return project


def test_cc12_settings_synced_to_nhed_write_through() -> None:
    project = _as_cc12_project()
    project.bits_per_channel = BitsPerChannel.SIXTEEN
    assert project.bits_per_channel == BitsPerChannel.SIXTEEN
    assert project._nhed.bits_per_channel == project._nnhd.bits_per_channel
    assert project._nnhd.synthetic


def test_cc12_working_gamma_has_nowhere_to_go_and_raises() -> None:
    project = _as_cc12_project()
    with pytest.raises(AttributeError, match="requires AE 13"):
        project.working_gamma = 2.4


def test_cc12_timecode_default_base_synced_to_nhed() -> None:
    project = _as_cc12_project()
    project._timecode_default_base = 60
    assert project._nhed.timecode_default_base == 60


def test_cc12_timecode_default_base_above_u1_raises() -> None:
    project = _as_cc12_project()
    with pytest.raises(ValueError):
        project._timecode_default_base = 300
    assert project._nhed.timecode_default_base == 30


def test_gpu_accel_type_software_without_gpug() -> None:
    project = _as_cc12_project()
    project._gpug_utf8 = Utf8Chunk(
        value=GpuAccelType.to_binary(GpuAccelType.SOFTWARE), synthetic=True
    )
    assert project.gpu_accel_type == GpuAccelType.SOFTWARE
