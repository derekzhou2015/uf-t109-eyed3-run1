"""Regression tests: a single corrupt frame must not abort parsing of the
whole tag/file.

Corrupt frame payloads surface as many different exception types
(zlib.error, UnicodeDecodeError, TypeError, ValueError, IndexError, ...).
These are normalized to ``eyed3.id3.frames.FrameException`` at the frame
factory (``createFrame``), which ``FrameSet.parse`` already handles by
logging and skipping the offending frame -- the library's long-standing
leniency towards bad frames.
"""
import zlib

import pytest

import eyed3
from eyed3.id3 import ID3_V2_3, ID3_V2_4
from eyed3.id3.tag import Tag
from eyed3.id3.frames import FrameException, createFrame
from eyed3.id3.headers import FrameHeader, TagHeader
from eyed3.utils.binfuncs import bytes2bin


def _syncsafe(n):
    return bytes([(n >> 21) & 0x7f, (n >> 14) & 0x7f, (n >> 7) & 0x7f,
                  n & 0x7f])


def _v23_frame(fid, data, flags=b"\x00\x00"):
    return fid + len(data).to_bytes(4, "big") + flags + data


def _v24_frame(fid, data, flags=b"\x00\x00"):
    return fid + _syncsafe(len(data)) + flags + data


def _tag(version, frames_data):
    if version == ID3_V2_3:
        hdr = b"ID3" + bytes([3, 0, 0])
    else:
        hdr = b"ID3" + bytes([4, 0, 0])
    return hdr + _syncsafe(len(frames_data)) + frames_data


def _text_frame_data(text):
    # latin1 encoding byte + latin1 text
    return b"\x00" + text.encode("latin1")


# A healthy frame placed before and after each corrupt one.
GOOD_FRAME_1 = _v23_frame(b"TIT2", _text_frame_data("Healthy Title"))
GOOD_FRAME_2 = _v23_frame(b"TPE1", _text_frame_data("Healthy Artist"))
GOOD_FRAME_1_V24 = _v24_frame(b"TIT2", _text_frame_data("Healthy Title"))
GOOD_FRAME_2_V24 = _v24_frame(b"TPE1", _text_frame_data("Healthy Artist"))

# v2.3 frame with the compression flag set but a payload that is not zlib
# data -> zlib.error during decompression.
CORRUPT_ZLIB_V23 = _v23_frame(b"TIT2", b"\x00\x00\x00\x20" + b"not-zlib-data",
                              flags=b"\x00\x80")
# TXXX with declared utf-16 encoding but truncated (odd-length) payload
# -> UnicodeDecodeError.
CORRUPT_UTF16_TXXX_V23 = _v23_frame(b"TXXX", b"\x01\xff")
# APIC truncated right after the encoding byte -> TypeError (ord of b"").
CORRUPT_APIC_V23 = _v23_frame(b"APIC", b"\x00")
# RVA2 without the required \x00 separator -> ValueError (unpack).
CORRUPT_RVA2_V24 = _v24_frame(b"RVA2", b"no-null-separator")
# RVAD too short to hold the bytes-per-volume byte -> IndexError.
CORRUPT_RVAD_V23 = _v23_frame(b"RVAD", b"\x10")

V23_CORRUPT = [CORRUPT_ZLIB_V23, CORRUPT_UTF16_TXXX_V23, CORRUPT_APIC_V23,
               CORRUPT_RVAD_V23]
V24_CORRUPT = [CORRUPT_RVA2_V24]

# Minimal but valid MPEG1 Layer III audio (128kbps, 44100Hz): 3 frames.
_MP3_FRAME = b"\xff\xfb\x90\x00" + b"\x00" * 413
MP3_AUDIO = _MP3_FRAME * 3


def _parse_tag_file(tmp_path, tag_bytes, name="test.id3"):
    tag_file = tmp_path / name
    tag_file.write_bytes(tag_bytes)
    tag = Tag()
    found = tag.parse(str(tag_file))
    return found, tag


@pytest.mark.parametrize("fid,version,data,flags,cause_type", [
    # compressed frame with non-zlib payload
    (b"TIT2", ID3_V2_3, b"\x00\x00\x00\x20" + b"not-zlib-data", b"\x00\x80",
     zlib.error),
    # utf-16 declared, truncated payload
    (b"TXXX", ID3_V2_3, b"\x01\xff", b"\x00\x00", UnicodeDecodeError),
    # APIC truncated after encoding byte
    (b"APIC", ID3_V2_3, b"\x00", b"\x00\x00", TypeError),
    # RVA2 missing \x00 separator
    (b"RVA2", ID3_V2_4, b"no-null-separator", b"\x00\x00", ValueError),
    # RVAD truncated
    (b"RVAD", ID3_V2_3, b"\x10", b"\x00\x00", IndexError),
])
def test_createFrame_normalizes_scattered_exceptions(fid, version, data,
                                                     flags, cause_type):
    """Whatever a corrupt frame payload raises, createFrame re-raises it as
    the library's frame-level FrameException, chaining the original."""
    tag_header = TagHeader(version=version)
    frame_header = FrameHeader(fid, version)
    frame_header._flags = bytes2bin(flags)
    frame_header.data_size = len(data)

    with pytest.raises(FrameException) as excinfo:
        createFrame(tag_header, frame_header, data)
    assert isinstance(excinfo.value.__cause__, cause_type)


@pytest.mark.parametrize("corrupt", V23_CORRUPT,
                         ids=["zlib", "utf16-txxx", "truncated-apic",
                              "truncated-rvad"])
def test_corrupt_frame_skipped_v23_tag_survives(tmp_path, corrupt):
    """One corrupt frame in a v2.3 tag: the frame is skipped, the healthy
    frames before and after it are parsed, and Tag.parse does not raise."""
    tag_bytes = _tag(ID3_V2_3, GOOD_FRAME_1 + corrupt + GOOD_FRAME_2)
    found, tag = _parse_tag_file(tmp_path, tag_bytes)

    assert found
    assert tag.title == "Healthy Title"
    assert tag.artist == "Healthy Artist"
    # The corrupt frame did not make it into the frame set.
    corrupt_fid = corrupt[:4]
    if corrupt_fid == b"TIT2":
        # title slot is taken by the healthy frame; the corrupt one must not
        # have replaced it (asserted above via tag.title).
        pass
    else:
        assert corrupt_fid not in tag.frame_set


@pytest.mark.parametrize("corrupt", V24_CORRUPT, ids=["rva2-no-separator"])
def test_corrupt_frame_skipped_v24_tag_survives(tmp_path, corrupt):
    tag_bytes = _tag(ID3_V2_4, GOOD_FRAME_1_V24 + corrupt + GOOD_FRAME_2_V24)
    found, tag = _parse_tag_file(tmp_path, tag_bytes)

    assert found
    assert tag.title == "Healthy Title"
    assert tag.artist == "Healthy Artist"
    assert b"RVA2" not in tag.frame_set


def test_frame_raising_FrameException_still_skipped(tmp_path):
    """Pre-existing leniency: frames raising FrameException directly were
    already skipped; that behavior must not change."""
    # RVA2 is only valid in v2.4 tags; in a v2.3 tag its parse raises
    # FrameException("Invalid frame version ...").
    bad = _v23_frame(b"RVA2", b"id\x00\x01\x00\x10")
    tag_bytes = _tag(ID3_V2_3, GOOD_FRAME_1 + bad + GOOD_FRAME_2)
    found, tag = _parse_tag_file(tmp_path, tag_bytes)

    assert found
    assert tag.title == "Healthy Title"
    assert tag.artist == "Healthy Artist"
    assert b"RVA2" not in tag.frame_set


def test_healthy_tag_unaffected(tmp_path):
    """Leniency must not narrow: all healthy frames still load."""
    tag_bytes = _tag(ID3_V2_3, GOOD_FRAME_1 + GOOD_FRAME_2)
    found, tag = _parse_tag_file(tmp_path, tag_bytes)
    assert found
    assert tag.title == "Healthy Title"
    assert tag.artist == "Healthy Artist"


def test_corrupt_frame_does_not_abort_id3_file_load(tmp_path):
    """eyed3.load() of an .id3 file with a corrupt frame returns a TagFile
    with the healthy frames intact."""
    tag_bytes = _tag(ID3_V2_3, GOOD_FRAME_1 + CORRUPT_ZLIB_V23 + GOOD_FRAME_2)
    id3_file = tmp_path / "corrupt.id3"
    id3_file.write_bytes(tag_bytes)

    audio_file = eyed3.load(id3_file)
    assert audio_file is not None
    assert audio_file.type == eyed3.core.AUDIO_NONE
    assert audio_file.tag is not None
    assert audio_file.tag.title == "Healthy Title"
    assert audio_file.tag.artist == "Healthy Artist"


def test_corrupt_frame_does_not_abort_mp3_file_load(tmp_path):
    """eyed3.load() of an mp3 whose tag contains corrupt frames: the file
    loads, audio info is read, and healthy frames survive."""
    tag_bytes = _tag(ID3_V2_3,
                     GOOD_FRAME_1 + CORRUPT_UTF16_TXXX_V23 +
                     CORRUPT_APIC_V23 + GOOD_FRAME_2)
    mp3_file = tmp_path / "corrupt.mp3"
    mp3_file.write_bytes(tag_bytes + MP3_AUDIO)

    audio_file = eyed3.load(mp3_file)
    assert audio_file is not None
    assert audio_file.tag is not None
    assert audio_file.tag.title == "Healthy Title"
    assert audio_file.tag.artist == "Healthy Artist"
    assert audio_file.info is not None
    assert audio_file.info.time_secs > 0
