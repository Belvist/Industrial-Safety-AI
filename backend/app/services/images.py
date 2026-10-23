"""Check untrusted JPEG/PNG dimensions before allocating decoded pixels."""

import struct


def frame_dimensions(raw):
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24 and raw[12:16] == b"IHDR":
        return struct.unpack(">II", raw[16:24])
    if raw[:2] != b"\xff\xd8":
        raise ValueError("Expected JPEG or PNG")
    offset = 2
    while offset + 4 <= len(raw):
        if raw[offset] != 0xFF:
            raise ValueError("Malformed JPEG header")
        while offset < len(raw) and raw[offset] == 0xFF:
            offset += 1
        if offset + 3 > len(raw):
            break
        marker = raw[offset]
        offset += 1
        if marker in {0xD8, 0xD9, 0xDA}:
            break
        length = int.from_bytes(raw[offset : offset + 2], "big")
        if length < 2 or offset + length > len(raw):
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if length < 8:
                break
            height, width = struct.unpack(">HH", raw[offset + 3 : offset + 7])
            return width, height
        offset += length
    raise ValueError("JPEG dimensions not found")
