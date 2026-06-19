"""
FedFOD Protocol Buffer Stubs
==============================

Hand-written, compiler-free message classes that serialise / deserialise
using ``struct`` + ``numpy.tobytes`` for efficient binary transport over
gRPC or raw sockets.  Each message class exposes:

    .serialize()   → bytes
    @classmethod .deserialize(data) → instance

All multi-byte integers are stored **big-endian** (network byte-order).
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ===================================================================== #
#                       INTERNAL HELPERS                                #
# ===================================================================== #

def _pack_numpy_dict(d: Dict[str, np.ndarray]) -> bytes:
    """Serialise a ``{name: ndarray}`` dict into a self-describing blob.

    Layout (per entry):
        4 bytes  — name length (uint32)
        N bytes  — name (utf-8)
        4 bytes  — dtype string length
        M bytes  — dtype string (utf-8, e.g. '<f4')
        4 bytes  — ndim (uint32)
        ndim×4   — shape (each uint32)
        4 bytes  — data length in bytes (uint32)
        D bytes  — raw array data

    Header:
        4 bytes  — number of entries (uint32)
    """
    parts: List[bytes] = []
    parts.append(struct.pack("!I", len(d)))

    for name, arr in d.items():
        name_bytes = name.encode("utf-8")
        dtype_str = arr.dtype.str.encode("utf-8")
        data_bytes = arr.tobytes()

        parts.append(struct.pack("!I", len(name_bytes)))
        parts.append(name_bytes)
        parts.append(struct.pack("!I", len(dtype_str)))
        parts.append(dtype_str)
        parts.append(struct.pack("!I", arr.ndim))
        for s in arr.shape:
            parts.append(struct.pack("!I", s))
        parts.append(struct.pack("!I", len(data_bytes)))
        parts.append(data_bytes)

    return b"".join(parts)


def _unpack_numpy_dict(b: bytes) -> Dict[str, np.ndarray]:
    """Inverse of :func:`_pack_numpy_dict`."""
    offset = 0

    (n_entries,) = struct.unpack_from("!I", b, offset)
    offset += 4

    result: Dict[str, np.ndarray] = {}
    for _ in range(n_entries):
        (name_len,) = struct.unpack_from("!I", b, offset)
        offset += 4
        name = b[offset: offset + name_len].decode("utf-8")
        offset += name_len

        (dtype_len,) = struct.unpack_from("!I", b, offset)
        offset += 4
        dtype_str = b[offset: offset + dtype_len].decode("utf-8")
        offset += dtype_len

        (ndim,) = struct.unpack_from("!I", b, offset)
        offset += 4
        shape = []
        for _ in range(ndim):
            (s,) = struct.unpack_from("!I", b, offset)
            offset += 4
            shape.append(s)

        (data_len,) = struct.unpack_from("!I", b, offset)
        offset += 4
        data = b[offset: offset + data_len]
        offset += data_len

        arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).copy()
        if shape:
            arr = arr.reshape(shape)
        result[name] = arr

    return result


def _pack_string(s: str) -> bytes:
    """Length-prefixed UTF-8 string."""
    encoded = s.encode("utf-8")
    return struct.pack("!I", len(encoded)) + encoded


def _unpack_string(b: bytes, offset: int) -> tuple[str, int]:
    """Read a length-prefixed UTF-8 string, return (string, new_offset)."""
    (length,) = struct.unpack_from("!I", b, offset)
    offset += 4
    s = b[offset: offset + length].decode("utf-8")
    return s, offset + length


def _pack_bytes_blob(data: bytes) -> bytes:
    """Length-prefixed raw bytes."""
    return struct.pack("!I", len(data)) + data


def _unpack_bytes_blob(b: bytes, offset: int) -> tuple[bytes, int]:
    """Read a length-prefixed blob, return (blob, new_offset)."""
    (length,) = struct.unpack_from("!I", b, offset)
    offset += 4
    return b[offset: offset + length], offset + length


# Magic bytes for message type identification
_MAGIC_GRADIENT = b"GRDM"
_MAGIC_GLOBAL_UPDATE = b"GLUP"
_MAGIC_PROTOTYPE = b"PROT"
_MAGIC_REGISTRATION = b"CREG"
_MAGIC_HEARTBEAT = b"HBTM"


# ===================================================================== #
#                      MESSAGE DATACLASSES                              #
# ===================================================================== #

@dataclass
class GradientMessage:
    """Client → Aggregator gradient upload."""

    client_id: str = ""
    round_number: int = 0
    compressed_delta: bytes = b""
    cv_delta: bytes = b""
    precision_q: float = 1.0
    num_samples: int = 0
    staleness: int = 0
    is_async: bool = False

    # -- convenience constructors from numpy dicts --------------------

    @classmethod
    def from_numpy(
        cls,
        client_id: str,
        round_number: int,
        weights_delta: Dict[str, np.ndarray],
        cv_delta: Optional[Dict[str, np.ndarray]] = None,
        precision_q: float = 1.0,
        num_samples: int = 0,
        staleness: int = 0,
        is_async: bool = False,
    ) -> GradientMessage:
        compressed = _pack_numpy_dict(weights_delta)
        cv_bytes = _pack_numpy_dict(cv_delta) if cv_delta else b""
        return cls(
            client_id=client_id,
            round_number=round_number,
            compressed_delta=compressed,
            cv_delta=cv_bytes,
            precision_q=precision_q,
            num_samples=num_samples,
            staleness=staleness,
            is_async=is_async,
        )

    def to_numpy(self) -> Dict[str, np.ndarray]:
        return _unpack_numpy_dict(self.compressed_delta)

    def cv_to_numpy(self) -> Optional[Dict[str, np.ndarray]]:
        if not self.cv_delta:
            return None
        return _unpack_numpy_dict(self.cv_delta)

    # -- wire format ---------------------------------------------------

    def serialize(self) -> bytes:
        parts: List[bytes] = [_MAGIC_GRADIENT]
        parts.append(_pack_string(self.client_id))
        parts.append(struct.pack("!i", self.round_number))
        parts.append(_pack_bytes_blob(self.compressed_delta))
        parts.append(_pack_bytes_blob(self.cv_delta))
        parts.append(struct.pack("!d", self.precision_q))
        parts.append(struct.pack("!I", self.num_samples))
        parts.append(struct.pack("!i", self.staleness))
        parts.append(struct.pack("!?", self.is_async))
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> GradientMessage:
        offset = 0
        magic = data[offset: offset + 4]
        assert magic == _MAGIC_GRADIENT, f"Bad magic: {magic!r}"
        offset += 4

        client_id, offset = _unpack_string(data, offset)
        (round_number,) = struct.unpack_from("!i", data, offset)
        offset += 4
        compressed_delta, offset = _unpack_bytes_blob(data, offset)
        cv_delta, offset = _unpack_bytes_blob(data, offset)
        (precision_q,) = struct.unpack_from("!d", data, offset)
        offset += 8
        (num_samples,) = struct.unpack_from("!I", data, offset)
        offset += 4
        (staleness,) = struct.unpack_from("!i", data, offset)
        offset += 4
        (is_async,) = struct.unpack_from("!?", data, offset)
        offset += 1

        return cls(
            client_id=client_id,
            round_number=round_number,
            compressed_delta=compressed_delta,
            cv_delta=cv_delta,
            precision_q=precision_q,
            num_samples=num_samples,
            staleness=staleness,
            is_async=is_async,
        )


@dataclass
class GlobalUpdateMessage:
    """Aggregator → Clients global model broadcast."""

    round_number: int = 0
    global_weights: bytes = b""
    global_cv: bytes = b""
    novel_prototypes: List[bytes] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)

    # -- convenience ---------------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        round_number: int,
        global_weights: Dict[str, np.ndarray],
        global_cv: Optional[Dict[str, np.ndarray]] = None,
        novel_prototypes: Optional[List[np.ndarray]] = None,
        class_names: Optional[List[str]] = None,
    ) -> GlobalUpdateMessage:
        gw = _pack_numpy_dict(global_weights)
        gc = _pack_numpy_dict(global_cv) if global_cv else b""
        proto_bytes = [p.tobytes() for p in (novel_prototypes or [])]
        return cls(
            round_number=round_number,
            global_weights=gw,
            global_cv=gc,
            novel_prototypes=proto_bytes,
            class_names=list(class_names or []),
        )

    def weights_to_numpy(self) -> Dict[str, np.ndarray]:
        return _unpack_numpy_dict(self.global_weights)

    def cv_to_numpy(self) -> Optional[Dict[str, np.ndarray]]:
        if not self.global_cv:
            return None
        return _unpack_numpy_dict(self.global_cv)

    # -- wire format ---------------------------------------------------

    def serialize(self) -> bytes:
        parts: List[bytes] = [_MAGIC_GLOBAL_UPDATE]
        parts.append(struct.pack("!i", self.round_number))
        parts.append(_pack_bytes_blob(self.global_weights))
        parts.append(_pack_bytes_blob(self.global_cv))

        # novel prototypes list
        parts.append(struct.pack("!I", len(self.novel_prototypes)))
        for pb in self.novel_prototypes:
            parts.append(_pack_bytes_blob(pb))

        # class names list
        parts.append(struct.pack("!I", len(self.class_names)))
        for cn in self.class_names:
            parts.append(_pack_string(cn))

        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> GlobalUpdateMessage:
        offset = 0
        magic = data[offset: offset + 4]
        assert magic == _MAGIC_GLOBAL_UPDATE, f"Bad magic: {magic!r}"
        offset += 4

        (round_number,) = struct.unpack_from("!i", data, offset)
        offset += 4
        global_weights, offset = _unpack_bytes_blob(data, offset)
        global_cv, offset = _unpack_bytes_blob(data, offset)

        (n_protos,) = struct.unpack_from("!I", data, offset)
        offset += 4
        novel_prototypes: List[bytes] = []
        for _ in range(n_protos):
            pb, offset = _unpack_bytes_blob(data, offset)
            novel_prototypes.append(pb)

        (n_names,) = struct.unpack_from("!I", data, offset)
        offset += 4
        class_names: List[str] = []
        for _ in range(n_names):
            cn, offset = _unpack_string(data, offset)
            class_names.append(cn)

        return cls(
            round_number=round_number,
            global_weights=global_weights,
            global_cv=global_cv,
            novel_prototypes=novel_prototypes,
            class_names=class_names,
        )


@dataclass
class PrototypeMessage:
    """Client → Aggregator novel-class prototype submission."""

    client_id: str = ""
    embedding: bytes = b""
    confidence: float = 0.0
    class_name: str = "unknown"
    metadata_json: str = "{}"

    @classmethod
    def from_numpy(
        cls,
        client_id: str,
        embedding: np.ndarray,
        confidence: float = 0.0,
        class_name: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PrototypeMessage:
        return cls(
            client_id=client_id,
            embedding=embedding.astype(np.float32).tobytes(),
            confidence=confidence,
            class_name=class_name,
            metadata_json=json.dumps(metadata or {}),
        )

    def embedding_to_numpy(self, dim: int = 512) -> np.ndarray:
        return np.frombuffer(self.embedding, dtype=np.float32).copy().reshape(dim)

    def serialize(self) -> bytes:
        parts: List[bytes] = [_MAGIC_PROTOTYPE]
        parts.append(_pack_string(self.client_id))
        parts.append(_pack_bytes_blob(self.embedding))
        parts.append(struct.pack("!d", self.confidence))
        parts.append(_pack_string(self.class_name))
        parts.append(_pack_string(self.metadata_json))
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> PrototypeMessage:
        offset = 0
        magic = data[offset: offset + 4]
        assert magic == _MAGIC_PROTOTYPE, f"Bad magic: {magic!r}"
        offset += 4

        client_id, offset = _unpack_string(data, offset)
        embedding, offset = _unpack_bytes_blob(data, offset)
        (confidence,) = struct.unpack_from("!d", data, offset)
        offset += 8
        class_name, offset = _unpack_string(data, offset)
        metadata_json, offset = _unpack_string(data, offset)

        return cls(
            client_id=client_id,
            embedding=embedding,
            confidence=confidence,
            class_name=class_name,
            metadata_json=metadata_json,
        )


@dataclass
class ClientRegistration:
    """Client → Aggregator initial registration (no serialisation needed
    for MVP — JSON over REST is fine)."""

    client_id: str = ""
    profile: str = "default"
    hardware_info: str = ""
    bandwidth_mbps: float = 0.0

    def serialize(self) -> bytes:
        parts: List[bytes] = [_MAGIC_REGISTRATION]
        parts.append(_pack_string(self.client_id))
        parts.append(_pack_string(self.profile))
        parts.append(_pack_string(self.hardware_info))
        parts.append(struct.pack("!d", self.bandwidth_mbps))
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> ClientRegistration:
        offset = 0
        magic = data[offset: offset + 4]
        assert magic == _MAGIC_REGISTRATION, f"Bad magic: {magic!r}"
        offset += 4

        client_id, offset = _unpack_string(data, offset)
        profile, offset = _unpack_string(data, offset)
        hardware_info, offset = _unpack_string(data, offset)
        (bandwidth_mbps,) = struct.unpack_from("!d", data, offset)
        offset += 8

        return cls(
            client_id=client_id,
            profile=profile,
            hardware_info=hardware_info,
            bandwidth_mbps=bandwidth_mbps,
        )


@dataclass
class HeartbeatMessage:
    """Periodic keep-alive from client to aggregator."""

    client_id: str = ""
    timestamp: float = field(default_factory=time.time)
    status: str = "alive"
    round_number: int = 0

    def serialize(self) -> bytes:
        parts: List[bytes] = [_MAGIC_HEARTBEAT]
        parts.append(_pack_string(self.client_id))
        parts.append(struct.pack("!d", self.timestamp))
        parts.append(_pack_string(self.status))
        parts.append(struct.pack("!i", self.round_number))
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> HeartbeatMessage:
        offset = 0
        magic = data[offset: offset + 4]
        assert magic == _MAGIC_HEARTBEAT, f"Bad magic: {magic!r}"
        offset += 4

        client_id, offset = _unpack_string(data, offset)
        (timestamp,) = struct.unpack_from("!d", data, offset)
        offset += 8
        status, offset = _unpack_string(data, offset)
        (round_number,) = struct.unpack_from("!i", data, offset)
        offset += 4

        return cls(
            client_id=client_id,
            timestamp=timestamp,
            status=status,
            round_number=round_number,
        )
