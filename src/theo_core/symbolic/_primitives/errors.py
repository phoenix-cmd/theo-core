"""Error hierarchy for the symbolic runtime.

All symbolic errors inherit from ``SymbolicError``.
Validation errors form a subhierarchy under ``ValidationError``.
Serialization errors form a subhierarchy under ``SerializationError``.
"""


class SymbolicError(Exception):
    """Base exception for all symbolic runtime errors."""


class ValidationError(SymbolicError):
    """Base for structural validation failures."""


class DuplicateIdError(ValidationError):
    """Raised when a duplicate identifier is inserted."""


class CycleDetectedError(ValidationError):
    """Raised when a cycle is detected in an acyclic graph."""


class OrphanNodeError(ValidationError):
    """Raised when a node has no edges and is not explicitly isolated."""


class InvalidRelationError(ValidationError):
    """Raised when an edge has an unsupported or invalid relation type."""


class DanglingEdgeError(ValidationError):
    """Raised when an edge references a node that does not exist."""


class ConstraintViolationError(ValidationError):
    """Raised when a FATAL constraint violation is detected during validation."""


class SerializationError(SymbolicError):
    """Base for serialization/deserialization failures."""


class DeserializationError(SerializationError):
    """Raised when deserialized data is corrupt or incompatible."""


class ChecksumMismatchError(DeserializationError):
    """Raised when a checksum does not match the deserialized payload."""


class SchemaVersionError(DeserializationError):
    """Raised when a schema version is incompatible."""
