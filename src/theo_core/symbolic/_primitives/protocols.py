"""Protocols (abstract interfaces) for repositories and serializers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from theo_core.symbolic._primitives.identifiers import SymbolicId

T = TypeVar("T")


@runtime_checkable
class Repository(Protocol[T]):
    """Generic repository protocol.

    Behavioral guarantees:
    - ``save()`` preserves data integrity.
    - ``load()`` reconstructs identical data.
    - Operations MUST NOT mutate the supplied object.
    - Loading the same data multiple times yields equivalent objects.
    """

    def save(self, entity_id: SymbolicId, entity: T) -> None:
        """Persist an entity.

        Args:
            entity_id: Unique identifier for the entity.
            entity: The entity to persist.

        """
        ...

    def load(self, entity_id: SymbolicId) -> T | None:
        """Load an entity by identifier.

        Args:
            entity_id: Unique identifier to look up.

        Returns:
            The entity if found, else ``None``.

        """
        ...

    def exists(self, entity_id: SymbolicId) -> bool:
        """Check whether an entity exists.

        Args:
            entity_id: Unique identifier to check.

        Returns:
            ``True`` if the entity exists.

        """
        ...

    def delete(self, entity_id: SymbolicId) -> None:
        """Remove an entity.

        Args:
            entity_id: Unique identifier of the entity to remove.

        """
        ...


@runtime_checkable
class Serializer(Protocol[T]):
    """Deterministic serialization contract.

    Guarantees:
    - ``serialize(deserialize(s))`` produces an equivalent string.
    - Output is deterministic for identical input.
    """

    def serialize(self, entity: T) -> str:
        """Serialize an entity to a JSON string.

        Args:
            entity: The entity to serialize.

        Returns:
            A deterministic JSON string.

        """
        ...

    def deserialize(self, data: str) -> T:
        """Deserialize a JSON string to an entity.

        Args:
            data: A JSON string previously produced by ``serialize``.

        Returns:
            The deserialized entity.

        """
        ...
