from collections.abc import Callable, Sequence
import contextlib
from typing import Any, Generic, TypeVar
import uuid

import beartype
import jaxtyping as jt

ListenerT = TypeVar("ListenerT")

HANDLE = uuid.UUID


class EventGenerator(Generic[ListenerT]):
    """Manage listeners and fire events."""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(self, default_args: Sequence[Any] = tuple()):
        self._default_args = tuple(default_args)
        self._event_listeners: dict[HANDLE, ListenerT] = {}

    def add_listener(self, listener: ListenerT) -> uuid.UUID:
        """Add listener."""
        handle = uuid.uuid4()
        self._event_listeners[handle] = listener
        return handle

    def remove_listener(self, handle: HANDLE) -> ListenerT:
        """Remove listener."""
        return self._event_listeners.pop(handle)

    def fire_event(self, event_fn: Callable, *args, **kwargs):
        """Fire event."""
        args = self._default_args + args
        for listener in list(self._event_listeners.values()):
            getattr(listener, event_fn.__name__)(*args, **kwargs)

    @contextlib.contextmanager
    def listen_context(self, *listener: ListenerT):
        """Listen context."""
        uuids = tuple()
        try:
            uuids = tuple(map(self.add_listener, listener))
            yield
        finally:
            tuple(map(self.remove_listener, uuids))

    T = TypeVar("T", bound=ListenerT)

    def get(self, handle: HANDLE) -> T:
        """Get a listener using its handle."""
        return self._event_listeners[handle]

    def find(
        # pylint: disable=redefined-builtin
        self,
        *,
        type: type[T] | None = None,
    ) -> list[T]:
        """Find listeners matching the passed filter(s)."""

        def filtr(listener: ListenerT) -> bool:
            if type is not None and not isinstance(listener, type):
                return False

            return True

        return [listener for listener in self._event_listeners.values() if filtr(listener)]

    def reset(self):
        """Reset the event generator."""
        self._event_listeners.clear()
