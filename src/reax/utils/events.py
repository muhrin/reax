import contextlib
from typing import Callable, Generic, TypeVar
import uuid

ListenerT = TypeVar("ListenerT")


class EventGenerator(Generic[ListenerT]):
    """Manage listeners and fire events"""

    def __init__(self):
        self._event_listeners: dict[uuid.UUID, ListenerT] = {}

    def add_listener(self, listener: ListenerT) -> uuid.UUID:
        handle = uuid.uuid4()
        self._event_listeners[handle] = listener
        return handle

    def remove_listener(self, handle: uuid.UUID) -> ListenerT:
        return self._event_listeners.pop(handle)

    def fire_event(self, event_fn: Callable, *args, **kwargs):
        for listener in self._event_listeners.values():
            getattr(listener, event_fn.__name__)(*args, **kwargs)

    @contextlib.contextmanager
    def listen_context(self, *listener: ListenerT):
        uuids = tuple()
        try:
            uuids = tuple(map(self.add_listener, listener))
            yield
        finally:
            tuple(map(self.remove_listener, uuids))
