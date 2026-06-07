"""Descriptor helpers for slot-backed storage in lightweight container types."""


class SlotDescriptor:
    """Base descriptor that stores values in a private slot."""

    def __set_name__(self, owner, name):
        self.name = name
        self.storage_name = f"{name}__slot"

    def __get__(self, instance, owner=None):
        if instance is None:
            return self

        try:
            return getattr(instance, self.storage_name)
        except AttributeError as exc:
            raise AttributeError(self.name) from exc


class SlottedDataDescriptor(SlotDescriptor):

    def __set__(self, instance, value):
        setattr(instance, self.storage_name, value)

    def __delete__(self, instance):
        try:
            delattr(instance, self.storage_name)
        except AttributeError as exc:
            raise AttributeError(self.name) from exc
