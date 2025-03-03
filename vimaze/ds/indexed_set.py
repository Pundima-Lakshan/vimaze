from typing import TypeVar, List, Set, Generic

T = TypeVar('T')  # Define a generic type


class IndexedSet(Generic[T]):
    def __init__(self):
        self.set: Set[T] = set()  # Set to ensure uniqueness and O(1) lookup
        self.list: List[T] = []  # List to maintain order and support indexing

    def add(self, value: T):
        """Add a value to the IndexedSet if it doesn't already exist."""
        if value not in self.set:
            self.set.add(value)
            self.list.append(value)

    def get_by_index(self, index: int) -> T:
        """Access an item by index."""
        return self.list[index]

    def lookup(self, value: T) -> bool:
        """Check if a value exists in the IndexedSet."""
        return value in self.set

    def pop_at(self, index: int):
        """Remove a value at given index and returns it"""
        value = self.list.pop(index)
        self.set.remove(value)
        return value

    def __len__(self) -> int:
        """Return the number of items in the IndexedSet."""
        return len(self.list)
