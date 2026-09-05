"""A tiny in-memory inventory. Quantities are whole units; a SKU is a non-empty string."""
import json


class Store:
    def __init__(self):
        self._items = {}

    def add(self, sku, qty):
        """Add qty units of sku. qty must be a positive integer."""
        if not isinstance(sku, str) or not sku:
            raise ValueError("sku must be a non-empty string")
        if not isinstance(qty, int) or qty <= 0:
            raise ValueError(f"qty must be a positive integer, got {qty!r}")
        self._items[sku] = self._items.get(sku, 0) + qty
        return self._items[sku]

    def remove(self, sku, qty):
        """Remove qty units of sku and return the remaining quantity."""
        if not isinstance(qty, int) or qty <= 0:
            raise ValueError(f"qty must be a positive integer, got {qty!r}")
        current = self._items.get(sku, 0)
        self._items[sku] = current - qty
        return self._items[sku]

    def quantity(self, sku):
        return self._items.get(sku, 0)

    def total(self):
        return sum(self._items.values())

    def skus(self):
        return sorted(self._items)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._items, fh, sort_keys=True)

    def load(self, path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("inventory file must hold an object")
        self._items = {str(k): int(v) for k, v in data.items()}
        return self
