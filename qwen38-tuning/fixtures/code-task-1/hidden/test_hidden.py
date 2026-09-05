"""Hidden tests: copied in by the harness AFTER the model's run. Never shown to the model."""
import pytest
from inventory import Store


def test_remove_cannot_go_negative_and_names_the_sku_and_remaining():
    s = Store()
    s.add("apple", 2)
    with pytest.raises(ValueError) as exc:
        s.remove("apple", 3)
    msg = str(exc.value)
    assert "apple" in msg and "2" in msg
    assert s.quantity("apple") == 2          # unchanged after the failed remove


def test_remove_to_exactly_zero_is_allowed():
    s = Store()
    s.add("apple", 2)
    assert s.remove("apple", 2) == 0


def test_remove_unknown_sku_raises():
    s = Store()
    with pytest.raises(ValueError):
        s.remove("ghost", 1)


def test_remove_still_rejects_bad_qty():
    s = Store()
    s.add("apple", 5)
    for bad in (0, -2, 1.5):
        with pytest.raises(ValueError):
            s.remove("apple", bad)


def test_low_stock_orders_by_qty_then_sku_and_uses_strict_threshold():
    s = Store()
    s.add("pear", 1); s.add("apple", 1); s.add("fig", 3); s.add("kiwi", 5)
    assert s.low_stock(3) == ["apple", "pear"]          # qty < 3, ties by sku
    assert s.low_stock(4) == ["apple", "pear", "fig"]
    assert s.low_stock(1) == []


def test_low_stock_on_empty_store_and_does_not_mutate():
    s = Store()
    assert s.low_stock(10) == []
    s.add("apple", 1)
    before = s.total()
    s.low_stock(10)
    assert s.total() == before


def test_existing_behaviour_unchanged():
    s = Store()
    s.add("apple", 3); s.add("pear", 2)
    assert s.total() == 5 and s.skus() == ["apple", "pear"]
