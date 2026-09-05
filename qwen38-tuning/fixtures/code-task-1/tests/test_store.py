from inventory import Store


def test_add_and_total():
    s = Store()
    s.add("apple", 3)
    s.add("pear", 2)
    s.add("apple", 1)
    assert s.quantity("apple") == 4
    assert s.total() == 6


def test_add_rejects_bad_qty():
    s = Store()
    for bad in (0, -1, 1.5, "2"):
        try:
            s.add("apple", bad)
        except ValueError:
            continue
        assert False, f"add accepted {bad!r}"


def test_save_and_load_roundtrip(tmp_path):
    s = Store()
    s.add("apple", 3)
    p = tmp_path / "inv.json"
    s.save(p)
    t = Store().load(p)
    assert t.quantity("apple") == 3 and t.skus() == ["apple"]
