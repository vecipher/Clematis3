def test_import_optional():
    # Extension may not be present; ensure module import still succeeds
    import clematis.native.t1 as nt1
    assert hasattr(nt1, "available")
    assert isinstance(nt1.available(), bool)
