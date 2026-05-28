from src.config.label_schema import label_binary_from_multiclass, map_label


def test_coswara_positive_maps_to_covid():
    label, note = map_label("coswara", {"covid_status": "positive"})
    assert label == "COVID"
    assert "Coswara" in note


def test_tb_negative_maps_to_control():
    label, _ = map_label("tb", {"tb_status": "negative"})
    assert label == "HEALTHY_OR_NONTARGET"
    assert label_binary_from_multiclass(label) == "CONTROL"


def test_unknown_coughvid_label_is_excluded():
    label, _ = map_label("coughvid", {"status": "unknown"})
    assert label is None
