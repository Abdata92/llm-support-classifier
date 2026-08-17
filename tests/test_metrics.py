from src.evaluation.metrics import compute_weighted_f1


def test_compute_weighted_f1_perfect():
    y_true = ["Billing", "Product Support", "Technical"]
    y_pred = ["Billing", "Product Support", "Technical"]
    weighted_f1, f1_map = compute_weighted_f1(y_true, y_pred)
    assert weighted_f1 == 1.0
    assert f1_map["Billing"] == 1.0


def test_compute_weighted_f1_imperfect():
    y_true = ["Billing", "Billing", "Technical"]
    y_pred = ["Billing", "Technical", "Technical"]
    weighted_f1, _ = compute_weighted_f1(y_true, y_pred)
    assert 0.0 < weighted_f1 < 1.0