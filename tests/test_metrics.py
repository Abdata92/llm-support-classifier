import pytest
from src.evaluation.metrics import compute_weighted_f1


def test_compute_weighted_f1_exact_match():
    y_true = ["Product Support", "Billing", "Billing", "Technical"]
    y_pred = ["Product Support", "Billing", "Billing", "Technical"]

    weighted_f1, f1_per_class = compute_weighted_f1(y_true, y_pred)

    assert weighted_f1 == 1.0
    assert f1_per_class["Billing"] == 1.0


def test_compute_weighted_f1_imperfect():
    y_true = ["Product Support", "Billing", "Billing", "Technical"]
    y_pred = ["Product Support", "Billing", "Technical", "Technical"]

    weighted_f1, f1_per_class = compute_weighted_f1(y_true, y_pred)

    assert 0.0 < weighted_f1 < 1.0