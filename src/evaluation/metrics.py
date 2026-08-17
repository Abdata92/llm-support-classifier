from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import classification_report, f1_score


def compute_weighted_f1(
    y_true: List[str], y_pred: List[str]
) -> Tuple[float, Dict[str, float]]:
    """Calcule le F1-score par classe (F1_j) ainsi que le Weighted F1-score (F1 global).

    Formules :
        F1_j = 2 * TP_j / (2 * TP_j + FP_j + FN_j)
        F1_weighted = sum(alpha_j * F1_j) avec alpha_j = n_j / n
    """
    # Calcul du Weighted F1 global
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    # Detail par classe F1_j
    report = classification_report(y_true, y_pred, output_dict=True)

    f1_per_class = {
        label: float(metrics["f1-score"])
        for label, metrics in report.items()
        if label not in ["accuracy", "macro avg", "weighted avg"]
    }

    return weighted_f1, f1_per_class


def print_evaluation_summary(y_true: List[str], y_pred: List[str]):
    """Affiche un récapitulatif clair dans la console."""
    weighted_f1, f1_per_class = compute_weighted_f1(y_true, y_pred)

    print("\n" + "=" * 50)
    print(" 📊 RAPPORT D'ÉVALUATION - F1-SCORE")
    print("=" * 50)
    print(f"Weighted F1-Score Global : {weighted_f1 * 100:.2f}%")
    print(f"Objectif requis (>= 92%) : {'✅ ATTEINT' if weighted_f1 >= 0.92 else '❌ NON ATTEINT'}\n")

    print("Détail par catégorie (F1_j) :")
    for category, score in f1_per_class.items():
        print(f"  - {category:<25} : {score * 100:.2f}%")
    print("=" * 50 + "\n")