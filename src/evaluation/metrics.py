from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, f1_score


def compute_weighted_f1(
    y_true: List[str], y_pred: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Calcule le F1-score par classe et le Weighted F1-score global.
    """
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    report = classification_report(y_true, y_pred, output_dict=True)

    f1_per_class = {
        label: float(metrics["f1-score"])
        for label, metrics in report.items()
        if label not in ["accuracy", "macro avg", "weighted avg"]
    }

    return weighted_f1, f1_per_class


def print_evaluation_report(
    y_true: List[str], y_pred: List[str], model_name: str = "Modèle"
):
    """
    Affiche un rapport structuré des performances dans la console.
    """
    weighted_f1, f1_per_class = compute_weighted_f1(y_true, y_pred)

    print("\n" + "=" * 55)
    print(f" 📊 RAPPORT D'ÉVALUATION - {model_name.upper()}")
    print("=" * 55)
    print(f"Weighted F1-Score Global : {weighted_f1 * 100:.2f}%")
    status = "✅ ATTEINT" if weighted_f1 >= 0.92 else "❌ NON ATTEINT"
    print(f"Objectif de performance (>= 92%) : {status}\n")

    print("Détail des F1-scores par classe (F1_j) :")
    for category, score in f1_per_class.items():
        print(f"  - {category:<25} : {score * 100:.2f}%")
    print("=" * 55 + "\n")