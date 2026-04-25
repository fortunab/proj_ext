from __future__ import annotations

import pandas as pd

from bionnica.config import OUTPUTS_DIR
from bionnica.utils.io_utils import ensure_dir, save_text_report
from bionnica.utils.plotting import save_bar_chart


def main() -> None:
    output_dir = ensure_dir(OUTPUTS_DIR / "synthetic")
    metrics_df = pd.DataFrame(RESULTS)

    csv_path = output_dir / "synthetic_metrics.csv"
    txt_path = output_dir / "synthetic_metrics.txt"
    summary_path = output_dir / "metrics_summary.txt"
    table_path = output_dir / "metrics_table.txt"

    metrics_df.to_csv(csv_path, index=False)
    save_text_report(txt_path, [metrics_df.to_string(index=False)])

    summary_lines = ["=== Model Performance Summary ===", ""]
    for _, row in metrics_df.iterrows():
        summary_lines.extend([
            f"Model: {row['Model']}",
            f"  Accuracy:     {row['Accuracy']:.4f}",
            f"  Sensitivity:  {row['Sensitivity']:.4f}",
            f"  Specificity:  {row['Specificity']:.4f}",
            f"  Precision:    {row['Precision']:.4f}",
            f"  F1-score:     {row['F1-score']:.4f}",
            f"  AUC:          {row['AUC']:.4f}",
            "",
        ])
    save_text_report(summary_path, summary_lines)

    compact_table = metrics_df.to_string(index=False)
    save_text_report(table_path, [compact_table])

    models = metrics_df["Model"].tolist()
    metric_files = {
        "Accuracy": "accuracy_bar.png",
        "Sensitivity": "sensitivity_bar.png",
        "Specificity": "specificity_bar.png",
        "Precision": "precision_bar.png",
        "F1-score": "f1_bar.png",
        "AUC": "auc_bar.png",
    }

    for metric_name, filename in metric_files.items():
        save_bar_chart(
            models,
            metrics_df[metric_name].tolist(),
            "Score",
            f"{metric_name} by Model",
            output_dir / filename,
            ylim=(0.75, 0.95),
        )

    report_lines = [
        f"Saved CSV: {csv_path}",
        f"Saved TXT: {txt_path}",
        f"Saved summary TXT: {summary_path}",
        f"Saved compact table TXT: {table_path}",
        "",
        "Metrics table:",
        compact_table,
    ]
    save_text_report(output_dir / "generation_summary.txt", report_lines)
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
