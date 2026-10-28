#!/usr/bin/env python3
"""Genera un tablero HTML local a partir del reporte de retrieval."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("evaluation/retrieval_benchmark_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/retrieval_metrics_v2.html"))
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    metric_names = ["command_hit@1", "command_hit@3", "command_hit@5", "command_hit@10", "mrr", "ndcg@5"]
    rows = []
    for name in metric_names:
        value = float(metrics[name])
        rows.append(f"<tr><th>{html.escape(name)}</th><td>{value:.4f}</td><td><div class='bar'><span style='width:{value * 100:.2f}%'></span></div></td></tr>")
    failures = [item for item in report["queries"] if item["first_relevant_rank"] is None]
    page = f"""<!doctype html>
<html lang='en'><meta charset='utf-8'><title>it-shell retrieval metrics</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 44px auto; max-width: 820px; color: #17212b; }}
h1 {{ margin-bottom: 6px; }} p {{ color: #52606d; }} table {{ border-collapse: collapse; width: 100%; margin: 28px 0; }}
th, td {{ border-bottom: 1px solid #d8dee4; padding: 12px; text-align: left; }}
.bar {{ height: 18px; width: 100%; background: #edf2f5; }} .bar span {{ display:block; height:100%; background:#167c80; }}
code {{ background:#edf2f5; padding:2px 5px; }}
</style>
<body><h1>it-shell retrieval metrics</h1><p>Index vectors: {report['index_vectors']} | Queries: {metrics['queries']}</p>
<table><thead><tr><th>Metric</th><th>Value</th><th>Scale</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Queries without a relevant result in Top-10</h2>
<p>{html.escape(', '.join(item['id'] for item in failures)) or 'None'}</p>
<p><code>CommandHit@k</code> means that a variant of the expected command occurred in the first k results.</p>
</body></html>"""
    args.output.write_text(page, encoding="utf-8")
    print(f"Dashboard: {args.output}")


if __name__ == "__main__":
    main()
