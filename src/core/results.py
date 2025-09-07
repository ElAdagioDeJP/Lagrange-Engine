from __future__ import annotations
from typing import List
from src.solvers.base import SolverResult

def format_results_table(results: List[SolverResult]) -> str:
    if not results:
        return "(Sin resultados)"
    headers = ["Método", "f(x)", "Convergió", "Iteraciones"]
    rows = []
    for r in results:
        rows.append([
            r.method,
            f"{r.objective_value:.6g}",
            "Sí" if r.converged else "No",
            str(r.iterations)
        ])
    col_widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    def fmt_row(row):
        return " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))
    sep = "-+-".join('-'*w for w in col_widths)
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in rows)
    # Append points below
    for r in results:
        out.append(f"{r.method} punto: {r.point}")
    return "\n".join(out)
