from __future__ import annotations
import customtkinter as ctk
import threading
import numpy as np
import os, sys

# Ajuste path para ejecución flexible
if 'src' not in {p.split(os.sep)[-1] for p in sys.path}:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.core.function_parser import FunctionParser
from src.solvers.gradient_descent import GradientDescentSolver
from src.solvers.lagrange import LagrangeSolver
from src.solvers.unconstrained import UnconstrainedMinimizer
from src.solvers.penalty import PenaltyInequalitySolver
from src.core.constraint_checker import evaluate_constraints
from src.core.results import format_results_table


class MainApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Lagrange Engine - Optimizador Avanzado")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Entradas principales
        self.var_entry = ctk.CTkEntry(self, placeholder_text="Variables (ej: x y)")
        self.var_entry.pack(fill="x", padx=10, pady=5)

        self.obj_entry = ctk.CTkTextbox(self, height=80)
        self.obj_entry.insert("1.0", "x**2 + y**2")
        self.obj_entry.pack(fill="x", padx=10, pady=5)

        self.constr_entry = ctk.CTkTextbox(self, height=120)
        self.constr_entry.insert("1.0", "")
        self.constr_entry.pack(fill="x", padx=10, pady=5)

        # Frame métodos
        self.method_frame = ctk.CTkFrame(self)
        self.method_frame.pack(fill="x", padx=10, pady=5)
        self.use_uncon = ctk.CTkCheckBox(self.method_frame, text="Min. sin Restricciones (BFGS)")
        self.use_gd = ctk.CTkCheckBox(self.method_frame, text="Gradiente Descendente")
        self.use_lagrange = ctk.CTkCheckBox(self.method_frame, text="Lagrange (igualdades)")
        self.use_penalty = ctk.CTkCheckBox(self.method_frame, text="Penalización (desigualdades)")
        for cb in (self.use_uncon, self.use_gd):
            cb.select()
        for cb in (self.use_uncon, self.use_gd, self.use_lagrange, self.use_penalty):
            cb.pack(side="left", padx=5)

        # Punto inicial
        self.start_entry = ctk.CTkEntry(self, placeholder_text="Punto inicial (ej: 0 0)")
        self.start_entry.insert(0, "0 0")
        self.start_entry.pack(fill="x", padx=10, pady=5)

        # Botón ejecutar
        self.run_btn = ctk.CTkButton(self, text="Resolver", command=self.run_solvers)
        self.run_btn.pack(pady=10)

        # Salida
        self.output = ctk.CTkTextbox(self)
        self.output.pack(fill="both", expand=True, padx=10, pady=10)

    def run_solvers(self) -> None:
        threading.Thread(target=self._run_solvers_impl, daemon=True).start()

    def _parse_start(self, n: int) -> np.ndarray:
        text = self.start_entry.get().strip()
        if not text:
            return np.zeros(n)
        vals = [float(p) for p in text.replace(',', ' ').split() if p]
        if len(vals) != n:
            raise ValueError("Dimensión de punto inicial no coincide con variables")
        return np.array(vals, dtype=float)

    def _run_solvers_impl(self) -> None:
        self.output.delete("1.0", "end")
        try:
            vars_text = self.var_entry.get().strip()
            obj_text = self.obj_entry.get("1.0", "end").strip()
            constr_text = self.constr_entry.get("1.0", "end").strip()
            problem = FunctionParser.parse_problem(obj_text, vars_text, constr_text)
            start = self._parse_start(len(problem.variables))
            results = []
            # Ejecutar métodos seleccionados
            if self.use_uncon.get() == 1:
                results.append(UnconstrainedMinimizer(problem.objective_expr, problem.variables).solve(start))
            if self.use_gd.get() == 1:
                results.append(GradientDescentSolver(problem.objective_expr, problem.variables).solve(start))
            if self.use_lagrange.get() == 1 and problem.equality_constraints:
                try:
                    results.append(LagrangeSolver(problem.objective_expr, problem.variables, problem.equality_constraints).solve(start))
                except Exception as e:
                    self.output.insert("end", f"[Lagrange] Error: {e}\n")
            if self.use_penalty.get() == 1 and problem.inequality_constraints:
                results.append(PenaltyInequalitySolver(problem.objective_expr, problem.variables, problem.inequality_constraints).solve(start))

            # Mostrar resultados
            self.output.insert("end", format_results_table(results) + "\n\n")
            eq_status, ineq_status = evaluate_constraints(problem, results)
            if eq_status:
                self.output.insert("end", "Restricciones de igualdad:\n" + "\n".join(eq_status) + "\n")
            if ineq_status:
                self.output.insert("end", "\nRestricciones de desigualdad:\n" + "\n".join(ineq_status) + "\n")

            # Trayectoria gradiente
            for r in results:
                if r.method == 'GradientDescent' and 'history' in r.extra:
                    self.output.insert("end", "\nTrayectoria Gradiente (iter, ||grad||):\n")
                    for it, xvec, gnorm in r.extra['history'][:30]:
                        self.output.insert("end", f"{it}: {gnorm:.3e} {xvec}\n")

            # Plot si 2 variables
            if len(problem.variables) == 2 and results:
                try:
                    from src.visualization.plots import surface_with_point
                    best = min(results, key=lambda r: r.objective_value)
                    bundle = surface_with_point(problem.objective_expr, problem.variables, best.point)
                    tmp_path = os.path.join(os.path.dirname(__file__), 'last_plot.html')
                    bundle.figure.write_html(tmp_path)
                    self.output.insert("end", f"\nGráfica guardada: {tmp_path}\n")
                except Exception as e:  # pragma: no cover
                    self.output.insert("end", f"[Plot] Error: {e}\n")
        except Exception as e:  # pragma: no cover
            self.output.insert("end", f"Error: {e}\n")


def run() -> None:
    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    run()
