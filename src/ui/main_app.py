from __future__ import annotations
import customtkinter as ctk
import threading
import numpy as np
from src.core.function_parser import FunctionParser
from src.solvers.gradient_descent import GradientDescentSolver
from src.solvers.lagrange import LagrangeSolver
from src.solvers.unconstrained import UnconstrainedMinimizer
from src.core.constraint_checker import evaluate_constraints
from src.core.results import format_results_table


class MainApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Lagrange Engine - Optimizador Avanzado")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Variables
        self.var_entry = ctk.CTkEntry(self, placeholder_text="Variables (ej: x y)")
        self.var_entry.pack(fill="x", padx=10, pady=5)

        # Objetivo
        self.obj_entry = ctk.CTkTextbox(self, height=80)
        self.obj_entry.insert("1.0", "x**2 + y**2")
        self.obj_entry.pack(fill="x", padx=10, pady=5)

        # Restricciones
        self.constr_entry = ctk.CTkTextbox(self, height=120)
        self.constr_entry.insert("1.0", "")
        self.constr_entry.pack(fill="x", padx=10, pady=5)

        # Selección de métodos
        self.method_frame = ctk.CTkFrame(self)
        self.method_frame.pack(fill="x", padx=10, pady=5)
        self.use_uncon = ctk.CTkCheckBox(self.method_frame, text="Min. sin Restricciones (BFGS)")
        self.use_gd = ctk.CTkCheckBox(self.method_frame, text="Gradiente Descendente")
        self.use_lagrange = ctk.CTkCheckBox(self.method_frame, text="Lagrange (solo igualdades)")
        self.use_uncon.select()
        self.use_gd.select()
        self.use_uncon.pack(side="left", padx=5)
        self.use_gd.pack(side="left", padx=5)
        self.use_lagrange.pack(side="left", padx=5)

        # Botón de acción
        self.run_btn = ctk.CTkButton(self, text="Resolver", command=self.run_solvers)
        self.run_btn.pack(pady=10)

        # Salida de resultados
        self.output = ctk.CTkTextbox(self)
        self.output.pack(fill="both", expand=True, padx=10, pady=10)

    def run_solvers(self) -> None:
        t = threading.Thread(target=self._run_solvers_impl, daemon=True)
        t.start()

    def _run_solvers_impl(self) -> None:
        self.output.delete("1.0", "end")
        try:
            vars_text = self.var_entry.get().strip()
            obj_text = self.obj_entry.get("1.0", "end").strip()
            constr_text = self.constr_entry.get("1.0", "end").strip()
            problem = FunctionParser.parse_problem(obj_text, vars_text, constr_text)
            start = np.zeros(len(problem.variables))
            results = []
            if self.use_uncon.get() == 1:
                results.append(UnconstrainedMinimizer(problem.objective_expr, problem.variables).solve(start))
            if self.use_gd.get() == 1:
                results.append(GradientDescentSolver(problem.objective_expr, problem.variables).solve(start))
            if self.use_lagrange.get() == 1 and problem.equality_constraints:
                results.append(LagrangeSolver(problem.objective_expr, problem.variables, problem.equality_constraints).solve(start))

            table = format_results_table(results)
            self.output.insert("end", table + "\n\n")
            eq_status, ineq_status = evaluate_constraints(problem, results)
            if eq_status:
                self.output.insert("end", "Restricciones de igualdad:\n")
                for line in eq_status:
                    self.output.insert("end", line + "\n")
            if ineq_status:
                self.output.insert("end", "\nRestricciones de desigualdad:\n")
                for line in ineq_status:
                    self.output.insert("end", line + "\n")
        except Exception as e:  # pragma: no cover
            self.output.insert("end", f"Error: {e}\n")


def run() -> None:
    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    run()
