"""
Parcial VI Métodos Cuantitativos - Proyecto - Programacion No lineal
Juan Vargas - 30.448.315
Irisbel Ruiz - 30.864.236
"""

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
        self.title("🌸 Lagrange Engine - Optimizador Avanzado 🌸")
        self.geometry("1100x700")

        # Tema 
        ctk.set_appearance_mode("dark")

        # 🎨 Paleta de colores 
        self.bg_color = "#D98CB3"     # Rosa pastel de fondo
        self.frame_color = "#F4CCE3"  # Rosa lila más intenso
        self.entry_color = "#F7DDEB"  # Rosa claro para entradas
        self.accent_color = "#E6A4C5" # Botones rosados
        self.hover_color = "#D98CB3"  # Hover más oscuro
        self.text_color = "#4A3C4C"   # Gris-violeta suave

        self.configure(fg_color=self.bg_color)

        # Entradas principales
        self.var_entry = ctk.CTkEntry(
            self, placeholder_text="Variables (ej: x y)",
            fg_color=self.entry_color, border_color=self.accent_color, text_color=self.text_color
        )
        self.var_entry.pack(fill="x", padx=10, pady=5)

        self.obj_placeholder = "Función Objetivo (ej: x**2 + y**2)"
        self.obj_entry = ctk.CTkTextbox(
            self, height=80, fg_color=self.entry_color, text_color="gray", border_color=self.accent_color
        )
        self.obj_entry.pack(fill="x", padx=10, pady=5)
        self.obj_entry.bind("<FocusIn>", self._obj_focus_in)
        self.obj_entry.bind("<FocusOut>", self._obj_focus_out)

        self.constr_placeholder = "Restricciones (ej: x + y = 10)"
        self.constr_entry = ctk.CTkTextbox(
            self, height=120, fg_color=self.entry_color, text_color="gray", border_color=self.accent_color
        )
        self.constr_entry.pack(fill="x", padx=10, pady=5)
        self.constr_entry.bind("<FocusIn>", self._constr_focus_in)
        self.constr_entry.bind("<FocusOut>", self._constr_focus_out)

        # Frame métodos
        self.method_frame = ctk.CTkFrame(self, fg_color=self.frame_color, corner_radius=15)
        self.method_frame.pack(fill="x", padx=10, pady=5)

        self.use_uncon = ctk.CTkCheckBox(self.method_frame, text="🌸 Min. sin Restricciones (BFGS)", text_color=self.text_color, fg_color=self.accent_color, hover_color=self.hover_color)
        self.use_gd = ctk.CTkCheckBox(self.method_frame, text="🌸 Gradiente Descendente", text_color=self.text_color, fg_color=self.accent_color, hover_color=self.hover_color)
        self.use_lagrange = ctk.CTkCheckBox(self.method_frame, text="🌸 Lagrange (igualdades)", text_color=self.text_color, fg_color=self.accent_color, hover_color=self.hover_color)
        self.use_penalty = ctk.CTkCheckBox(self.method_frame, text="🌸 Penalización (desigualdades)", text_color=self.text_color, fg_color=self.accent_color, hover_color=self.hover_color)
        for cb in (self.use_uncon, self.use_gd):
            cb.select()
        for cb in (self.use_uncon, self.use_gd, self.use_lagrange, self.use_penalty):
            cb.pack(side="left", padx=5, pady=5)

        # Punto inicial
        self.start_entry = ctk.CTkEntry(
            self, placeholder_text="Punto inicial (ej: 0 0)",
            fg_color=self.entry_color, border_color=self.accent_color, text_color=self.text_color
        )
        self.start_entry.insert(0, "0 0")
        self.start_entry.pack(fill="x", padx=10, pady=5)

        # Botón ejecutar kawaii
        self.run_btn = ctk.CTkButton(
            self, text="💖 Resolver 💖",
            fg_color=self.accent_color, hover_color=self.hover_color,
            text_color="white", corner_radius=25, height=45, font=("Comic Sans MS", 16, "bold")
        )
        self.run_btn.configure(command=self.run_solvers)
        self.run_btn.pack(pady=15)

        # Salida
        self.output = ctk.CTkTextbox(self, fg_color=self.frame_color, text_color=self.text_color, border_color=self.accent_color)
        self.output.pack(fill="both", expand=True, padx=10, pady=10)

        # Set initial placeholder state
        self._obj_focus_out(None)
        self._constr_focus_out(None)

    def _obj_focus_in(self, event):
        if self.obj_entry.cget("text_color") == "gray":
            self.obj_entry.delete("1.0", "end")
            self.obj_entry.configure(text_color=self.text_color)

    def _obj_focus_out(self, event):
        if not self.obj_entry.get("1.0", "end-1c"):
            self.obj_entry.insert("1.0", self.obj_placeholder)
            self.obj_entry.configure(text_color="gray")

    def _constr_focus_in(self, event):
        if self.constr_entry.cget("text_color") == "gray":
            self.constr_entry.delete("1.0", "end")
            self.constr_entry.configure(text_color=self.text_color)

    def _constr_focus_out(self, event):
        if not self.constr_entry.get("1.0", "end-1c"):
            self.constr_entry.insert("1.0", self.constr_placeholder)
            self.constr_entry.configure(text_color="gray")

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
            if self.obj_entry.cget("text_color") == "gray":
                obj_text = ""
            constr_text = self.constr_entry.get("1.0", "end").strip()
            if self.constr_entry.cget("text_color") == "gray":
                constr_text = ""
            problem = FunctionParser.parse_problem(obj_text, vars_text, constr_text)
            start = self._parse_start(len(problem.variables))
            results = []
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

            self.output.insert("end", format_results_table(results) + "\n\n")
            eq_status, ineq_status = evaluate_constraints(problem, results)
            if eq_status:
                self.output.insert("end", "Restricciones de igualdad:\n" + "\n".join(eq_status) + "\n")
            if ineq_status:
                self.output.insert("end", "\nRestricciones de desigualdad:\n" + "\n".join(ineq_status) + "\n")

            for r in results:
                if r.method == 'GradientDescent' and 'history' in r.extra:
                    self.output.insert("end", "\nTrayectoria Gradiente (iter, ||grad||):\n")
                    for it, xvec, gnorm in r.extra['history'][:30]:
                        self.output.insert("end", f"{it}: {gnorm:.3e} {xvec}\n")

            if len(problem.variables) == 2 and results:
                try:
                    from src.visualization.plots import surface_with_point
                    best = min(results, key=lambda r: r.objective_value)
                    bundle = surface_with_point(problem.objective_expr, problem.variables, best.point)
                    tmp_path = os.path.join(os.path.dirname(__file__), 'last_plot.html')
                    bundle.figure.write_html(tmp_path)
                    self.output.insert("end", f"\n🌈 Gráfica guardada: {tmp_path}\n")
                except Exception as e:
                    self.output.insert("end", f"[Plot] Error: {e}\n")
        except Exception as e:
            self.output.insert("end", f"Error: {e}\n")


def run() -> None:
    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    run()
