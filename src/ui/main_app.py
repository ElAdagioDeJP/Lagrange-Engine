from __future__ import annotations
import customtkinter as ctk
import threading
import numpy as np
import os, sys
import json

# Ajuste path para ejecución flexible
if 'src' not in {p.split(os.sep)[-1] for p in sys.path}:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.core.function_parser import FunctionParser
from src.core.input_validator import InputValidator
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
        self.geometry("1400x900")
        self.minsize(1200, 700)  # Tamaño mínimo
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Crear notebook para pestañas
        self.notebook = ctk.CTkTabview(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Pestaña 1: Problemas Predefinidos
        self.templates_tab = self.notebook.add("📋 Problemas Predefinidos")
        self._setup_templates_tab()

        # Pestaña 2: Problema Personalizado
        self.custom_tab = self.notebook.add("⚙️ Problema Personalizado")
        self._setup_custom_tab()

        # Pestaña 3: Resultados
        self.results_tab = self.notebook.add("📊 Resultados")
        self._setup_results_tab()

    def _setup_templates_tab(self):
        """Configura la pestaña de problemas predefinidos"""
        # Título
        title = ctk.CTkLabel(self.templates_tab, text="Selecciona un problema predefinido:", 
                           font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=10)

        # Frame para botones de problemas
        problems_frame = ctk.CTkScrollableFrame(self.templates_tab)
        problems_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Problema 1: Sin restricciones
        prob1_frame = ctk.CTkFrame(problems_frame)
        prob1_frame.pack(fill="x", pady=5)
        prob1_title = ctk.CTkLabel(prob1_frame, text="1. Problema sin restricciones (Minimización)", 
                                 font=ctk.CTkFont(size=14, weight="bold"))
        prob1_title.pack(anchor="w", padx=10, pady=5)
        prob1_desc = ctk.CTkLabel(prob1_frame, 
                                text="C(x,y) = x² + y² - 4x - 6y + 20\nEncontrar valores que minimizan el costo")
        prob1_desc.pack(anchor="w", padx=10, pady=5)
        prob1_btn = ctk.CTkButton(prob1_frame, text="Resolver", 
                                command=lambda: self._load_template(1))
        prob1_btn.pack(anchor="e", padx=10, pady=5)

        # Problema 2: Lagrange
        prob2_frame = ctk.CTkFrame(problems_frame)
        prob2_frame.pack(fill="x", pady=5)
        prob2_title = ctk.CTkLabel(prob2_frame, text="2. Método de Lagrange (Maximización)", 
                                 font=ctk.CTkFont(size=14, weight="bold"))
        prob2_title.pack(anchor="w", padx=10, pady=5)
        prob2_desc = ctk.CTkLabel(prob2_frame, 
                                text="A(x,y) = x * y\nRestricción: 2x + 2y = 40\nRectángulo de máxima área")
        prob2_desc.pack(anchor="w", padx=10, pady=5)
        prob2_btn = ctk.CTkButton(prob2_frame, text="Resolver", 
                                command=lambda: self._load_template(2))
        prob2_btn.pack(anchor="e", padx=10, pady=5)

        # Problema 3: Gradiente
        prob3_frame = ctk.CTkFrame(problems_frame)
        prob3_frame.pack(fill="x", pady=5)
        prob3_title = ctk.CTkLabel(prob3_frame, text="3. Método del Gradiente (Maximización)", 
                                 font=ctk.CTkFont(size=14, weight="bold"))
        prob3_title.pack(anchor="w", padx=10, pady=5)
        prob3_desc = ctk.CTkLabel(prob3_frame, 
                                text="f(x,y) = -(x² + y²) + 4x + 6y\nPunto inicial: (0,0)")
        prob3_desc.pack(anchor="w", padx=10, pady=5)
        prob3_btn = ctk.CTkButton(prob3_frame, text="Resolver", 
                                command=lambda: self._load_template(3))
        prob3_btn.pack(anchor="e", padx=10, pady=5)

        # Problema 4: Múltiples restricciones
        prob4_frame = ctk.CTkFrame(problems_frame)
        prob4_frame.pack(fill="x", pady=5)
        prob4_title = ctk.CTkLabel(prob4_frame, text="4. Múltiples restricciones (Maximización)", 
                                 font=ctk.CTkFont(size=14, weight="bold"))
        prob4_title.pack(anchor="w", padx=10, pady=5)
        prob4_desc = ctk.CTkLabel(prob4_frame, 
                                text="G(x,y) = 5x + 8y - 0.1x² - 0.2y²\nRestricciones: x + y ≤ 40, x ≥ 0, y ≥ 0")
        prob4_desc.pack(anchor="w", padx=10, pady=5)
        prob4_btn = ctk.CTkButton(prob4_frame, text="Resolver", 
                                command=lambda: self._load_template(4))
        prob4_btn.pack(anchor="e", padx=10, pady=5)

        # Problema 5: Volumen mínimo
        prob5_frame = ctk.CTkFrame(problems_frame)
        prob5_frame.pack(fill="x", pady=5)
        prob5_title = ctk.CTkLabel(prob5_frame, text="5. Volumen mínimo (Cilindro)", 
                                 font=ctk.CTkFont(size=14, weight="bold"))
        prob5_title.pack(anchor="w", padx=10, pady=5)
        prob5_desc = ctk.CTkLabel(prob5_frame, 
                                text="Volumen fijo: V = 500 cm³\nMinimizar área superficial del cilindro")
        prob5_desc.pack(anchor="w", padx=10, pady=5)
        prob5_btn = ctk.CTkButton(prob5_frame, text="Resolver", 
                                command=lambda: self._load_template(5))
        prob5_btn.pack(anchor="e", padx=10, pady=5)

    def _setup_custom_tab(self):
        """Configura la pestaña de problema personalizado"""
        # Crear frame principal con scroll
        main_frame = ctk.CTkScrollableFrame(self.custom_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Variables
        var_label = ctk.CTkLabel(main_frame, text="Variables (separadas por espacios):", 
                               font=ctk.CTkFont(size=12, weight="bold"))
        var_label.pack(anchor="w", padx=10, pady=(10,5))
        self.var_entry = ctk.CTkEntry(main_frame, placeholder_text="x y")
        self.var_entry.pack(fill="x", padx=10, pady=5)

        # Función objetivo
        obj_label = ctk.CTkLabel(main_frame, text="Función objetivo:", 
                               font=ctk.CTkFont(size=12, weight="bold"))
        obj_label.pack(anchor="w", padx=10, pady=(10,5))
        self.obj_entry = ctk.CTkTextbox(main_frame, height=80)
        self.obj_entry.insert("1.0", "x**2 + y**2")
        self.obj_entry.pack(fill="x", padx=10, pady=5)

        # Restricciones
        constr_label = ctk.CTkLabel(main_frame, text="Restricciones (una por línea):", 
                                  font=ctk.CTkFont(size=12, weight="bold"))
        constr_label.pack(anchor="w", padx=10, pady=(10,5))
        self.constr_entry = ctk.CTkTextbox(main_frame, height=120)
        self.constr_entry.insert("1.0", "# Ejemplos:\n# x + y = 10\n# x >= 0\n# y <= 5")
        self.constr_entry.pack(fill="x", padx=10, pady=5)

        # Tipo de optimización
        opt_frame = ctk.CTkFrame(main_frame)
        opt_frame.pack(fill="x", padx=10, pady=10)
        opt_label = ctk.CTkLabel(opt_frame, text="Tipo de optimización:", 
                               font=ctk.CTkFont(size=12, weight="bold"))
        opt_label.pack(anchor="w", padx=10, pady=5)
        self.optimization_type = ctk.CTkSegmentedButton(opt_frame, 
                                                      values=["Minimizar", "Maximizar"])
        self.optimization_type.set("Minimizar")
        self.optimization_type.pack(anchor="w", padx=10, pady=5)

        # Frame métodos
        method_label = ctk.CTkLabel(main_frame, text="Métodos de resolución:", 
                                  font=ctk.CTkFont(size=12, weight="bold"))
        method_label.pack(anchor="w", padx=10, pady=(10,5))
        self.method_frame = ctk.CTkFrame(main_frame)
        self.method_frame.pack(fill="x", padx=10, pady=5)
        self.use_uncon = ctk.CTkCheckBox(self.method_frame, text="Sin Restricciones (BFGS)")
        self.use_gd = ctk.CTkCheckBox(self.method_frame, text="Gradiente")
        self.use_lagrange = ctk.CTkCheckBox(self.method_frame, text="Lagrange (igualdades)")
        self.use_penalty = ctk.CTkCheckBox(self.method_frame, text="Penalización (desigualdades)")
        for cb in (self.use_uncon, self.use_gd):
            cb.select()
        for cb in (self.use_uncon, self.use_gd, self.use_lagrange, self.use_penalty):
            cb.pack(side="left", padx=5)

        # Punto inicial
        start_label = ctk.CTkLabel(main_frame, text="Punto inicial:", 
                                 font=ctk.CTkFont(size=12, weight="bold"))
        start_label.pack(anchor="w", padx=10, pady=(10,5))
        self.start_entry = ctk.CTkEntry(main_frame, placeholder_text="0 0")
        self.start_entry.insert(0, "0 0")
        self.start_entry.pack(fill="x", padx=10, pady=5)

        # Parámetros avanzados
        params_label = ctk.CTkLabel(main_frame, text="Parámetros avanzados (opcional):", 
                                  font=ctk.CTkFont(size=12, weight="bold"))
        params_label.pack(anchor="w", padx=10, pady=(10,5))
        self.params_entry = ctk.CTkTextbox(main_frame, height=80)
        example = '{"learning_rate": 0.1, "max_iter": 200, "tol": 1e-6}'
        self.params_entry.insert("1.0", example)
        self.params_entry.pack(fill="x", padx=10, pady=5)

        # Frame para botones de acción - FIJADO EN LA PARTE INFERIOR
        action_frame = ctk.CTkFrame(self.custom_tab)
        action_frame.pack(side="bottom", fill="x", padx=10, pady=10)
        
        # Botón resolver (más prominente y visible)
        self.run_btn = ctk.CTkButton(
            action_frame, 
            text="🚀 CALCULAR Y VER RESULTADOS", 
            command=self.run_solvers, 
            height=60,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color=("#28a745", "#1e7e34"),
            hover_color=("#1e7e34", "#28a745")
        )
        self.run_btn.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        # Botón para limpiar formulario
        self.clear_btn = ctk.CTkButton(
            action_frame, 
            text="🗑️ Limpiar", 
            command=self.clear_form,
            height=60,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=("#dc3545", "#c82333"),
            hover_color=("#c82333", "#dc3545")
        )
        self.clear_btn.pack(side="right", padx=10, pady=10)
        
        # Información adicional
        info_label = ctk.CTkLabel(
            action_frame, 
            text="💡 Los resultados aparecerán automáticamente en la pestaña 'Resultados'",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#17a2b8"
        )
        info_label.pack(pady=5)

    def _setup_results_tab(self):
        """Configura la pestaña de resultados"""
        # Título de la pestaña
        title_label = ctk.CTkLabel(
            self.results_tab, 
            text="📊 Resultados de Optimización", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=10)
        
        # Frame para botones de control
        control_frame = ctk.CTkFrame(self.results_tab)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Botón para limpiar resultados
        clear_btn = ctk.CTkButton(
            control_frame, 
            text="🗑️ Limpiar Resultados", 
            command=self.clear_results,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        clear_btn.pack(side="left", padx=5, pady=5)
        
        # Botón para exportar resultados
        export_btn = ctk.CTkButton(
            control_frame, 
            text="💾 Exportar", 
            command=self.export_results,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        export_btn.pack(side="right", padx=5, pady=5)
        
        # Área de resultados
        self.output = ctk.CTkTextbox(
            self.results_tab, 
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word"
        )
        self.output.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Mensaje inicial
        self.output.insert("1.0", "🎯 Bienvenido al Optimizador Lagrange Engine\n\n")
        self.output.insert("end", "Para comenzar:\n")
        self.output.insert("end", "1. Ve a la pestaña 'Problemas Predefinidos' y selecciona un problema\n")
        self.output.insert("end", "2. O ve a 'Problema Personalizado' y define tu propio problema\n")
        self.output.insert("end", "3. Haz clic en 'CALCULAR Y VER RESULTADOS'\n")
        self.output.insert("end", "4. Los resultados aparecerán aquí automáticamente\n\n")
        self.output.insert("end", "Los resultados incluirán:\n")
        self.output.insert("end", "• Punto óptimo encontrado\n")
        self.output.insert("end", "• Valor de la función objetivo\n")
        self.output.insert("end", "• Información de convergencia\n")
        self.output.insert("end", "• Gráficas 3D interactivas (si aplica)\n")
        self.output.insert("end", "• Análisis de restricciones\n")

    def _load_template(self, template_num: int) -> None:
        """Carga un problema predefinido en la pestaña personalizada"""
        self.notebook.set("⚙️ Problema Personalizado")
        
        templates = {
            1: {
                "vars": "x y",
                "obj": "x**2 + y**2 - 4*x - 6*y + 20",
                "constr": "",
                "start": "0 0",
                "opt_type": "Minimizar"
            },
            2: {
                "vars": "x y",
                "obj": "x * y",
                "constr": "2*x + 2*y = 40",
                "start": "10 10",
                "opt_type": "Maximizar"
            },
            3: {
                "vars": "x y",
                "obj": "-(x**2 + y**2) + 4*x + 6*y",
                "constr": "",
                "start": "0 0",
                "opt_type": "Maximizar"
            },
            4: {
                "vars": "x y",
                "obj": "5*x + 8*y - 0.1*x**2 - 0.2*y**2",
                "constr": "x + y <= 40\nx >= 0\ny >= 0",
                "start": "0 0",
                "opt_type": "Maximizar"
            },
            5: {
                "vars": "r h",
                "obj": "2*pi*r**2 + 2*pi*r*h",
                "constr": "pi*r**2*h = 500",
                "start": "5 10",
                "opt_type": "Minimizar"
            }
        }
        
        if template_num in templates:
            template = templates[template_num]
            self.var_entry.delete(0, "end")
            self.var_entry.insert(0, template["vars"])
            self.obj_entry.delete("1.0", "end")
            self.obj_entry.insert("1.0", template["obj"])
            self.constr_entry.delete("1.0", "end")
            self.constr_entry.insert("1.0", template["constr"])
            self.start_entry.delete(0, "end")
            self.start_entry.insert(0, template["start"])
            self.optimization_type.set(template["opt_type"])
            
            # Configurar métodos según el problema
            for cb in (self.use_uncon, self.use_gd, self.use_lagrange, self.use_penalty):
                cb.deselect()
            
            if template_num == 1:  # Sin restricciones
                self.use_uncon.select()
                self.use_gd.select()
            elif template_num == 2:  # Lagrange
                self.use_lagrange.select()
            elif template_num == 3:  # Gradiente
                self.use_gd.select()
            elif template_num == 4:  # Múltiples restricciones
                self.use_penalty.select()
            elif template_num == 5:  # Volumen mínimo
                self.use_lagrange.select()

    def clear_form(self) -> None:
        """Limpia todos los campos del formulario personalizado"""
        self.var_entry.delete(0, "end")
        self.obj_entry.delete("1.0", "end")
        self.constr_entry.delete("1.0", "end")
        self.start_entry.delete(0, "end")
        self.start_entry.insert(0, "0 0")
        self.params_entry.delete("1.0", "end")
        example = '{"learning_rate": 0.1, "max_iter": 200, "tol": 1e-6}'
        self.params_entry.insert("1.0", example)
        self.optimization_type.set("Minimizar")
        
        # Deseleccionar todos los métodos
        for cb in (self.use_uncon, self.use_gd, self.use_lagrange, self.use_penalty):
            cb.deselect()
        
        # Seleccionar métodos por defecto
        self.use_uncon.select()
        self.use_gd.select()

    def clear_results(self) -> None:
        """Limpia el área de resultados"""
        self.output.delete("1.0", "end")
        self.output.insert("1.0", "🎯 Bienvenido al Optimizador Lagrange Engine\n\n")
        self.output.insert("end", "Para comenzar:\n")
        self.output.insert("end", "1. Ve a la pestaña 'Problemas Predefinidos' y selecciona un problema\n")
        self.output.insert("end", "2. O ve a 'Problema Personalizado' y define tu propio problema\n")
        self.output.insert("end", "3. Haz clic en 'CALCULAR Y VER RESULTADOS'\n")
        self.output.insert("end", "4. Los resultados aparecerán aquí automáticamente\n\n")
        self.output.insert("end", "Los resultados incluirán:\n")
        self.output.insert("end", "• Punto óptimo encontrado\n")
        self.output.insert("end", "• Valor de la función objetivo\n")
        self.output.insert("end", "• Información de convergencia\n")
        self.output.insert("end", "• Gráficas 3D interactivas (si aplica)\n")
        self.output.insert("end", "• Análisis de restricciones\n")

    def export_results(self) -> None:
        """Exporta los resultados a un archivo de texto"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")],
                title="Guardar resultados"
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.output.get("1.0", "end"))
                self.output.insert("end", f"\n✅ Resultados exportados a: {filename}\n")
        except Exception as e:
            self.output.insert("end", f"\n❌ Error al exportar: {e}\n")

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

    def _parse_solver_params(self, raw: str) -> dict:
        raw = (raw or '').strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            # Fallback: parse simple key=value lines
            params = {}
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip()
                    if v.lower() in ('true', 'false'):
                        params[k] = v.lower() == 'true'
                    else:
                        try:
                            if '.' in v or 'e' in v.lower():
                                params[k] = float(v)
                            else:
                                params[k] = int(v)
                        except Exception:
                            params[k] = v
            return params

    def _run_solvers_impl(self) -> None:
        # Cambiar a la pestaña de resultados
        self.notebook.set("📊 Resultados")
        self.output.delete("1.0", "end")
        
        try:
            vars_text = self.var_entry.get().strip()
            obj_text = self.obj_entry.get("1.0", "end").strip()
            constr_text = self.constr_entry.get("1.0", "end").strip()
            params_text = self.params_entry.get("1.0", "end").strip()
            
            # Validar entrada completa
            self.output.insert("end", "🔍 Validando entrada...\n")
            valid, msg, validation_results = InputValidator.validate_complete_problem(
                vars_text, obj_text, constr_text, 
                self.start_entry.get().strip(), params_text
            )
            
            if not valid:
                self.output.insert("end", f"❌ {msg}\n")
                return
            
            self.output.insert("end", f"✅ {msg}\n\n")
            
            solver_params = validation_results['parameters']['data']
            
            # Determinar si es maximización o minimización
            is_maximization = self.optimization_type.get() == "Maximizar"
            
            # Si es maximización, negar la función objetivo
            if is_maximization:
                obj_text = f"-({obj_text})"
            
            problem = FunctionParser.parse_problem(obj_text, vars_text, constr_text)
            
            # initial point: prefer JSON 'initial' if provided, else UI field
            if 'initial' in solver_params:
                init = solver_params['initial']
                if isinstance(init, dict):
                    # map by variable name
                    vals = [float(init.get(str(v), 0.0)) for v in problem.variables]
                else:
                    vals = [float(x) for x in init]
                start = np.array(vals, dtype=float)
            else:
                start = self._parse_start(len(problem.variables))
            
            results = []
            
            # Ejecutar métodos seleccionados
            if self.use_uncon.get() == 1:
                uncon_kwargs = {}
                if 'method' in solver_params:
                    uncon_kwargs['method'] = solver_params['method']
                if 'tol' in solver_params:
                    uncon_kwargs['tol'] = solver_params['tol']
                result = UnconstrainedMinimizer(problem.objective_expr, problem.variables, **uncon_kwargs).solve(start)
                if is_maximization:
                    result.objective_value = -result.objective_value
                results.append(result)
                
            if self.use_gd.get() == 1:
                gd_kwargs = {}
                if 'learning_rate' in solver_params:
                    gd_kwargs['learning_rate'] = solver_params['learning_rate']
                elif 'step_size' in solver_params:
                    gd_kwargs['learning_rate'] = solver_params['step_size']
                if 'max_iter' in solver_params:
                    gd_kwargs['max_iter'] = int(solver_params['max_iter'])
                if 'tol' in solver_params:
                    gd_kwargs['tol'] = float(solver_params['tol'])
                result = GradientDescentSolver(problem.objective_expr, problem.variables, **gd_kwargs).solve(start)
                if is_maximization:
                    result.objective_value = -result.objective_value
                results.append(result)
                
            if self.use_lagrange.get() == 1 and problem.equality_constraints:
                try:
                    result = LagrangeSolver(problem.objective_expr, problem.variables, problem.equality_constraints).solve(start)
                    if is_maximization:
                        result.objective_value = -result.objective_value
                    results.append(result)
                except Exception as e:
                    self.output.insert("end", f"[Lagrange] Error: {e}\n")
                    
            if self.use_penalty.get() == 1 and problem.inequality_constraints:
                pen_kwargs = {}
                if 'rho0' in solver_params:
                    pen_kwargs['rho0'] = float(solver_params['rho0'])
                if 'rho_factor' in solver_params:
                    pen_kwargs['rho_factor'] = float(solver_params['rho_factor'])
                if 'tol_cons' in solver_params:
                    pen_kwargs['tol_cons'] = float(solver_params['tol_cons'])
                if 'max_outer' in solver_params:
                    pen_kwargs['max_outer'] = int(solver_params['max_outer'])
                result = PenaltyInequalitySolver(problem.objective_expr, problem.variables, problem.inequality_constraints, **pen_kwargs).solve(start)
                if is_maximization:
                    result.objective_value = -result.objective_value
                results.append(result)

            # Mostrar resultados
            self.output.insert("end", f"🎯 PROBLEMA: {'MAXIMIZACIÓN' if is_maximization else 'MINIMIZACIÓN'}\n")
            self.output.insert("end", f"📊 Función objetivo: {self.obj_entry.get('1.0', 'end').strip()}\n")
            if constr_text.strip():
                self.output.insert("end", f"🔒 Restricciones:\n{constr_text}\n")
            self.output.insert("end", f"🚀 Punto inicial: {self.start_entry.get()}\n\n")
            
            self.output.insert("end", format_results_table(results) + "\n\n")
            
            eq_status, ineq_status = evaluate_constraints(problem, results)
            if eq_status:
                self.output.insert("end", "✅ Restricciones de igualdad:\n" + "\n".join(eq_status) + "\n")
            if ineq_status:
                self.output.insert("end", "\n✅ Restricciones de desigualdad:\n" + "\n".join(ineq_status) + "\n")

            # Trayectoria gradiente
            for r in results:
                if r.method == 'GradientDescent' and 'history' in r.extra:
                    self.output.insert("end", "\n📈 Trayectoria del Gradiente (iteración, ||gradiente||, f(x)):\n")
                    for it, xvec, gnorm, obj_val in r.extra['history'][:30]:
                        self.output.insert("end", f"  {it:3d}: ||∇f||={gnorm:.3e}, f(x)={obj_val:.6f} → {xvec}\n")

            # Plot si 2 variables
            if len(problem.variables) == 2 and results:
                try:
                    from src.visualization.plots import surface_with_point, gradient_convergence_plot, cylinder_analysis_plot
                    best = min(results, key=lambda r: r.objective_value)
                    
                    # Obtener historial del gradiente si está disponible
                    gradient_history = None
                    for r in results:
                        if r.method == 'GradientDescent' and 'history' in r.extra:
                            gradient_history = r.extra['history']
                            break
                    
                    # Crear visualización principal
                    bundle = surface_with_point(
                        problem.objective_expr, 
                        problem.variables, 
                        best.point,
                        gradient_history=gradient_history,
                        constraints=problem.equality_constraints + problem.inequality_constraints
                    )
                    
                    # Guardar gráfica principal
                    tmp_path = os.path.join(os.path.dirname(__file__), 'last_plot.html')
                    bundle.figure.write_html(tmp_path)
                    self.output.insert("end", f"\n📊 Gráfica principal guardada: {tmp_path}\n")
                    
                    # Crear gráfica de convergencia si hay historial
                    if gradient_history:
                        conv_bundle = gradient_convergence_plot(gradient_history)
                        conv_path = os.path.join(os.path.dirname(__file__), 'convergence_plot.html')
                        conv_bundle.figure.write_html(conv_path)
                        self.output.insert("end", f"📈 Gráfica de convergencia: {conv_path}\n")
                    
                    # Análisis especial para cilindros
                    if len(problem.variables) == 2 and 'r' in [str(v) for v in problem.variables] and 'h' in [str(v) for v in problem.variables]:
                        try:
                            r_val = best.point.get('r', 0)
                            h_val = best.point.get('h', 0)
                            volume = float(problem.objective_expr.subs({problem.variables[0]: r_val, problem.variables[1]: h_val}))
                            surface_area = 2 * 3.14159 * r_val**2 + 2 * 3.14159 * r_val * h_val
                            
                            cyl_bundle = cylinder_analysis_plot(r_val, h_val, volume, surface_area)
                            cyl_path = os.path.join(os.path.dirname(__file__), 'cylinder_analysis.html')
                            cyl_bundle.figure.write_html(cyl_path)
                            self.output.insert("end", f"🔧 Análisis del cilindro: {cyl_path}\n")
                        except:
                            pass
                            
                except Exception as e:  # pragma: no cover
                    self.output.insert("end", f"[Plot] Error: {e}\n")
                    
        except Exception as e:  # pragma: no cover
            self.output.insert("end", f"❌ Error: {e}\n")


def run() -> None:
    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    run()
