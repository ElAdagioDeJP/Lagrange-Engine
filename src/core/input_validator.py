from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import sympy as sp
import numpy as np
import re


class InputValidator:
    """Validador de entrada para problemas de optimización"""
    
    @staticmethod
    def validate_variables(var_text: str) -> Tuple[bool, str, List[str]]:
        """Valida el texto de variables"""
        if not var_text.strip():
            return False, "No se proporcionaron variables", []
        
        # Limpiar y separar variables
        var_names = [v.strip() for v in var_text.replace(',', ' ').split() if v.strip()]
        
        if not var_names:
            return False, "No se encontraron variables válidas", []
        
        # Validar nombres de variables
        for var in var_names:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var):
                return False, f"Nombre de variable inválido: '{var}'", []
        
        # Verificar duplicados
        if len(var_names) != len(set(var_names)):
            return False, "Variables duplicadas encontradas", []
        
        return True, "Variables válidas", var_names
    
    @staticmethod
    def validate_objective(obj_text: str, variables: List[str]) -> Tuple[bool, str, str]:
        """Valida la función objetivo"""
        if not obj_text.strip():
            return False, "No se proporcionó función objetivo", ""
        
        try:
            # Crear diccionario local con variables y funciones de SymPy
            local_dict = {var: sp.Symbol(var) for var in variables}
            local_dict.update({name: getattr(sp, name) for name in dir(sp) if not name.startswith('_')})
            
            # Intentar parsear la expresión
            expr = sp.sympify(obj_text, locals=local_dict)
            
            # Verificar que todas las variables estén presentes
            used_vars = [str(s) for s in expr.free_symbols if str(s) in variables]
            missing_vars = set(variables) - set(used_vars)
            
            if missing_vars:
                return False, f"Variables no utilizadas en la función objetivo: {missing_vars}", ""
            
            return True, "Función objetivo válida", str(expr)
            
        except Exception as e:
            return False, f"Error en función objetivo: {str(e)}", ""
    
    @staticmethod
    def validate_constraints(constr_text: str, variables: List[str]) -> Tuple[bool, str, List[str]]:
        """Valida las restricciones"""
        if not constr_text.strip():
            return True, "Sin restricciones", []
        
        lines = [line.strip() for line in constr_text.splitlines() if line.strip() and not line.strip().startswith('#')]
        
        if not lines:
            return True, "Sin restricciones válidas", []
        
        valid_constraints = []
        errors = []
        
        for i, line in enumerate(lines, 1):
            try:
                # Crear diccionario local
                local_dict = {var: sp.Symbol(var) for var in variables}
                local_dict.update({name: getattr(sp, name) for name in dir(sp) if not name.startswith('_')})
                
                # Verificar que la línea contenga operadores de comparación
                if not any(op in line for op in ['=', '<', '>', '<=', '>=', '==']):
                    errors.append(f"Línea {i}: No se encontró operador de comparación")
                    continue
                
                # Intentar parsear la restricción
                if '==' in line:
                    left, right = line.split('==', 1)
                    left_expr = sp.sympify(left.strip(), locals=local_dict)
                    right_expr = sp.sympify(right.strip(), locals=local_dict)
                elif '=' in line and '<' not in line and '>' not in line:
                    left, right = line.split('=', 1)
                    left_expr = sp.sympify(left.strip(), locals=local_dict)
                    right_expr = sp.sympify(right.strip(), locals=local_dict)
                elif '<=' in line:
                    left, right = line.split('<=', 1)
                    left_expr = sp.sympify(left.strip(), locals=local_dict)
                    right_expr = sp.sympify(right.strip(), locals=local_dict)
                elif '>=' in line:
                    left, right = line.split('>=', 1)
                    left_expr = sp.sympify(left.strip(), locals=local_dict)
                    right_expr = sp.sympify(right.strip(), locals=local_dict)
                elif '<' in line:
                    left, right = line.split('<', 1)
                    left_expr = sp.sympify(left.strip(), locals=local_dict)
                    right_expr = sp.sympify(right.strip(), locals=local_dict)
                elif '>' in line:
                    left, right = line.split('>', 1)
                    left_expr = sp.sympify(left.strip(), locals=local_dict)
                    right_expr = sp.sympify(right.strip(), locals=local_dict)
                else:
                    errors.append(f"Línea {i}: Formato de restricción no reconocido")
                    continue
                
                valid_constraints.append(line)
                
            except Exception as e:
                errors.append(f"Línea {i}: {str(e)}")
        
        if errors:
            return False, f"Errores en restricciones: {'; '.join(errors)}", valid_constraints
        
        return True, f"{len(valid_constraints)} restricciones válidas", valid_constraints
    
    @staticmethod
    def validate_start_point(start_text: str, num_vars: int) -> Tuple[bool, str, np.ndarray]:
        """Valida el punto inicial"""
        if not start_text.strip():
            return True, "Punto inicial por defecto (ceros)", np.zeros(num_vars)
        
        try:
            # Parsear valores
            vals = [float(p) for p in start_text.replace(',', ' ').split() if p.strip()]
            
            if len(vals) != num_vars:
                return False, f"Se esperaban {num_vars} valores, se encontraron {len(vals)}", np.array([])
            
            # Verificar que no sean NaN o infinito
            if any(not np.isfinite(v) for v in vals):
                return False, "El punto inicial contiene valores no finitos", np.array([])
            
            return True, "Punto inicial válido", np.array(vals, dtype=float)
            
        except ValueError as e:
            return False, f"Error al parsear punto inicial: {str(e)}", np.array([])
    
    @staticmethod
    def validate_solver_params(params_text: str) -> Tuple[bool, str, Dict]:
        """Valida los parámetros del solver"""
        if not params_text.strip():
            return True, "Parámetros por defecto", {}
        
        try:
            import json
            params = json.loads(params_text)
            
            # Validar parámetros conocidos
            valid_params = {}
            for key, value in params.items():
                if key in ['learning_rate', 'step_size', 'tol', 'rho0', 'rho_factor', 'tol_cons']:
                    if not isinstance(value, (int, float)) or not np.isfinite(value):
                        return False, f"Parámetro '{key}' debe ser un número finito", {}
                    valid_params[key] = float(value)
                elif key in ['max_iter', 'max_outer']:
                    if not isinstance(value, int) or value <= 0:
                        return False, f"Parámetro '{key}' debe ser un entero positivo", {}
                    valid_params[key] = int(value)
                elif key in ['method']:
                    if not isinstance(value, str):
                        return False, f"Parámetro '{key}' debe ser una cadena", {}
                    valid_params[key] = str(value)
                elif key in ['initial']:
                    if isinstance(value, list):
                        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in value):
                            return False, f"Parámetro '{key}' debe ser una lista de números finitos", {}
                        valid_params[key] = [float(v) for v in value]
                    elif isinstance(value, dict):
                        valid_params[key] = value
                    else:
                        return False, f"Parámetro '{key}' debe ser una lista o diccionario", {}
                else:
                    # Parámetro desconocido, pero lo aceptamos
                    valid_params[key] = value
            
            return True, "Parámetros válidos", valid_params
            
        except json.JSONDecodeError as e:
            return False, f"Error en formato JSON: {str(e)}", {}
        except Exception as e:
            return False, f"Error en parámetros: {str(e)}", {}
    
    @classmethod
    def validate_complete_problem(cls, vars_text: str, obj_text: str, constr_text: str, 
                                start_text: str, params_text: str) -> Tuple[bool, str, Dict]:
        """Valida un problema completo"""
        results = {}
        
        # Validar variables
        valid, msg, vars_list = cls.validate_variables(vars_text)
        results['variables'] = {'valid': valid, 'message': msg, 'data': vars_list}
        if not valid:
            return False, f"Error en variables: {msg}", results
        
        # Validar función objetivo
        valid, msg, obj_expr = cls.validate_objective(obj_text, vars_list)
        results['objective'] = {'valid': valid, 'message': msg, 'data': obj_expr}
        if not valid:
            return False, f"Error en función objetivo: {msg}", results
        
        # Validar restricciones
        valid, msg, constr_list = cls.validate_constraints(constr_text, vars_list)
        results['constraints'] = {'valid': valid, 'message': msg, 'data': constr_list}
        if not valid:
            return False, f"Error en restricciones: {msg}", results
        
        # Validar punto inicial
        valid, msg, start_point = cls.validate_start_point(start_text, len(vars_list))
        results['start_point'] = {'valid': valid, 'message': msg, 'data': start_point}
        if not valid:
            return False, f"Error en punto inicial: {msg}", results
        
        # Validar parámetros
        valid, msg, params = cls.validate_solver_params(params_text)
        results['parameters'] = {'valid': valid, 'message': msg, 'data': params}
        if not valid:
            return False, f"Error en parámetros: {msg}", results
        
        return True, "Problema válido", results
