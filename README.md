# Lagrange Engine

Calculadora de optimización no lineal avanzada en Python (POO). Resuelve funciones con y sin restricciones usando métodos como Descenso por Gradiente y Multiplicadores de Lagrange. Incluye visualización en 2D/3D e interfaz moderna.

## 🚀 Visión

Herramienta educativa y profesional para experimentar con problemas de optimización multivariable, diseñada con arquitectura limpia, modular y extensible.

## 🧱 Arquitectura (POO)

```text
src/
	core/
		function_parser.py      # Parser de funciones y restricciones (SymPy)
	solvers/
		base.py                 # Clases base y contratos
		gradient_descent.py     # Implementación iterativa
		lagrange.py             # Método de multiplicadores de Lagrange
		unconstrained.py        # Minimización sin restricciones (SciPy)
	visualization/
		plots.py                # Gráficas interactivas Plotly
	ui/
		main_app.py             # Interfaz principal (CustomTkinter)
	tests/
		test_parser.py          # Prueba básica del parser
```

### Componentes Clave

- `FunctionParser`: Convierte texto en expresiones simbólicas (`sympy.Expr`), variables y listas de restricciones (igualdad / desigualdad).
- `OptimizationEngine`: Clase abstracta que define la interfaz `solve` y prepara la función objetivo vectorizada.
- `GradientDescentSolver`: Implementa un descenso por gradiente básico configurable.
- `LagrangeSolver`: Construye el Lagrangiano y resuelve el sistema estacionario mediante `sympy.nsolve`.
- `UnconstrainedMinimizer`: Envoltorio sobre `scipy.optimize.minimize` para soluciones rápidas.
- `visualization.plots`: Genera superficies 3D y marca el punto óptimo.
- `MainApp`: Orquesta la UI, captura entrada del usuario, ejecuta métodos en background y muestra resultados.

## 📦 Dependencias

Ver `requirements.txt`.

Instalación rápida:

```bash
pip install -r requirements.txt
```

## ▶️ Ejecución

```bash
python -m src.ui.main_app
```

## ✏️ Ejemplo de Uso

1. Variables: `x y`
1. Función objetivo: `x**2 + y**2 + 3*x - 2*y`
1. Restricciones (opcional):

```text
x + y = 0
```

1. Selecciona métodos y pulsa "Resolver".

## 📊 Resultados

Se listan por método:

- Punto óptimo aproximado
- Valor de la función
- Estado de convergencia
- Iteraciones / información extra

## 🧮 Fundamento Matemático (Resumen)

### Descenso por Gradiente

Iteración: `x_{k+1} = x_k - α ∇f(x_k)` hasta `||∇f|| < tol`.

### Multiplicadores de Lagrange

Problema: minimizar `f(x)` s.a. `g_i(x)=0`.
Se construye: `L(x, λ) = f(x) + Σ λ_i g_i(x)`.
Condición estacionaria: `∇_x L = 0`, `g_i(x)=0` → sistema no lineal resuelto con `nsolve`.

## 🧪 Tests

Ejecutar (si tienes pytest instalado):

```bash
pytest -q
```

## 🔜 Roadmap Sugerido

- Soporte para restricciones de desigualdad vía KKT.
- Métodos adicionales: Newton, quasi-Newton (BFGS), Penalty / Barrier.
- Visualización combinada de trayectorias.
- Validación y resaltado de sintaxis avanzado.
- Exportar reporte en PDF / HTML.

## 📄 Licencia

Ver `LICENSE`.

---

Proyecto en construcción inicial. Contribuciones y sugerencias son bienvenidas.
