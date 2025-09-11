from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass

@dataclass
class PlotBundle:
    figure: go.Figure
    description: str


def surface_with_point(objective: sp.Expr, variables: List[sp.Symbol], point: Dict[str, float], 
                      gradient_history: Optional[List[Tuple[int, np.ndarray, float, float]]] = None,
                      constraints: Optional[List[sp.Expr]] = None):
    """Crea una visualización mejorada de la superficie con punto óptimo y trayectoria"""
    if len(variables) != 2:
        raise ValueError("Surface plot only valid for 2 variables")
    
    x_sym, y_sym = variables
    f_lambda = sp.lambdify((x_sym, y_sym), objective, 'numpy')
    x0 = point[str(x_sym)]
    y0 = point[str(y_sym)]

    # Rango adaptativo basado en el punto óptimo
    range_factor = max(3, abs(x0), abs(y0))
    xs = np.linspace(x0 - range_factor, x0 + range_factor, 100)
    ys = np.linspace(y0 - range_factor, y0 + range_factor, 100)
    X, Y = np.meshgrid(xs, ys)
    Z = f_lambda(X, Y)

    # Crear figura con subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'scatter'}]],
        subplot_titles=('Superficie 3D', 'Proyección 2D y Trayectoria'),
        horizontal_spacing=0.1
    )

    # Superficie 3D
    fig.add_trace(
        go.Surface(
            x=xs, y=ys, z=Z, 
            colorscale='Viridis', 
            opacity=0.8,
            name='Función objetivo'
        ),
        row=1, col=1
    )

    # Punto óptimo en 3D
    z0 = f_lambda(x0, y0)
    fig.add_trace(
        go.Scatter3d(
            x=[x0], y=[y0], z=[z0], 
            mode='markers', 
            marker=dict(size=8, color='red', symbol='diamond'),
            name='Punto óptimo',
            showlegend=False
        ),
        row=1, col=1
    )

    # Contour plot 2D
    fig.add_trace(
        go.Contour(
            x=xs, y=ys, z=Z,
            colorscale='Viridis',
            name='Curvas de nivel',
            showscale=False
        ),
        row=1, col=2
    )

    # Punto óptimo en 2D
    fig.add_trace(
        go.Scatter(
            x=[x0], y=[y0],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Punto óptimo',
            showlegend=False
        ),
        row=1, col=2
    )

    # Trayectoria del gradiente si está disponible
    if gradient_history and len(gradient_history) > 0:
        try:
            traj_x = [point[str(x_sym)] for _, point, _, _ in gradient_history]
            traj_y = [point[str(y_sym)] for _, point, _, _ in gradient_history]
        except (IndexError, KeyError, TypeError) as e:
            # Si hay error al acceder al historial, no mostrar trayectoria
            traj_x = []
            traj_y = []
        
        if traj_x and traj_y:  # Solo agregar si hay datos válidos
            fig.add_trace(
                go.Scatter(
                    x=traj_x, y=traj_y,
                    mode='lines+markers',
                    line=dict(color='orange', width=2),
                    marker=dict(size=4, color='orange'),
                    name='Trayectoria del gradiente'
                ),
                row=1, col=2
            )

    # Restricciones si están disponibles
    if constraints:
        for i, constraint in enumerate(constraints):
            if constraint.has(x_sym) and constraint.has(y_sym):
                # Intentar resolver la restricción para y en función de x
                try:
                    y_expr = sp.solve(constraint, y_sym)[0]
                    y_func = sp.lambdify(x_sym, y_expr, 'numpy')
                    x_constraint = np.linspace(xs[0], xs[-1], 50)
                    y_constraint = y_func(x_constraint)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_constraint, y=y_constraint,
                            mode='lines',
                            line=dict(color='blue', width=2, dash='dash'),
                            name=f'Restricción {i+1}',
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                except:
                    pass

    # Actualizar layout
    fig.update_layout(
        title=f'Análisis de Optimización: {str(objective)}',
        scene=dict(
            xaxis_title=str(x_sym),
            yaxis_title=str(y_sym),
            zaxis_title='f(x,y)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=1200,
        height=600
    )

    # Actualizar ejes del subplot 2D
    fig.update_xaxes(title_text=str(x_sym), row=1, col=2)
    fig.update_yaxes(title_text=str(y_sym), row=1, col=2)

    return PlotBundle(fig, 'Análisis completo de optimización')


def gradient_convergence_plot(gradient_history: List[Tuple[int, np.ndarray, float, float]]):
    """Crea un gráfico de convergencia del gradiente"""
    if not gradient_history or len(gradient_history) == 0:
        return None
        
    try:
        iterations = [h[0] for h in gradient_history]
        grad_norms = [h[2] for h in gradient_history]
        obj_values = [h[3] for h in gradient_history]
    except (IndexError, TypeError) as e:
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Convergencia del Gradiente', 'Valor de la Función Objetivo'),
        horizontal_spacing=0.1
    )

    # Gráfico de norma del gradiente
    fig.add_trace(
        go.Scatter(
            x=iterations, y=grad_norms,
            mode='lines+markers',
            name='||∇f||',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Gráfico del valor objetivo
    fig.add_trace(
        go.Scatter(
            x=iterations, y=obj_values,
            mode='lines+markers',
            name='f(x)',
            line=dict(color='red', width=2)
        ),
        row=1, col=2
    )

    fig.update_layout(
        title='Análisis de Convergencia',
        width=1000,
        height=400
    )

    fig.update_xaxes(title_text='Iteración', row=1, col=1)
    fig.update_yaxes(title_text='||∇f||', row=1, col=1)
    fig.update_xaxes(title_text='Iteración', row=1, col=2)
    fig.update_yaxes(title_text='f(x)', row=1, col=2)

    return PlotBundle(fig, 'Análisis de convergencia')


def cylinder_analysis_plot(radius: float, height: float, volume: float, surface_area: float):
    """Crea un gráfico de análisis para problemas de cilindro"""
    # Crear cilindro 3D
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, height, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_cyl = radius * np.cos(theta_grid)
    y_cyl = radius * np.sin(theta_grid)
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]],
        subplot_titles=('Cilindro 3D', 'Comparación de Dimensiones', 
                       'Análisis de Eficiencia', 'Relación Óptima'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # Cilindro 3D
    fig.add_trace(
        go.Surface(
            x=x_cyl, y=y_cyl, z=z_grid,
            colorscale='Blues',
            opacity=0.8,
            name='Cilindro'
        ),
        row=1, col=1
    )

    # Comparación de dimensiones
    dimensions = ['Radio', 'Altura']
    values = [radius, height]
    fig.add_trace(
        go.Bar(x=dimensions, y=values, name='Dimensiones'),
        row=1, col=2
    )

    # Análisis de eficiencia
    optimal_ratio = 2.0
    actual_ratio = height / radius if radius > 0 else 0
    efficiency = optimal_ratio / actual_ratio if actual_ratio > 0 else 0
    
    fig.add_trace(
        go.Scatter(
            x=['Relación Actual', 'Relación Óptima'],
            y=[actual_ratio, optimal_ratio],
            mode='markers',
            marker=dict(size=15, color=['red', 'green']),
            name='Eficiencia'
        ),
        row=2, col=1
    )

    # Métricas
    metrics = ['Volumen', 'Área Superficial']
    metric_values = [volume, surface_area]
    fig.add_trace(
        go.Scatter(
            x=metrics, y=metric_values,
            mode='markers+text',
            marker=dict(size=15, color='blue'),
            text=[f'{v:.2f}' for v in metric_values],
            textposition='top center',
            name='Métricas'
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=f'Análisis del Cilindro (r={radius:.2f}, h={height:.2f})',
        width=1200,
        height=800
    )

    return PlotBundle(fig, 'Análisis completo del cilindro')
