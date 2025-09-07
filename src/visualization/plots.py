from __future__ import annotations
from typing import List, Dict
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from dataclasses import dataclass

@dataclass
class PlotBundle:
    figure: go.Figure
    description: str


def surface_with_point(objective: sp.Expr, variables: List[sp.Symbol], point: Dict[str, float]):
    if len(variables) != 2:
        raise ValueError("Surface plot only valid for 2 variables")
    x_sym, y_sym = variables
    f_lambda = sp.lambdify((x_sym, y_sym), objective, 'numpy')
    x0 = point[str(x_sym)]
    y0 = point[str(y_sym)]

    xs = np.linspace(x0 - 5, x0 + 5, 80)
    ys = np.linspace(y0 - 5, y0 + 5, 80)
    X, Y = np.meshgrid(xs, ys)
    Z = f_lambda(X, Y)

    fig = go.Figure(data=[go.Surface(x=xs, y=ys, z=Z, colorscale='Viridis', opacity=0.85)])
    z0 = f_lambda(x0, y0)
    fig.add_trace(go.Scatter3d(x=[x0], y=[y0], z=[z0], mode='markers', marker=dict(size=6, color='red'), name='Óptimo'))
    fig.update_layout(title='Superficie y punto óptimo', scene=dict(xaxis_title=str(x_sym), yaxis_title=str(y_sym), zaxis_title='f'))
    return PlotBundle(fig, 'Superficie 3D con punto óptimo')
