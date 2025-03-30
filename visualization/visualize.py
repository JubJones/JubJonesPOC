import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

# Data from your provided information
data = {
    'Model': ['yolov11x', 'yolov11x', 'yolov11l', 'yolov11l', 'yolov9e', 'yolov9e', 'rtdetr-x', 'rtdetr-x', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn'],
    'Environment': ['Factory', 'Campus', 'Factory', 'Campus', 'Factory', 'Campus', 'Factory', 'Campus', 'Campus', 'Factory'],
    'Time (s)': [8, 10, 6, 8, 11, 11, 18, 19, 31, 33],
    'Detected': [48, 82, 56, 93, 67, 92, 78, 114, 66, 101],
    'Ground Truth': [64, 102, 64, 102, 64, 102, 64, 102, 102, 64]
}

df = pd.DataFrame(data)

# Enhanced Color Palette
colors = {
    'Factory Detected': 'rgb(255, 127, 14)',  # Orange
    'Campus Detected': 'rgb(31, 119, 180)',   # Blue
    'Factory Ground Truth': 'rgb(44, 160, 44)', # Green
    'Campus Ground Truth': 'rgb(148, 103, 189)', # Purple
    'Factory Time': 'rgb(214, 39, 40)', #Red
    'Campus Time' : 'rgb(173, 216, 230)' #Light blue
}

# Visualization 1: Bar plot comparing Detected vs. Ground Truth
fig1 = go.Figure()

for env in df['Environment'].unique():
    env_df = df[df['Environment'] == env]
    fig1.add_trace(go.Bar(
        x=env_df['Model'],
        y=env_df['Detected'],
        name=f'{env} Detected',
        marker_color=colors[f'{env} Detected'],
    ))

    fig1.add_trace(go.Scatter(
        x=env_df['Model'],
        y=env_df['Ground Truth'],
        name=f'{env} Ground Truth',
        mode='markers',
        marker=dict(size=10, symbol='square', color=colors[f'{env} Ground Truth']),
    ))

fig1.update_layout(
    title='Detected Objects vs. Ground Truth by Model and Environment',
    xaxis_title='Model',
    yaxis_title='Number of Objects',
    barmode='group',
)
fig1.show()

# Visualization 2: Bar plot comparing Time (s)
fig2 = go.Figure()

for env in df['Environment'].unique():
    env_df = df[df['Environment'] == env]
    fig2.add_trace(go.Bar(
        x=env_df['Model'],
        y=env_df['Time (s)'],
        name=f'{env} Time',
        marker_color=colors[f'{env} Time'],
    ))

fig2.update_layout(
    title='Processing Time by Model and Environment',
    xaxis_title='Model',
    yaxis_title='Time (s)',
    barmode='group',
)
fig2.show()

# Visualization 3: Scatter plot of Time vs. Detected
fig3 = go.Figure()

for model in df['Model'].unique():
    model_df = df[df['Model'] == model]
    fig3.add_trace(go.Scatter(
        x=model_df['Time (s)'],
        y=model_df['Detected'],
        name=model,
        mode='markers',
        marker=dict(size=10),
        text=model_df['Environment'],
    ))

fig3.update_layout(
    title='Time vs. Detected Objects by Model and Environment',
    xaxis_title='Time (s)',
    yaxis_title='Number of Objects Detected',
)
fig3.show()

# Visualization 4: Facet Plot (Subplots)
fig4 = make_subplots(rows=2, cols=1, subplot_titles=("Detected vs. Ground Truth", "Processing Time"))

for env in df['Environment'].unique():
    env_df = df[df['Environment'] == env]
    fig4.add_trace(go.Bar(
        x=env_df['Model'],
        y=env_df['Detected'],
        name=f'{env} Detected',
        marker_color=colors[f'{env} Detected'],
    ), row=1, col=1)

    fig4.add_trace(go.Scatter(
        x=env_df['Model'],
        y=env_df['Ground Truth'],
        name=f'{env} Ground Truth',
        mode='markers',
        marker=dict(size=10, symbol='square', color=colors[f'{env} Ground Truth']),
    ), row=1, col=1)

    fig4.add_trace(go.Bar(
        x=env_df['Model'],
        y=env_df['Time (s)'],
        name=f'{env} Time',
        marker_color=colors[f'{env} Time'],
    ), row=2, col=1)

fig4.update_layout(
    title='Model Performance Comparison',
    barmode='group',
)
fig4.show()