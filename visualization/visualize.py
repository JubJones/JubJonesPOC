import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

data = {
    "Model": [
        "yolov11x",
        "yolov11x",
        "yolov11l",
        "yolov11l",
        "yolov9e",
        "yolov9e",
        "rtdetr-x",
        "rtdetr-x",
        "rfdetr_base",
        "rfdetr_base",
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_resnet50_fpn",
    ],
    "Environment": [
        "Factory",
        "Campus",
        "Factory",
        "Campus",
        "Factory",
        "Campus",
        "Factory",
        "Campus",
        "Factory",
        "Campus",
        "Factory",
        "Campus",
    ],
    "Time (s)": [8, 10, 6, 8, 11, 11, 18, 19, 8, 6, 33, 31],
    "Detected": [48, 82, 56, 93, 67, 92, 78, 114, 49, 90, 66, 101],
    "Ground Truth": [64, 102, 64, 102, 64, 102, 64, 102, 64, 102, 64, 102],
}

df = pd.DataFrame(data)

# --- Calculate Average Time for Sorting ---
model_avg_time = df.groupby("Model")["Time (s)"].mean().reset_index()
# Sort models by average time, ascending
model_order_by_time = model_avg_time.sort_values(by="Time (s)", ascending=True)[
    "Model"
].tolist()

print(f"Model Order by Average Time (Ascending): {model_order_by_time}")
# Note: Based on this order, rtdetr-x and rfdetr_base might not be naturally adjacent.
# Prioritizing sorting by time as requested.

# --- Colors ---
colors = {
    "Factory Detected": "rgb(255, 127, 14)",  # Orange
    "Campus Detected": "rgb(31, 119, 180)",  # Blue
    "Factory Ground Truth": "rgb(44, 160, 44)",  # Green
    "Campus Ground Truth": "rgb(148, 103, 189)",  # Purple
    "Factory Time": "rgb(214, 39, 40)",  # Red
    "Campus Time": "rgb(173, 216, 230)",  # Light blue
}
env_symbols = {"Factory": "square", "Campus": "circle"}

# --- Visualization 1: Bar plot comparing Detected vs. Ground Truth (Ordered by Time) ---
fig1 = go.Figure()

# Loop through environments and add traces directly using the filtered dataframe
for env in df["Environment"].unique():
    env_df = df[df["Environment"] == env]  # Filter for the specific environment
    fig1.add_trace(
        go.Bar(
            x=env_df["Model"],  # Use models from this environment's slice
            y=env_df["Detected"],
            name=f"{env} Detected",
            marker_color=colors[f"{env} Detected"],
            text=env_df["Detected"],  # Add value as text
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{env} Detected: %{{y}}<extra></extra>",
        )
    )

# Add Ground Truth markers using the same filtered dataframe slice
for env in df["Environment"].unique():
    env_df = df[df["Environment"] == env]
    fig1.add_trace(
        go.Scatter(
            x=env_df["Model"],
            y=env_df["Ground Truth"],
            name=f"{env} Ground Truth",
            mode="markers",
            marker=dict(
                size=10,
                symbol=env_symbols[env],
                color=colors[f"{env} Ground Truth"],
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            hovertemplate=f"<b>%{{x}}</b><br>{env} Ground Truth: %{{y}}<extra></extra>",
            showlegend=True,
        )
    )

# Apply the desired category order to the x-axis
fig1.update_layout(
    title="Detected Objects vs. Ground Truth",
    xaxis_title="Model",
    yaxis_title="Number of Objects",
    xaxis={
        "categoryorder": "array",
        "categoryarray": model_order_by_time,
    },  # Apply sorted order
    barmode="group",
    legend_title_text="Metric & Environment",
    hovermode="x unified",
)
fig1.update_traces(textfont_size=10)
fig1.show()

# --- Visualization 2: Bar plot comparing Time (s) (Ordered by Time) ---
fig2 = go.Figure()

for env in df["Environment"].unique():
    env_df = df[df["Environment"] == env]  # Filter for the specific environment
    fig2.add_trace(
        go.Bar(
            x=env_df["Model"],
            y=env_df["Time (s)"],
            name=f"{env} Time",
            marker_color=colors[f"{env} Time"],
            text=env_df["Time (s)"],  # Add value as text
            texttemplate="%{text:.1f}s",  # Format text
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{env} Time: %{{y:.1f}}s<extra></extra>",
        )
    )

# Apply the desired category order to the x-axis
fig2.update_layout(
    title="Processing Time",
    xaxis_title="Model",
    yaxis_title="Time (s)",
    xaxis={
        "categoryorder": "array",
        "categoryarray": model_order_by_time,
    },  # Apply sorted order
    barmode="group",
    legend_title_text="Environment",
    hovermode="x unified",
)
fig2.update_traces(textfont_size=10)
fig2.show()

# --- Visualization 3: Scatter plot of Time vs. Detected (Text includes Model Name) ---
fig3 = go.Figure()

# Assign colors consistently based on the sorted model order
model_colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
model_color_map = {
    model: model_colors_list[i % len(model_colors_list)]
    for i, model in enumerate(model_order_by_time)
}

hover_template_scatter = "<b>%{text} (%{customdata[0]})</b><br>Time: %{x:.1f}s<br>Detected: %{y}<extra></extra>"

# Plot points for each model individually for legend clarity
for model in model_order_by_time:  # Iterate in sorted order
    model_df = df[df["Model"] == model]
    fig3.add_trace(
        go.Scatter(
            x=model_df["Time (s)"],
            y=model_df["Detected"],
            name=model,  # Legend entry per model
            mode="markers+text",  # Add text next to markers
            marker=dict(
                size=12,
                color=model_color_map[model],  # Color by model
                symbol=[
                    env_symbols[env] for env in model_df["Environment"]
                ],  # Symbol by environment
            ),
            text=model_df["Model"],  # Add Model name as text label
            textposition="top right",
            textfont=dict(size=9),
            customdata=model_df[["Environment"]],  # Store environment for hover
            hovertemplate=hover_template_scatter,
        )
    )

fig3.update_layout(
    title="Time vs. Detected Objects by Model and Environment",
    xaxis_title="Time (s)",
    yaxis_title="Number of Objects Detected",
    legend_title_text="Model",
)
fig3.show()


# --- Visualization 4: Facet Plot (Subplots - Ordered by Time) ---
fig4 = make_subplots(
    rows=2, cols=1, subplot_titles=("Detected vs. Ground Truth", "Processing Time (s)")
)

# Plot 1: Detected vs GT
for env in df["Environment"].unique():
    env_df = df[df["Environment"] == env]  # Filter for the specific environment
    fig4.add_trace(
        go.Bar(
            x=env_df["Model"],
            y=env_df["Detected"],
            name=f"{env} Detected",
            marker_color=colors[f"{env} Detected"],
            text=env_df["Detected"],
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{env} Detected: %{{y}}<extra></extra>",
            legendgroup="detect",  # Group legend items
        ),
        row=1,
        col=1,
    )

# Plot 1: GT Markers
for env in df["Environment"].unique():
    env_df = df[df["Environment"] == env]  # Filter for the specific environment
    fig4.add_trace(
        go.Scatter(
            x=env_df["Model"],
            y=env_df["Ground Truth"],
            name=f"{env} Ground Truth",
            mode="markers",
            marker=dict(
                size=10,
                symbol=env_symbols[env],
                color=colors[f"{env} Ground Truth"],
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            hovertemplate=f"<b>%{{x}}</b><br>{env} Ground Truth: %{{y}}<extra></extra>",
            legendgroup="detect",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

# Plot 2: Time
for env in df["Environment"].unique():
    env_df = df[df["Environment"] == env]  # Filter for the specific environment
    fig4.add_trace(
        go.Bar(
            x=env_df["Model"],
            y=env_df["Time (s)"],
            name=f"{env} Time",
            marker_color=colors[f"{env} Time"],
            text=env_df["Time (s)"],
            texttemplate="%{text:.1f}s",
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{env} Time: %{{y:.1f}}s<extra></extra>",
            legendgroup="time",  # Group legend items
        ),
        row=2,
        col=1,
    )

# Update layout and axes for subplots
fig4.update_layout(
    title_text=f"Model Performance Comparison",
    height=800,
    barmode="group",
    hovermode="x unified",
    legend_title_text="Metric & Environment",
)

# Update axis order and titles for subplots
fig4.update_xaxes(
    categoryorder="array", categoryarray=model_order_by_time, row=1, col=1
)  # Apply order
fig4.update_xaxes(
    categoryorder="array",
    categoryarray=model_order_by_time,
    title_text="Model",
    row=2,
    col=1,
)  # Apply order & title
fig4.update_yaxes(title_text="Number of Objects", row=1, col=1)
fig4.update_yaxes(title_text="Time (s)", row=2, col=1)
fig4.update_traces(textfont_size=10, selector=dict(type="bar"))


fig4.show()
