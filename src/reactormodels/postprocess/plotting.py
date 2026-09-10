"""plotting.py"""

from pathlib import Path
from collections.abc import Sequence
import numpy as np
import plotly.graph_objs as go
from ..properties.breakthrough import Breakthrough

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
]
markers = [
    "circle",
    "diamond",
    "triangle-up",
    "triangle-down",
    "star",
    "cross",
    "square",
    "cross",
]


class Plotting:
    """Class for plotting breakthrough and model fits."""

    def __init__(self, breakthroughs: Breakthrough | Sequence[Breakthrough]) -> None:
        """Initialize breakthrough class or classes."""
        if isinstance(breakthroughs, Breakthrough):
            self.breakthroughs = [breakthroughs]
        else:
            self.breakthroughs = list(breakthroughs)

    @staticmethod
    def _get_breakthrough_data(
        breakthrough: Breakthrough,
        normalized: bool,
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
        """Return x/y data and axis titles for a breakthrough."""
        if breakthrough.time_is_derived:
            if breakthrough.bed_volumes is None:
                raise ValueError("Breakthrough has no bed volumes.")
            x = breakthrough.bed_volumes
            x_title = "Bed Volumes"
        else:
            if breakthrough.time is None:
                raise ValueError("Breakthrough has no time.")
            x = breakthrough.time
            x_title = "Time"

        if breakthrough.effluent_concentrations is None:
            raise ValueError(
                f"Chemical: {breakthrough.chemical.name} has no "
                "effluent concentrations."
            )

        if normalized:
            y = breakthrough.normalize_concentration()
            y_title = "C/C<sub>o</sub>"
        else:
            y = breakthrough.effluent_concentrations
            y_title = "C"

        return x, y, x_title, y_title

    @staticmethod
    def _get_axis_ranges(
        x_data: Sequence[np.ndarray],
        y_data: Sequence[np.ndarray],
    ) -> tuple[float, float, float, float]:
        """Return axis ranges for breakthrough data."""
        x_min, x_max = 0, np.max(x_data)
        y_min, y_max = -np.min(y_data) * 0.05, np.max(y_data) * 1.05

        return x_min, x_max, y_min, y_max

    @staticmethod
    def _format_figure(
        fig: go.Figure,
        x_title: str,
        y_title: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        orientation: str | None = "v",
        xanchor: str | None = "left",
        yanchor: str | None = "bottom",
        x_loc: float | None = 1.01,
        y_loc: float | None = 0,
    ) -> None:
        """Apply common formatting to figures."""
        fig.update_layout(
            font={
                "family": "Arial",
                "size": 32,
                "color": "black",
            },
            plot_bgcolor="white",
            xaxis={
                "linecolor": "black",
                "ticks": "outside",
                "mirror": True,
                "linewidth": 3,
                "tickwidth": 3,
                "title_standoff": 10,
                "title": x_title,
                "range": [x_min, x_max],
                "showgrid": False,
            },
            yaxis={
                "linecolor": "black",
                "ticks": "outside",
                "mirror": True,
                "linewidth": 3,
                "tickwidth": 3,
                "title_standoff": 10,
                "title": y_title,
                "range": [y_min, y_max],
                "showgrid": False,
            },
            legend={
                "title": "",
                "font": {"size": 24},
                "bgcolor": "rgba(255, 255, 255, 0)",
                "x": x_loc,
                "y": y_loc,
                "xanchor": xanchor,
                "yanchor": yanchor,
                "orientation": orientation,
            },
            showlegend=True,
            width=1000,
            height=700,
        )

    @staticmethod
    def _save_figure(
        fig: go.Figure,
        save_path: str | Path | None,
    ) -> None:
        """Save figure if a path is provided."""
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig.write_image(save_path)

            print(f"Saved plot to {save_path}")

    def plot_breakthrough(
        self,
        names: Sequence[str],
        save_path: str | Path,
        normalized: bool = True,
        show: bool = False,
    ) -> go.Figure:
        """Plot breakthrough data only."""
        fig = go.Figure()

        for i, breakthrough in enumerate(self.breakthroughs):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            x, concentration, x_title, y_title = self._get_breakthrough_data(
                breakthrough,
                normalized,
            )

            name = breakthrough.chemical.name if names is None else names[i]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=concentration,
                    mode="markers",
                    name=f"{name} Data",
                    marker=dict(
                        color=color,
                        symbol=marker,
                        size=15,
                        line=dict(
                            width=2,
                            color=color,
                        ),
                    ),
                )
            )

        x_data = []
        y_data = []

        for breakthrough in self.breakthroughs:
            x, y, x_title, y_title = self._get_breakthrough_data(
                breakthrough,
                normalized,
            )
            x_data.append(x)
            y_data.append(y)

        x_min, x_max, y_min, y_max = self._get_axis_ranges(
            x_data,
            y_data,
        )

        self._format_figure(fig, x_title, y_title, x_min, x_max, y_min, y_max)
        self._save_figure(fig, save_path)

        if show:
            fig.show()

        return fig

    def plot_breakthrough_and_model(
        self,
        model_names: Sequence[str],
        model_outs: Sequence[Sequence[np.ndarray]],
        names: Sequence[str] | None = None,
        normalized: bool = True,
        show: bool = False,
        save_path: str | Path | None = None,
    ) -> go.Figure:
        """Plot breakthrough data and model fits for one or more species."""
        if len(model_outs) != len(self.breakthroughs):
            raise ValueError(
                "model_outs must contain one set of outputs for each breakthrough."
            )

        for outputs in model_outs:
            if len(outputs) != len(model_names):
                raise ValueError(
                    "Each species must have one model output for each model."
                )

        fig = go.Figure()

        for i, (breakthrough, outputs) in enumerate(
            zip(self.breakthroughs, model_outs, strict=True)
        ):
            x, concentration, x_title, y_title = self._get_breakthrough_data(
                breakthrough,
                normalized,
            )

            name = breakthrough.chemical.name if names is None else names[i]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=concentration,
                    mode="markers",
                    name=f"{name} Data",
                    marker=dict(
                        color=colors[i],
                        symbol=markers[i % len(markers)],
                        size=15,
                        line=dict(width=2, color=colors[i]),
                    ),
                )
            )

            for j, (model_name, model_out) in enumerate(
                zip(model_names, outputs, strict=True)
            ):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=model_out,
                        mode="lines",
                        name=f"{name} {model_name}",
                        line=dict(
                            color=colors[i],
                            width=4,
                            dash=["solid", "dash", "dot", "dashdot"][j % 4],
                        ),
                    )
                )

        x_data = []
        y_data = []

        for breakthrough in self.breakthroughs:
            x, y, x_title, y_title = self._get_breakthrough_data(
                breakthrough,
                normalized,
            )
            x_data.append(x)
            y_data.append(y)

        x_min, x_max, y_min, y_max = self._get_axis_ranges(
            x_data,
            y_data,
        )

        self._format_figure(fig, x_title, y_title, x_min, x_max, y_min, y_max)
        self._save_figure(fig, save_path)

        if show:
            fig.show()

        return fig
