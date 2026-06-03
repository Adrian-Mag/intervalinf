"""Interactive viewer for the normal-mode splitting data in this demo folder.

Run from ``paper_demos`` with::

    conda activate inferences
    python visualization/normal_mode_data_viewer.py

The viewer uses only Matplotlib widgets, so it can run as a normal
Python script without a notebook server. It can also render a static image::

    python visualization/normal_mode_data_viewer.py \
        --view kernels --parameter vs \
        --save /tmp/kernel.png --no-show
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

import _path_setup  # noqa: F401


matplotlib.use("TkAgg")


EARTH_RADIUS_KM = 6371.0

OBSERVED_DATASETS = {
    "arwen-paula": "data_arwen-paula",
    "SP12RTS": "data_SP12RTS",
}

KERNEL_PARAMETERS = ("vp", "vs", "rho", "topo")
VIEWS = ("data", "errors", "kernels", "synthetics")


@dataclass(frozen=True)
class ObservedSplittingData:
    """Observed splitting coefficients for one mode."""

    mode: str
    smax: int
    rows: np.ndarray


class NormalModeDataRepository:
    """Load observed splitting data, synthetics, and PREM-layer kernels."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.observed_root = data_root / "normal-mode-data"
        self.kernel_root = (
            data_root
            / "normal-mode-kernels"
            / "kernels-all_PREM-layers_Adrian"
        )
        self.synthetic_root = data_root / "normal-mode-synthetics"
        self._mode_lists: dict[str, list[str]] = {}
        self._radius_layers: dict[int, tuple[float, float]] | None = None
        self._synthetic_models: list[str] | None = None

    def list_observed_modes(self, dataset: str) -> list[str]:
        """Return sorted mode IDs available for an observed dataset."""
        if dataset not in self._mode_lists:
            if dataset not in OBSERVED_DATASETS:
                raise ValueError(f"Unknown observed dataset: {dataset}")
            list_name = {
                "arwen-paula": "data_list_arwen-paula",
                "SP12RTS": "data_list_SP12RTS",
            }[dataset]
            list_path = self.observed_root / list_name
            if list_path.exists():
                modes = [
                    line.strip()
                    for line in list_path.read_text().splitlines()
                    if line.strip() and not line.startswith("#")
                ]
            else:
                folder = self.observed_root / OBSERVED_DATASETS[dataset]
                modes = sorted(path.stem for path in folder.glob("*.new"))
            self._mode_lists[dataset] = modes
        return list(self._mode_lists[dataset])

    def load_observed(self, dataset: str, mode: str) -> ObservedSplittingData:
        """Load one ``.new`` observed splitting-coefficient file."""
        folder = self.observed_root / OBSERVED_DATASETS[dataset]
        path = folder / f"{mode}.new"
        if not path.exists():
            raise FileNotFoundError(f"Observed data file not found: {path}")

        lines = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        smax = int(lines[0].split()[0])
        rows = [
            [float(value) for value in line.split()[:4]]
            for line in lines[1:]
        ]
        return ObservedSplittingData(
            mode=mode,
            smax=smax,
            rows=np.asarray(rows),
        )

    def list_kernel_modes(self) -> list[str]:
        """Return mode IDs with at least one kernel file."""
        modes = {
            path.name.split("_")[1]
            for path in self.kernel_root.glob(
                "kern_*_PREM-layers-all_PREM-iso.kern"
            )
        }
        return sorted(modes)

    def load_radius_layers(self) -> dict[int, tuple[float, float]]:
        """Load PREM layer radius intervals keyed by layer number."""
        if self._radius_layers is None:
            layers: dict[int, tuple[float, float]] = {}
            path = self.kernel_root / "PREM_all-layers_radius"
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    continue
                layer = int(parts[0])
                layers[layer] = (float(parts[1]), float(parts[2]))
            self._radius_layers = layers
        return dict(self._radius_layers)

    def load_kernel(self, mode: str, parameter: str) -> np.ndarray:
        """Load one kernel file as columns ``s, layer, kern_val``."""
        if parameter not in KERNEL_PARAMETERS:
            raise ValueError(f"Unknown kernel parameter: {parameter}")
        path = (
            self.kernel_root
            / f"kern_{mode}_{parameter}_PREM-layers-all_PREM-iso.kern"
        )
        if not path.exists():
            raise FileNotFoundError(f"Kernel file not found: {path}")
        rows = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append([float(value) for value in stripped.split()[:3]])
        return np.asarray(rows)

    def list_synthetic_models(self) -> list[str]:
        """Return synthetic Earth-model labels from ``raw_*.raw`` files."""
        if self._synthetic_models is None:
            models = set()
            for path in self.synthetic_root.glob("raw_*.raw"):
                name = path.name[len("raw_"):-len(".raw")]
                try:
                    _, model = name.split("_", 1)
                except ValueError:
                    continue
                models.add(model)
            self._synthetic_models = sorted(models)
        return list(self._synthetic_models)

    def load_synthetic(self, mode: str, model: str) -> np.ndarray:
        """Load one synthetic ``raw_${mode}_${model}.raw`` file."""
        path = self.synthetic_root / f"raw_{mode}_{model}.raw"
        if not path.exists():
            raise FileNotFoundError(f"Synthetic data file not found: {path}")
        rows = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append([float(value) for value in stripped.split()[:4]])
        return np.asarray(rows)

    def summary_lines(self) -> list[str]:
        """Return concise dataset coverage lines for ``--list``."""
        lines: list[str] = []
        for dataset in OBSERVED_DATASETS:
            modes = self.list_observed_modes(dataset)
            smax = Counter()
            total_rows = 0
            for mode in modes:
                observed = self.load_observed(dataset, mode)
                smax[observed.smax] += 1
                total_rows += len(observed.rows)
            lines.append(
                f"{dataset}: {len(modes)} modes, {total_rows} rows, "
                f"smax distribution {dict(sorted(smax.items()))}"
            )
        kernel_modes = self.list_kernel_modes()
        lines.append(f"kernels: {len(kernel_modes)} modes")
        synthetic_count = len(self.list_synthetic_models())
        lines.append(f"synthetics: {synthetic_count} model variants")
        return lines


class NormalModeViewer:
    """Matplotlib-widget front end for normal-mode data exploration."""

    def __init__(
        self,
        repository: NormalModeDataRepository,
        args: argparse.Namespace,
        plt,
        button_class,
        radio_class,
        slider_class,
    ) -> None:
        self.repository = repository
        self.plt = plt
        self.Button = button_class
        self.RadioButtons = radio_class
        self.Slider = slider_class

        self.dataset = args.dataset
        self.view = args.view
        self.parameter = args.parameter
        self.synthetic_models = self.repository.list_synthetic_models()
        if not self.synthetic_models:
            raise SystemExit(
                "No synthetic model files found under "
                f"{self.repository.synthetic_root}. Check --data-root."
            )
        self.synthetic_model = (
            args.synthetic_model or self._default_synthetic_model()
        )
        if self.synthetic_model not in self.synthetic_models:
            available = ", ".join(self.synthetic_models)
            raise SystemExit(
                f"Unknown synthetic model {self.synthetic_model!r}. "
                f"Available: {available}"
            )

        self.modes = self.repository.list_observed_modes(self.dataset)
        self.mode_index = self._initial_mode_index(args.mode)

        self.fig, self.ax = self.plt.subplots(figsize=(13.5, 7.5))
        self.fig.subplots_adjust(left=0.24, right=0.77, bottom=0.16, top=0.91)
        self._build_widgets()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.update()

    @property
    def mode(self) -> str:
        """Currently selected mode ID."""
        return self.modes[self.mode_index]

    def show(self) -> None:
        """Show the Matplotlib window."""
        self.plt.show()

    def save(self, path: Path) -> None:
        """Save the current figure."""
        self.fig.savefig(path, dpi=160, bbox_inches="tight")

    def _default_synthetic_model(self) -> str:
        preferred = "S20RTS-Crust5.1"
        if preferred in self.synthetic_models:
            return preferred
        return self.synthetic_models[0]

    def _initial_mode_index(self, requested_mode: str | None) -> int:
        if requested_mode is None:
            return 0
        if requested_mode not in self.modes:
            available = ", ".join(self.modes[:10])
            raise SystemExit(
                f"Mode {requested_mode!r} is not in {self.dataset}. "
                f"First modes: {available}"
            )
        return self.modes.index(requested_mode)

    def _build_widgets(self) -> None:
        view_ax = self.fig.add_axes([0.03, 0.60, 0.16, 0.24])
        self.view_radio = self.RadioButtons(
            view_ax,
            VIEWS,
            active=VIEWS.index(self.view),
        )
        self.view_radio.on_clicked(self._set_view)

        dataset_ax = self.fig.add_axes([0.03, 0.44, 0.16, 0.12])
        dataset_labels = tuple(OBSERVED_DATASETS)
        self.dataset_radio = self.RadioButtons(
            dataset_ax,
            dataset_labels,
            active=dataset_labels.index(self.dataset),
        )
        self.dataset_radio.on_clicked(self._set_dataset)

        parameter_ax = self.fig.add_axes([0.82, 0.64, 0.13, 0.18])
        self.parameter_radio = self.RadioButtons(
            parameter_ax,
            KERNEL_PARAMETERS,
            active=KERNEL_PARAMETERS.index(self.parameter),
        )
        self.parameter_radio.on_clicked(self._set_parameter)

        model_ax = self.fig.add_axes([0.79, 0.14, 0.20, 0.43])
        model_labels = [
            self._short_model_label(model)
            for model in self.synthetic_models
        ]
        self.model_label_to_name = dict(
            zip(model_labels, self.synthetic_models, strict=True)
        )
        self.model_radio = self.RadioButtons(
            model_ax,
            model_labels,
            active=self.synthetic_models.index(self.synthetic_model),
        )
        self.model_radio.on_clicked(self._set_synthetic_model)

        slider_ax = self.fig.add_axes([0.28, 0.07, 0.43, 0.03])
        self.mode_slider = self.Slider(
            slider_ax,
            "mode index",
            0,
            max(len(self.modes) - 1, 0),
            valinit=self.mode_index,
            valstep=1,
            valfmt="%0.0f",
        )
        self.mode_slider.on_changed(self._set_mode_index)

        prev_ax = self.fig.add_axes([0.28, 0.02, 0.10, 0.035])
        next_ax = self.fig.add_axes([0.61, 0.02, 0.10, 0.035])
        self.prev_button = self.Button(prev_ax, "previous")
        self.next_button = self.Button(next_ax, "next")
        self.prev_button.on_clicked(lambda _event: self._step_mode(-1))
        self.next_button.on_clicked(lambda _event: self._step_mode(1))

    def _set_view(self, label: str) -> None:
        self.view = label
        self.update()

    def _set_dataset(self, label: str) -> None:
        old_mode = self.mode
        old_index = self.mode_index
        self.dataset = label
        self.modes = self.repository.list_observed_modes(self.dataset)
        if old_mode in self.modes:
            self.mode_index = self.modes.index(old_mode)
        else:
            self.mode_index = min(old_index, len(self.modes) - 1)
        self.mode_slider.valmax = max(len(self.modes) - 1, 0)
        self.mode_slider.ax.set_xlim(0, max(len(self.modes) - 1, 0))
        self.mode_slider.set_val(self.mode_index)
        self.update()

    def _set_parameter(self, label: str) -> None:
        self.parameter = label
        self.update()

    def _set_synthetic_model(self, label: str) -> None:
        self.synthetic_model = self.model_label_to_name[label]
        self.update()

    def _set_mode_index(self, value: float) -> None:
        self.mode_index = int(round(value))
        self.update()

    def _step_mode(self, step: int) -> None:
        self.mode_index = (self.mode_index + step) % len(self.modes)
        self.mode_slider.set_val(self.mode_index)

    def _on_key_press(self, event) -> None:
        if event.key == "left":
            self._step_mode(-1)
        elif event.key == "right":
            self._step_mode(1)
        elif event.key in {"d", "e", "k", "s"}:
            view_by_key = {
                "d": "data",
                "e": "errors",
                "k": "kernels",
                "s": "synthetics",
            }
            self.view = view_by_key[event.key]
            self.update()

    def update(self) -> None:
        """Redraw the active view."""
        self.ax.clear()
        try:
            if self.view == "data":
                self._plot_observed_coefficients(with_error_bars=True)
            elif self.view == "errors":
                self._plot_observed_errors()
            elif self.view == "kernels":
                self._plot_kernel()
            elif self.view == "synthetics":
                self._plot_synthetic_comparison()
            else:
                raise ValueError(f"Unknown view: {self.view}")
        except Exception as exc:
            self.ax.text(
                0.5,
                0.5,
                str(exc),
                ha="center",
                va="center",
                wrap=True,
            )
            self.ax.set_axis_off()
        self.fig.canvas.draw_idle()

    def _plot_observed_coefficients(self, with_error_bars: bool) -> None:
        observed = self.repository.load_observed(self.dataset, self.mode)
        for harmonic_degree in self._unique_s_values(observed.rows):
            rows = self._rows_for_s(observed.rows, harmonic_degree)
            kwargs = {
                "marker": "o",
                "markersize": 4,
                "linewidth": 1.2,
                "label": f"s={harmonic_degree}",
            }
            if with_error_bars:
                self.ax.errorbar(
                    rows[:, 1],
                    rows[:, 2],
                    yerr=rows[:, 3],
                    capsize=2,
                    **kwargs,
                )
            else:
                self.ax.plot(rows[:, 1], rows[:, 2], **kwargs)

        self.ax.axhline(0.0, color="0.45", linewidth=0.8)
        self.ax.set_xlabel("raw t index")
        self.ax.set_ylabel("splitting coefficient (raw)")
        self.ax.set_title(
            f"Observed splitting data: {self.dataset}, "
            f"mode {self.mode}, smax={observed.smax}"
        )
        self._finish_axes()

    def _plot_observed_errors(self) -> None:
        observed = self.repository.load_observed(self.dataset, self.mode)
        for harmonic_degree in self._unique_s_values(observed.rows):
            rows = self._rows_for_s(observed.rows, harmonic_degree)
            self.ax.plot(
                rows[:, 1],
                rows[:, 3],
                marker="o",
                markersize=4,
                linewidth=1.2,
                label=f"s={harmonic_degree}",
            )

        self.ax.set_xlabel("raw t index")
        self.ax.set_ylabel("reported coefficient error (raw)")
        self.ax.set_title(f"Reported errors: {self.dataset}, mode {self.mode}")
        self._finish_axes()

    def _plot_kernel(self) -> None:
        rows = self.repository.load_kernel(self.mode, self.parameter)
        if self.parameter == "topo":
            self._plot_topography_kernel(rows)
            return

        radius_layers = self.repository.load_radius_layers()
        for harmonic_degree in self._unique_s_values(rows):
            subset = self._rows_for_s(rows, harmonic_degree)
            radii, values = self._layer_step_curve(subset, radius_layers)
            if len(radii) == 0:
                continue
            self.ax.plot(
                radii,
                values,
                linewidth=1.1,
                label=f"s={harmonic_degree}",
            )

        self.ax.axhline(0.0, color="0.45", linewidth=0.8)
        self.ax.set_xlim(0.0, EARTH_RADIUS_KM)
        self.ax.set_xlabel("radius from centre (km)")
        self.ax.set_ylabel(f"{self.parameter} layer sensitivity weight")
        self.ax.set_title(
            f"{self.parameter} sensitivity kernels: mode {self.mode}"
        )
        self._finish_axes()

    def _plot_topography_kernel(self, rows: np.ndarray) -> None:
        order = np.lexsort((rows[:, 1], rows[:, 0]))
        rows = rows[order]
        s_values = rows[:, 0].astype(int)
        layer_ids = rows[:, 1].astype(int)
        values = rows[:, 2]
        positions = np.arange(len(rows))
        self.ax.bar(positions, values, color="tab:purple", alpha=0.8)
        self.ax.axhline(0.0, color="0.45", linewidth=0.8)
        self.ax.set_xticks(positions)
        self.ax.set_xticklabels([str(value) for value in s_values])
        self.ax.set_xlabel("s")
        self.ax.set_ylabel("topography sensitivity weight")
        layer_label = self._format_layer_ids(layer_ids)
        self.ax.set_title(
            f"Topography kernel: mode {self.mode}, PREM layer {layer_label}"
        )
        self._finish_axes(show_legend=False)

    def _plot_synthetic_comparison(self) -> None:
        observed = self.repository.load_observed(self.dataset, self.mode)
        synthetic = self.repository.load_synthetic(
            self.mode,
            self.synthetic_model,
        )
        synthetic_by_component = {
            (int(row[1]), int(row[2])): float(row[3]) for row in synthetic
        }

        for harmonic_degree in self._unique_s_values(observed.rows):
            rows = self._rows_for_s(observed.rows, harmonic_degree)
            synthetic_values = np.array(
                [
                    synthetic_by_component.get(
                        (harmonic_degree, int(row[1])),
                        np.nan,
                    )
                    for row in rows
                ]
            )
            observed_line = self.ax.errorbar(
                rows[:, 1],
                rows[:, 2],
                yerr=rows[:, 3],
                marker="o",
                markersize=4,
                linewidth=1.0,
                capsize=2,
                label=f"obs s={harmonic_degree}",
            )
            color = observed_line.lines[0].get_color()
            self.ax.plot(
                rows[:, 1],
                synthetic_values,
                linestyle="--",
                marker="x",
                markersize=4,
                linewidth=1.0,
                color=color,
                label=f"syn s={harmonic_degree}",
            )

        self.ax.axhline(0.0, color="0.45", linewidth=0.8)
        self.ax.set_xlabel("raw t index")
        self.ax.set_ylabel("splitting coefficient (raw)")
        self.ax.set_title(
            f"Observed vs synthetic: {self.dataset}, "
            f"mode {self.mode}, {self.synthetic_model}"
        )
        self._finish_axes(ncol=2)

    @staticmethod
    def _unique_s_values(rows: np.ndarray) -> list[int]:
        return sorted({int(value) for value in rows[:, 0]})

    @staticmethod
    def _rows_for_s(rows: np.ndarray, harmonic_degree: int) -> np.ndarray:
        subset = rows[rows[:, 0].astype(int) == harmonic_degree]
        return subset[np.argsort(subset[:, 1])]

    @staticmethod
    def _layer_step_curve(
        rows: np.ndarray,
        radius_layers: dict[int, tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        radii: list[float] = []
        values: list[float] = []
        previous_value: float | None = None
        for row in rows[np.argsort(rows[:, 1])]:
            layer = int(row[1])
            value = float(row[2])
            if layer not in radius_layers:
                continue
            radius_min, radius_max = radius_layers[layer]
            if radius_max <= radius_min:
                continue
            if not radii:
                radii.extend([radius_min, radius_max])
                values.extend([value, value])
            else:
                radii.extend([radius_min, radius_min, radius_max])
                values.extend(
                    [
                        previous_value
                        if previous_value is not None
                        else value,
                        value,
                        value,
                    ]
                )
            previous_value = value
        return np.asarray(radii), np.asarray(values)

    @staticmethod
    def _format_layer_ids(layer_ids: Iterable[int]) -> str:
        unique = sorted(set(int(layer_id) for layer_id in layer_ids))
        if len(unique) == 1:
            return str(unique[0])
        return ", ".join(str(layer_id) for layer_id in unique)

    @staticmethod
    def _short_model_label(model: str) -> str:
        replacements = {
            "S20RTS-Crust5.1": "S20RTS+Crust5.1",
            "S20RTS-Crust5.1_cmb_li-etal-1991-SAT": "cmb li-1991",
            (
                "S20RTS-Crust5.1_cmb_"
                "koelemeijer-2019-new-bestfit-neg"
            ): "cmb koel-2019",
            "S20RTS-Crust5.1_cmb_ishii-tromp-2001": "cmb ishii-2001",
            "S20RTS-Crust5.1_cmb_tanaka-2010": "cmb tanaka-2010",
            (
                "S20RTS-Crust5.1_cmb_"
                "soldati-etal-2012-TG-PPV200"
            ): "cmb soldati-2012",
        }
        if model in replacements:
            return replacements[model]
        return model.replace("_", " ")[:24]

    def _finish_axes(self, show_legend: bool = True, ncol: int = 1) -> None:
        self.ax.grid(True, alpha=0.25)
        if show_legend:
            self.ax.legend(fontsize=8, ncol=ncol, loc="best")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    script_dir = Path(__file__).resolve().parent
    demo_dir = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=demo_dir / "data",
        help="Path to the paper_demos data directory.",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(OBSERVED_DATASETS),
        default="arwen-paula",
    )
    parser.add_argument("--mode", help="Initial mode ID, for example 00s02.")
    parser.add_argument("--view", choices=VIEWS, default="data")
    parser.add_argument("--parameter", choices=KERNEL_PARAMETERS, default="vs")
    parser.add_argument(
        "--synthetic-model",
        help="Initial synthetic Earth model.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Save the current view to an image path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive window.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print dataset coverage and exit.",
    )
    return parser


def main() -> None:
    """Run the interactive viewer."""
    parser = build_parser()
    args = parser.parse_args()
    repository = NormalModeDataRepository(args.data_root)

    if args.list:
        for line in repository.summary_lines():
            print(line)
        return

    if args.no_show or args.save:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider

    viewer = NormalModeViewer(
        repository,
        args,
        plt,
        Button,
        RadioButtons,
        Slider,
    )
    if args.save:
        viewer.save(args.save)
    if not args.no_show:
        viewer.show()


if __name__ == "__main__":
    main()
