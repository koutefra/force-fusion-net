import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class FlowPlotter:
    @staticmethod
    def plot_flow_curve(flow_data: dict, out_path: str | Path, title: str = "Fundamental Diagram"):
        """
        flow_data: dict returned by FlowCurveEvaluator.evaluate_flow_curve()
                   must contain keys: rho_bins, Js_bins, v_bins
        out_path: path to save figure (str or Path)
        """
        rho = np.array(flow_data.get("rho_bins", []))
        Js  = np.array(flow_data.get("Js_bins", []))
        v   = np.array(flow_data.get("v_bins", []))

        fig, ax1 = plt.subplots(figsize=(6,4))
        color1 = "tab:blue"
        color2 = "tab:orange"

        ax1.set_title(title)
        ax1.set_xlabel("Density ρ [p/m²]")

        # Specific flow curve
        ax1.set_ylabel("Specific flow Js [p/s/m]", color=color1)
        ax1.plot(rho, Js, "o-", color=color1, label="Js vs ρ")
        ax1.tick_params(axis="y", labelcolor=color1)

        # Speed curve on secondary axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Mean speed v [m/s]", color=color2)
        ax2.plot(rho, v, "s--", color=color2, label="v vs ρ")
        ax2.tick_params(axis="y", labelcolor=color2)

        fig.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

    @staticmethod
    def plot_flow_comparison(flow_dict: dict[str, dict], out_dir: str | Path, title_prefix: str = "Comparison"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Specific flow comparison ---
        plt.figure(figsize=(6, 4))
        for name, data in flow_dict.items():
            rho = np.array(data.get("rho_bins", []))
            Js = np.array(data.get("Js_bins", []))
            plt.plot(rho, Js, "o-", label=name)
        plt.xlabel("Density ρ [p/m²]")
        plt.ylabel("Specific flow Js [p/s/m]")
        plt.title(f"{title_prefix}: Specific Flow")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "comparison_specific_flow.png", dpi=200)
        plt.close()

        # --- Mean speed comparison ---
        plt.figure(figsize=(6, 4))
        for name, data in flow_dict.items():
            rho = np.array(data.get("rho_bins", []))
            v = np.array(data.get("v_bins", []))
            plt.plot(rho, v, "s--", label=name)
        plt.xlabel("Density ρ [p/m²]")
        plt.ylabel("Mean speed v [m/s]")
        plt.title(f"{title_prefix}: Mean Speed")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "comparison_mean_speed.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    # --- Hardcoded flow data from your evaluations ---
    flow_data_gt = {
        "rho_bins": [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0],
        "Js_bins":  [0.0121826, 0.2926330, 0.6171399, 0.6582007, 0.7646258, 0.8934096, 0.9698081, 0.9847630],
        "v_bins":   [0.0389844, 0.4682128, 0.6582826, 0.4440419, 0.4078004, 0.4084158, 0.3879232, 0.3501380],
    }

    flow_data_sf = {
        "rho_bins": [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0],
        "Js_bins":  [0.1518033, 0.4959836, 0.7437219, 1.1188734, 1.5110240, 1.7478206, 1.9981855, 2.2629983],
        "v_bins":   [0.4857705, 0.7935738, 0.7933033, 0.7975302, 0.8058795, 0.7990037, 0.7992742, 0.8021805],
    }

    flow_data_fn = {
        "rho_bins": [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0],
        "Js_bins":  [0.0346790, 0.4252265, 0.6284698, 0.8609584, 1.0837007, 1.2392292, 1.4086873, 1.5282018],
        "v_bins":   [0.1109729, 0.6803624, 0.6703678, 0.6339670, 0.5779737, 0.5665048, 0.5634749, 0.5433607],
    }

    comparison_dict = {
        "Ground Truth": flow_data_gt,
        "Social Force": flow_data_sf,
        "FusionNet": flow_data_fn,
    }

    out_dir = "outputs"

    # Example: comparison plots
    FlowPlotter.plot_flow_comparison(comparison_dict, out_dir, title_prefix="Juelich B160")
