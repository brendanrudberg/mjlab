"""Tables and figures from a compare_policies run, for slides.

Usage: uv run src/mjlab/scripts/eval_report.py <out_dir> [--labels A B ...]

Reads <out_dir>/compare.csv and <out_dir>/samples.npz (written by
compare_policies.py --csv/--samples) and writes, into <out_dir>:
  tables.md                    capability, hardware-load and impact tables
  table_*.csv                  the same tables as CSV
  fig_capability.png           actual vs commanded speed, falls, tracking error
  fig_rms_torque.png           hip and knee RMS torque vs the 43 N*m rating
  fig_impacts.png              touchdown speed and foot force vs speed
  fig_envelope.png             torque-speed samples against the X8-120 curve
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mjlab.asset_zoo.robots.caltech_c1 import c1_constants as c1  # noqa: E402

# Reference palette (light): categorical slots in fixed order, neutral ink.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MIN_TOUCHDOWNS = 20
FALLS_FLAG = 5.0  # falls per robot-minute

plt.rcParams.update(
  {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
    "legend.frameon": False,
    "lines.linewidth": 2,
  }
)


def load(out: Path) -> dict[int, dict[float, dict]]:
  runs: dict[int, dict[float, dict]] = defaultdict(dict)
  with (out / "compare.csv").open() as fh:
    for row in csv.DictReader(fh):
      vals = {}
      for k, v in row.items():
        try:
          vals[k] = float(v)
        except ValueError:
          vals[k] = v
      # A median over a handful of touchdowns (e.g. a run that mostly falls) is
      # noise; report it as missing instead.
      if vals["td_left"] + vals["td_right"] < MIN_TOUCHDOWNS:
        vals["td_vz"] = float("nan")
      runs[int(vals["run"])][float(vals["cmd"])] = vals
  return dict(runs)


def fmt(v: float, spec: str) -> str:
  return "n/a" if v != v else format(v, spec)  # v != v catches NaN


def flag(r: dict) -> str:
  """Mark rows whose statistics are dominated by falling."""
  return " †" if r["falls_per_min"] > FALLS_FLAG else ""


def write_table(out: Path, name: str, title: str, header: list[str], rows: list):
  with (out / f"table_{name}.csv").open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(header)
    writer.writerows(rows)
  lines = [f"### {title}", "", "| " + " | ".join(header) + " |"]
  lines.append("|" + "|".join("---" for _ in header) + "|")
  lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
  return "\n".join(lines) + "\n"


def tables(out: Path, runs: dict, labels: list[str]) -> None:
  cmds = sorted(next(iter(runs.values())))
  rated = c1.RATED_TORQUE
  md = ["# C1 evaluation tables", ""]
  md.append(
    "64 robots x 10 s per command, deterministic policy, fixed commands, the first"
    " second after each reset excluded. Each run uses the gains it trained with on"
    " the X8-120 speed-torque model.\n"
  )

  rows = []
  for cmd in cmds:
    for i, label in enumerate(labels):
      r = runs[i][cmd]
      rows.append(
        [
          f"{cmd:+.1f}{flag(r)}",
          label,
          f"{r['vx']:+.2f}",
          f"{r['err']:.2f}",
          f"{r['falls_per_min']:.2f}",
          f"{int(r['td_left'])}/{int(r['td_right'])}",
          f"{r['balance']:.2f}",
        ]
      )
  md.append(
    write_table(
      out,
      "capability",
      "Capability",
      [
        "cmd (m/s)",
        "run",
        "actual (m/s)",
        "tracking error (m/s)",
        "falls / robot-min",
        "touchdowns L/R",
        "L/R balance",
      ],
      rows,
    )
  )

  rows = []
  for cmd in cmds:
    for i, label in enumerate(labels):
      r = runs[i][cmd]
      rows.append(
        [
          f"{cmd:+.1f}{flag(r)}",
          label,
          f"{r['hip_rms']:.1f} ({r['hip_rms'] / rated:.2f}x)",
          f"{r['knee_rms']:.1f} ({r['knee_rms'] / rated:.2f}x)",
          f"{100 * r['hip_sat']:.1f}%",
          f"{100 * r['knee_sat']:.1f}%",
          f"{r['vel_pk']:.1f}",
          fmt(r["cot"], ".2f"),
        ]
      )
  md.append(
    write_table(
      out,
      "hardware_load",
      f"Hardware load (rated torque {rated:.0f} N·m, no-load speed"
      f" {c1.NO_LOAD_SPEED:.2f} rad/s)",
      [
        "cmd (m/s)",
        "run",
        "hip RMS N·m (x rated)",
        "knee RMS N·m (x rated)",
        "hip saturated",
        "knee saturated",
        "peak joint speed (rad/s)",
        "cost of transport",
      ],
      rows,
    )
  )

  rows = []
  for cmd in cmds:
    for i, label in enumerate(labels):
      r = runs[i][cmd]
      rows.append(
        [
          f"{cmd:+.1f}{flag(r)}",
          label,
          fmt(r["td_vz"], ".2f"),
          f"{r['foot_p99_bw']:.1f}",
          f"{r['acc_p99_g']:.1f}",
        ]
      )
  md.append(
    write_table(
      out,
      "impacts",
      "Impacts (foot force and acceleration assume rigid contact: upper bounds)",
      [
        "cmd (m/s)",
        "run",
        "median downward touchdown speed (m/s)",
        "foot force p99 (BW)",
        "base accel p99 (g)",
      ],
      rows,
    )
  )
  md.append(
    f"† More than {FALLS_FLAG:g} falls per robot-minute: the robot spends much of"
    " the time falling or recovering, so its load and impact statistics describe"
    " failures, not the gait.\n"
  )
  (out / "tables.md").write_text("\n".join(md))
  print("saved", out / "tables.md")


def finish(fig, path: Path, note: str) -> None:
  fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=8.5, color=MUTED)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)
  print("saved", path)


def per_run(runs: dict, i: int, key: str) -> tuple[list[float], list[float]]:
  cmds = sorted(runs[i])
  return cmds, [runs[i][c][key] for c in cmds]


def figures(out: Path, runs: dict, labels: list[str]) -> None:
  n = len(labels)
  # Capability.
  fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
  cmds_all = sorted(next(iter(runs.values())))
  lim = (min(cmds_all) - 0.3, max(cmds_all) + 0.3)
  axes[0].plot(lim, lim, color=AXIS, lw=1, ls="--")
  width = 0.8 / n
  for i, label in enumerate(labels):
    cmds, vx = per_run(runs, i, "vx")
    axes[0].plot(cmds, vx, color=SERIES[i], marker="o", ms=5, label=label)
    _, falls = per_run(runs, i, "falls_per_min")
    axes[1].bar(
      np.array(cmds) + (i - (n - 1) / 2) * width,
      falls,
      width * 0.95,
      color=SERIES[i],
      label=label,
    )
    _, err = per_run(runs, i, "err")
    axes[2].plot(cmds, err, color=SERIES[i], marker="o", ms=5, label=label)
  axes[0].set_title("Actual vs commanded speed")
  axes[0].set_xlabel("commanded (m/s)")
  axes[0].set_ylabel("actual mean (m/s)")
  axes[1].set_title("Falls")
  axes[1].set_xlabel("commanded (m/s)")
  axes[1].set_ylabel("falls per robot-minute")
  # Keep a floor on the scale so a handful of falls doesn't look like many.
  peak_falls = max(max(per_run(runs, i, "falls_per_min")[1]) for i in range(n))
  axes[1].set_ylim(0, max(1.0, 1.15 * peak_falls))
  axes[1].grid(axis="x", visible=False)
  axes[2].set_title("Tracking error")
  axes[2].set_xlabel("commanded (m/s)")
  axes[2].set_ylabel("mean |actual - cmd| (m/s)")
  axes[0].legend(loc="upper left", fontsize=8)
  finish(
    fig,
    out / "fig_capability.png",
    "64 robots x 10 s per command, deterministic policy, fixed commands.",
  )

  # RMS torque against the rating.
  fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
  for ax, key, joint in ((axes[0], "hip_rms", "Hip"), (axes[1], "knee_rms", "Knee")):
    for i, label in enumerate(labels):
      cmds, y = per_run(runs, i, key)
      ax.plot(cmds, y, color=SERIES[i], marker="o", ms=5, label=label)
    ax.axhline(c1.RATED_TORQUE, color=INK2, lw=1, ls="--", label="43 N·m rated")
    ax.set_title(f"{joint} RMS torque (heat)")
    ax.set_xlabel("commanded (m/s)")
    ax.set_ylim(bottom=0)
  axes[0].set_ylabel("N·m, mean of left and right")
  axes[0].legend(loc="lower left", fontsize=8)
  finish(
    fig,
    out / "fig_rms_torque.png",
    "RMS torque against the X8-120's 43 N·m continuous rating; above the line the"
    " motor heats faster than it is rated to shed.",
  )

  # Impacts.
  fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
  for i, label in enumerate(labels):
    cmds, tdz = per_run(runs, i, "td_vz")
    axes[0].plot(cmds, tdz, color=SERIES[i], marker="o", ms=5, label=label)
    _, f = per_run(runs, i, "foot_p99_bw")
    axes[1].plot(cmds, f, color=SERIES[i], marker="o", ms=5, label=label)
  axes[0].set_title("Downward foot speed at touchdown (median)")
  axes[0].set_ylabel("m/s")
  axes[1].set_title("Foot force, 99th percentile")
  axes[1].set_ylabel("body weights")
  for ax in axes:
    ax.set_xlabel("commanded (m/s)")
    ax.set_ylim(bottom=0)
  axes[0].legend(loc="lower left", fontsize=8)
  finish(
    fig,
    out / "fig_impacts.png",
    "Touchdown speed is contact-model independent; foot force assumes rigid contact"
    " (an upper bound).",
  )

  # Torque-speed envelope.
  samples_path = out / "samples.npz"
  if not samples_path.exists():
    return
  d = np.load(samples_path)
  names = list(d["joint_names"])
  pick = [c for c in (2.0, 4.0) if f"run0_cmd{c:+.1f}" in d.files] or [max(cmds_all)]
  fig, axes = plt.subplots(n, 2, figsize=(11, 3.6 * n), sharex=True, sharey=True)
  axes = np.atleast_2d(axes)
  v_line = np.linspace(0, c1.NO_LOAD_SPEED, 200)
  env_line = np.clip(
    c1.VOLTAGE_LINE_STALL_TORQUE * (1 - v_line / c1.NO_LOAD_SPEED), 0, c1.TORQUE_CAP
  )
  for i, label in enumerate(labels):
    for col, joint in enumerate(("hip", "knee")):
      ax = axes[i, col]
      cols = [k for k, nm in enumerate(names) if joint in nm]
      for j, cmd in enumerate(pick):
        smp = d[f"run{i}_cmd{cmd:+.1f}"]  # [N, joints, 2] = (torque, speed)
        tau, vel = smp[:, cols, 0].ravel(), smp[:, cols, 1].ravel()
        driving = tau * vel > 0
        color = SERIES[i] if j == len(pick) - 1 else "#a8a7a0"
        ax.scatter(
          np.abs(vel[driving]),
          np.abs(tau[driving]),
          s=3,
          color=color,
          alpha=0.3,
          linewidths=0,
          label=f"{cmd:g} m/s",
        )
      ax.plot(v_line, env_line, color=INK, lw=2)
      ax.axhline(c1.RATED_TORQUE, color=INK2, lw=1, ls="--")
      ax.set_title(f"{label}: {joint}")
      ax.set_xlim(0, 17)
      ax.set_ylim(0, 125)
      if i == n - 1:
        ax.set_xlabel("|joint speed| (rad/s)")
      if col == 0:
        ax.set_ylabel("|torque| (N·m)")
      leg = ax.legend(loc="upper right", fontsize=8, markerscale=4)
      for h in leg.legend_handles:
        h.set_alpha(1)
  finish(
    fig,
    out / "fig_envelope.png",
    "Torque vs speed while the motor drives the motion (5 ms samples). Solid line:"
    " X8-120 limit (110 N·m cap, then the voltage line). Dashed: 43 N·m rating.",
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("out", type=Path)
  parser.add_argument("--labels", nargs="+")
  args = parser.parse_args()
  runs = load(args.out)
  labels = args.labels or [f"run {i}" for i in sorted(runs)]
  assert len(labels) == len(runs), "one label per run"
  tables(args.out, runs, labels)
  figures(args.out, runs, labels)


if __name__ == "__main__":
  main()
