"""Plot the X8-120 speed-torque envelope used by the C1 actuator model.

The voltage-limit line runs through the two datasheet points, rated
(43 N*m, 127 RPM) and no-load (0 N*m, 158 RPM), which is the same line
BuiltinDcMotorActuatorCfg uses. The peak envelope is that line capped at the
120 N*m datasheet peak; simulation caps torque lower, at TORQUE_CAP.

  uv run python src/mjlab/scripts/plot_motor.py [--out motor_envelope.pdf]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from mjlab.asset_zoo.robots.caltech_c1.c1_constants import (
  NO_LOAD_SPEED,
  PEAK_TORQUE,
  RATED_SPEED,
  RATED_TORQUE,
  TORQUE_CAP,
  VOLTAGE_LINE_STALL_TORQUE,
)


def voltage_line(w: np.ndarray | float) -> np.ndarray | float:
  return VOLTAGE_LINE_STALL_TORQUE * (1.0 - np.asarray(w) / NO_LOAD_SPEED)


def corner_speed(torque: float) -> float:
  """Speed at which the voltage line falls to the given torque."""
  return NO_LOAD_SPEED * (1.0 - torque / VOLTAGE_LINE_STALL_TORQUE)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", default="motor_envelope.pdf")
  args = parser.parse_args()

  plt.rcParams.update(
    {
      "font.family": "serif",
      "font.serif": ["STIXGeneral"],
      "mathtext.fontset": "stix",
      "font.size": 10,
    }
  )

  w = np.linspace(0.0, NO_LOAD_SPEED, 400)
  w_peak = corner_speed(PEAK_TORQUE)
  w_cap = corner_speed(TORQUE_CAP)

  fig, ax = plt.subplots(figsize=(3.5, 2.6))  # ~half text width
  # Sim samples, if overlaid:
  # ax.scatter(np.abs(qd), np.abs(tau), s=1, alpha=0.1, c="0.5", rasterized=True)

  ax.plot(w, np.minimum(PEAK_TORQUE, voltage_line(w)), "k", lw=1.5)
  ax.plot(
    [0.0, w_cap],
    [TORQUE_CAP, TORQUE_CAP],
    ":",
    c="k",
    lw=1.2,
  )
  # The continuous rating only holds up to rated speed; beyond it the voltage
  # line is the binding limit.
  ax.plot(
    [0.0, RATED_SPEED],
    [RATED_TORQUE, RATED_TORQUE],
    "--",
    c="k",
    lw=1,
  )

  ax.plot(
    [RATED_SPEED, NO_LOAD_SPEED],
    [RATED_TORQUE, 0.0],
    "ko",
    ms=4,
    zorder=3,
    clip_on=False,
  )  # datasheet points
  ax.plot(w_peak, PEAK_TORQUE, "o", mfc="white", mec="k", ms=4, zorder=3)

  small = 8
  ax.annotate(
    f"Rated: {RATED_TORQUE:.0f} N$\\cdot$m, {RATED_SPEED:.1f} rad/s",
    (RATED_SPEED, RATED_TORQUE),
    xytext=(-5, -4),
    textcoords="offset points",
    fontsize=small,
    ha="right",
    va="top",
  )
  ax.annotate(
    f"No-load\n{NO_LOAD_SPEED:.2f} rad/s",
    (NO_LOAD_SPEED, 0.0),
    xytext=(0, 30),
    textcoords="offset points",
    fontsize=small,
    ha="center",
    va="bottom",
    arrowprops={"arrowstyle": "-", "lw": 0.6, "shrinkA": 1, "shrinkB": 3},
  )
  ax.annotate(
    f"Corner: {w_peak:.1f} rad/s",
    (w_peak, PEAK_TORQUE),
    xytext=(6, 0),
    textcoords="offset points",
    fontsize=small,
    ha="left",
    va="center",
  )

  # Direct labels on each line instead of a legend.
  ax.text(0.3, PEAK_TORQUE + 2, "Peak envelope", fontsize=small, va="bottom")
  ax.text(
    0.3,
    TORQUE_CAP - 2,
    f"Sim. cap, {TORQUE_CAP:.0f} N$\\cdot$m",
    fontsize=small,
    va="top",
  )
  ax.text(0.3, RATED_TORQUE + 2, "Continuous rating", fontsize=small, va="bottom")

  ax.set(
    xlabel="Joint speed (rad/s)",
    ylabel="Torque (N$\\cdot$m)",
    xlim=(0, 20),
    ylim=(0, 135),
  )
  fig.tight_layout()
  fig.savefig(args.out)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
