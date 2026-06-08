"""
Утилиты: визуализация, сравнительная таблица, отслеживание ε.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import time
import sys, os
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from constructive_number import ConstructiveNumber
from optimizers import memory_bytes


# ── Запуск с замером времени ──────────────────────────────────────────────────

def run_timed(method_fn, *args, n_runs=3, **kwargs):
    """Запустить метод n_runs раз, вернуть (history, best_time_ms)."""
    best_t, best_h = float('inf'), None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        h = method_fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        if dt < best_t:
            best_t, best_h = dt, h
    return best_h, best_t * 1000


# ── Отслеживание ε вдоль траектории ──────────────────────────────────────────

def track_eps(f_raw, history: List[Dict], eps0: float) -> List[float]:
    """Вычислить ε выходного значения f для каждой точки истории."""
    eps_list = []
    for state in history:
        try:
            x_cn = [ConstructiveNumber.from_real(float(xi), eps0)
                    for xi in state['x']]
            result = f_raw(x_cn)
            eps_list.append(float(result.eps) if isinstance(result, ConstructiveNumber)
                            else 0.0)
        except Exception:
            eps_list.append(float('nan'))
    return eps_list


# ── Графики ──────────────────────────────────────────────────────────────────

def plot_convergence(histories: Dict[str, List[Dict]],
                     title: str = "", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for name, hist in histories.items():
        fvals = [s['f'] for s in hist]
        ax.semilogy(fvals, label=name)
    ax.set_xlabel("Итерация"); ax.set_ylabel("f (log)")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def plot_function_2d(f_raw, bounds, title="", n=200, ax=None, log_scale=False):
    """Нарисовать тепловую карту 2D функции."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    xs = np.linspace(bounds[0][0], bounds[0][1], n)
    ys = np.linspace(bounds[1][0], bounds[1][1], n)
    Z = np.zeros((n, n))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            try:
                Z[i, j] = float(f_raw([x, y]))
            except Exception:
                Z[i, j] = float('nan')
    norm = mc.LogNorm(vmin=max(Z[np.isfinite(Z)].min(), 1e-6)) if log_scale else None
    ax.contourf(xs, ys, Z, levels=30, cmap='viridis', norm=norm, alpha=0.8)
    ax.contour(xs, ys, Z, levels=15, colors='white', linewidths=0.4, alpha=0.4)
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    return ax


def plot_trajectory(history: List[Dict], label: str, ax, color=None, alpha=0.7):
    xs = [s['x'][0] for s in history]
    ys = [s['x'][1] for s in history]
    ax.plot(xs, ys, '.-', markersize=3, linewidth=1,
            label=label, color=color, alpha=alpha)
    ax.plot(xs[0], ys[0], 'o', markersize=6, color=color)
    ax.plot(xs[-1], ys[-1], '*', markersize=10, color=color)


# ── Сводная таблица ───────────────────────────────────────────────────────────

def summary_table(results: Dict[str, Dict]):
    """
    results: {key: {'f': float, 'iters': int, 'f_calls': int,
                    'time_ms': float, 'mem_bytes': int}}
    """
    w = [20, 12, 10, 10, 12, 12]
    hdr = (f"{'Метод / Функция':<{w[0]}} {'f*':>{w[1]}} {'Итер':>{w[2]}}"
           f" {'f calls':>{w[3]}} {'Время,мс':>{w[4]}} {'Память,Б':>{w[5]}}")
    print(hdr)
    print('─' * sum(w))
    for key, r in results.items():
        print(f"{key:<{w[0]}} {r['f']:{w[1]}.4e} {r['iters']:{w[2]}}"
              f" {r['f_calls']:{w[3]}} {r['time_ms']:{w[4]}.1f}"
              f" {r['mem_bytes']:{w[5]}}")
