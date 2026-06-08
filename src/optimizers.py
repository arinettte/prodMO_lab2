"""
Стохастические методы оптимизации.

SimulatedAnnealing  — метод отжига (Metropolis-Hastings)
ParticleSwarm       — метод роя частиц (PSO, gbest-топология)

Оба метода:
  - принимают BlackBox с интерфейсом __call__(x)
  - возвращают историю: список {'x', 'f', 'iter'}
  - совместимы с ConstructiveNumber через BlackBox
"""
import numpy as np
import math
from typing import List, Dict, Optional


def _to_float(x) -> np.ndarray:
    return np.array([float(xi) for xi in x], dtype=float)


# ============================================================
#  Simulated Annealing
# ============================================================

def simulated_annealing(
    f,
    x0,
    bounds=None,
    T0: float = 10.0,
    alpha: float = 0.995,
    n_iter: int = 10_000,
    sigma: float = 1.0,
    tol: float = 1e-8,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Метод имитации отжига.

    Схема:
      1. Случайный сосед: x_new = x + N(0, σ)
      2. Принятие: если f_new < f_old — всегда; иначе с P = exp(-ΔE / T)
      3. Охлаждение: T_k = T_0 · α^k

    Параметры:
        T0     — начальная температура
        alpha  — коэффициент охлаждения (0 < α < 1)
        n_iter — максимальное число итераций
        sigma  — масштаб шума при генерации соседа
        bounds — [(lo, hi), ...] — границы; None = без ограничений
        seed   — зерно генератора для воспроизводимости
    """
    rng = np.random.RandomState(seed)
    x = _to_float(x0).copy()
    fval = float(f(x))
    best_x, best_f = x.copy(), fval
    T = T0

    if bounds is not None:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

    history = [{'x': best_x.copy(), 'f': best_f, 'iter': 0, 'T': T}]

    for k in range(1, n_iter + 1):
        x_new = x + rng.randn(len(x)) * sigma
        if bounds is not None:
            x_new = np.clip(x_new, lo, hi)

        f_new = float(f(x_new))
        dE = f_new - fval

        if dE < 0 or rng.rand() < math.exp(-dE / max(T, 1e-12)):
            x, fval = x_new, f_new

        if fval < best_f:
            best_x, best_f = x.copy(), fval

        T *= alpha
        history.append({'x': best_x.copy(), 'f': best_f, 'iter': k, 'T': T})

        if best_f < tol:
            break

    return history


# ============================================================
#  Particle Swarm Optimization
# ============================================================

def particle_swarm(
    f,
    bounds,
    n_particles: int = 30,
    n_iter: int = 500,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    tol: float = 1e-8,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Метод роя частиц (PSO).

    Каждая частица i:
        v_i ← w·v_i + c1·r1·(pbest_i − x_i) + c2·r2·(gbest − x_i)
        x_i ← x_i + v_i

    Параметры:
        bounds      — [(lo, hi), ...] — обязательные границы поиска
        n_particles — число частиц
        w           — инерционный вес
        c1          — когнитивный коэффициент (личный лучший)
        c2          — социальный коэффициент (глобальный лучший)
        seed        — зерно генератора
    """
    rng = np.random.RandomState(seed)
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Инициализация: равномерно в bounds
    pos = rng.uniform(lo, hi, (n_particles, n))
    vel = rng.uniform(-(hi - lo), hi - lo, (n_particles, n)) * 0.1

    fvals = np.array([float(f(pos[i])) for i in range(n_particles)])
    pbest_pos = pos.copy()
    pbest_val = fvals.copy()

    gbest_idx = np.argmin(fvals)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    history = [{'x': gbest_pos.copy(), 'f': gbest_val, 'iter': 0}]

    for k in range(1, n_iter + 1):
        r1 = rng.rand(n_particles, n)
        r2 = rng.rand(n_particles, n)

        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        pos = np.clip(pos + vel, lo, hi)

        fvals = np.array([float(f(pos[i])) for i in range(n_particles)])

        improved = fvals < pbest_val
        pbest_pos[improved] = pos[improved]
        pbest_val[improved] = fvals[improved]

        best_idx = np.argmin(pbest_val)
        if pbest_val[best_idx] < gbest_val:
            gbest_pos = pbest_pos[best_idx].copy()
            gbest_val = pbest_val[best_idx]

        history.append({'x': gbest_pos.copy(), 'f': gbest_val, 'iter': k})

        if gbest_val < tol:
            break

    return history


# ============================================================
#  Оценка памяти (теоретическая)
# ============================================================

def memory_bytes(method: str, n: int, n_particles: int = 30) -> int:
    """
    Теоретическая оценка памяти метода в байтах (float64 = 8 байт).

    SA:  x_cur + x_best                      = 2n floats
    PSO: pos + vel + pbest + gbest            = (3·N_p + 1)·n floats
    NM:  симплекс (n+1) вершин               = (n+1)·n floats
    GD:  x + grad                            = 2n floats
    """
    B = 8  # float64
    m = {
        'SA':  2 * n,
        'PSO': (3 * n_particles + 1) * n,
        'NM':  (n + 1) * n,
        'GD':  2 * n,
        'GDA': 2 * n,
    }
    return m.get(method, 0) * B
