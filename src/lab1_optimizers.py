"""
Методы оптимизации 0-го, 1-го и 2-го порядка.

Все методы принимают BlackBox и начальную точку, возвращают
историю сходимости (список состояний на каждой итерации).

История содержит словари с ключами:
    'x'     -- текущая точка (np.ndarray)
    'f'     -- значение функции (float)
    'iter'  -- номер итерации
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Tuple


# ============================================================
#  Вспомогательные функции
# ============================================================

def _to_float_array(x) -> np.ndarray:
    """Конвертировать список (возможно ConstructiveNumber) в float np.ndarray."""
    return np.array([float(xi) for xi in x], dtype=float)


def _norm(v) -> float:
    return float(np.linalg.norm(v))


# ============================================================
#  Метод 0-го порядка: деформируемый многогранник (Нелдер-Мид)
# ============================================================

def nelder_mead(
    f: BlackBox,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 10_000,
    alpha: float = 1.0,    # коэффициент отражения
    gamma: float = 2.0,    # коэффициент растяжения
    rho: float = 0.5,      # коэффициент сжатия
    sigma: float = 0.5,    # коэффициент редукции
) -> List[Dict]:
    """
    Метод деформируемого многогранника (Нелдер-Мид).

    Параметры:
        f        -- целевая функция (BlackBox)
        x0       -- начальная точка (n,)
        tol      -- критерий останова по разбросу значений симплекса
        max_iter -- максимальное число итераций
        alpha, gamma, rho, sigma -- параметры метода

    Возвращает:
        history  -- список словарей с историей сходимости
    """
    n = len(x0)
    x0 = _to_float_array(x0)

    # инициализация симплекса: n+1 вершин
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        # стандартное смещение для инициализации
        xi[i] += 0.05 if x0[i] != 0 else 0.00025
        simplex.append(xi)
    simplex = np.array(simplex)  # shape (n+1, n)

    # вычисляем значения в вершинах
    fvals = np.array([float(f(simplex[i])) for i in range(n + 1)])

    history = []

    for iteration in range(max_iter):
        # сортируем по значению функции
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        best_x = simplex[0]
        best_f = fvals[0]
        worst_f = fvals[-1]
        second_worst_f = fvals[-2]

        history.append({
            'x': best_x.copy(),
            'f': best_f,
            'iter': iteration,
        })

        # критерий останова: стандартное отклонение значений
        if np.std(fvals) < tol:
            break

        # центр масс всех вершин кроме худшей
        centroid = simplex[:-1].mean(axis=0)

        # --- отражение ---
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = float(f(x_r))

        if fvals[0] <= f_r < fvals[-2]:
            # отражение принято
            simplex[-1] = x_r
            fvals[-1] = f_r
            continue

        if f_r < fvals[0]:
            # --- растяжение ---
            x_e = centroid + gamma * (x_r - centroid)
            f_e = float(f(x_e))
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
            continue

        # --- сжатие ---
        if f_r < fvals[-1]:
            x_c = centroid + rho * (x_r - centroid)
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
        f_c = float(f(x_c))

        if f_c < min(f_r, fvals[-1]):
            simplex[-1] = x_c
            fvals[-1] = f_c
            continue

        # --- редукция (уменьшение симплекса) ---
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            fvals[i] = float(f(simplex[i]))

    return history


# ============================================================
#  Метод 1-го порядка: градиентный спуск с постоянным шагом
# ============================================================

def gradient_descent(
    f: BlackBox,
    x0: np.ndarray,
    lr: float = 0.01,
    tol: float = 1e-6,
    max_iter: int = 10_000,
) -> List[Dict]:
    """
    Градиентный спуск с постоянным learning rate.

    x_{k+1} = x_k - lr * ∇f(x_k)

    Параметры:
        f        -- целевая функция (BlackBox с методом grad)
        x0       -- начальная точка
        lr       -- шаг (learning rate)
        tol      -- останов по норме градиента
        max_iter -- максимальное число итераций

    Возвращает историю сходимости.
    """
    x = _to_float_array(x0).copy()
    history = []

    for iteration in range(max_iter):
        fval = float(f(x))
        g = np.array(f.grad(x), dtype=float)
        grad_norm = _norm(g)

        history.append({
            'x': x.copy(),
            'f': fval,
            'grad_norm': grad_norm,
            'iter': iteration,
        })

        if grad_norm < tol:
            break

        x = x - lr * g

    return history


def gradient_descent_armijo(
    f: BlackBox,
    x0: np.ndarray,
    lr0: float = 1.0,
    c: float = 1e-4,
    tau: float = 0.5,
    tol: float = 1e-6,
    max_iter: int = 5_000,
) -> List[Dict]:
    """
    Градиентный спуск с линейным поиском (условие Армихо).
    Автоматически подбирает шаг на каждой итерации.

    Параметры:
        lr0  -- начальный пробный шаг
        c    -- параметр достаточного убывания
        tau  -- множитель при уменьшении шага
    """
    x = _to_float_array(x0).copy()
    history = []

    for iteration in range(max_iter):
        fval = float(f(x))
        g = np.array(f.grad(x), dtype=float)
        grad_norm = _norm(g)

        history.append({
            'x': x.copy(),
            'f': fval,
            'grad_norm': grad_norm,
            'iter': iteration,
        })

        if grad_norm < tol:
            break

        # линейный поиск по Армихо
        lr = lr0
        for _ in range(50):
            x_new = x - lr * g
            if float(f(x_new)) <= fval - c * lr * grad_norm ** 2:
                break
            lr *= tau
        else:
            # если поиск не сошёлся -- берём маленький шаг
            lr = 1e-8

        x = x - lr * g

    return history


# ============================================================
#  Метод 2-го порядка: метод Ньютона
# ============================================================

def newton_method(
    f: BlackBox,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1_000,
    regularize: float = 1e-8,
) -> List[Dict]:
    """
    Метод Ньютона: x_{k+1} = x_k - H(x_k)^{-1} · ∇f(x_k).

    Для надёжности добавляем регуляризацию гессиана:
        H_reg = H + λ·I, чтобы избежать вырожденности.

    Параметры:
        f           -- целевая функция (BlackBox с методами grad и hess)
        x0          -- начальная точка
        tol         -- останов по норме градиента
        max_iter    -- максимальное число итераций
        regularize  -- начальный коэффициент регуляризации
    """
    x = _to_float_array(x0).copy()
    n = len(x)
    history = []

    for iteration in range(max_iter):
        fval = float(f(x))
        g = np.array(f.grad(x), dtype=float)
        H = np.array(f.hess(x), dtype=float)
        grad_norm = _norm(g)

        history.append({
            'x': x.copy(),
            'f': fval,
            'grad_norm': grad_norm,
            'iter': iteration,
        })

        if grad_norm < tol:
            break

        # регуляризация: если гессиан вырожден, добавляем λI
        lam = regularize
        for _ in range(50):
            H_reg = H + lam * np.eye(n)
            try:
                direction = np.linalg.solve(H_reg, g)
                break
            except np.linalg.LinAlgError:
                lam *= 10
        else:
            direction = g  # fallback к градиентному спуску

        # шаг Ньютона с простым line search
        step = 1.0
        for _ in range(30):
            x_new = x - step * direction
            if float(f(x_new)) < fval:
                break
            step *= 0.5

        x = x - step * direction

    return history