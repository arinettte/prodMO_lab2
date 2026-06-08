"""
Тестовые функции для оптимизации.

Все функции совместимы с ConstructiveNumber благодаря перегрузке
операторов и вспомогательным функциям cn_math.

Rastrigin-2:   min f = 0 в (0, 0), область [-5.12, 5.12]²
Ackley-2:      min f = 0 в (0, 0), область [-5, 5]²
Desmos:        из https://www.desmos.com/3d/cfh7o2ckkx, 2D
"""
import math
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from cn_math import cn_cos, cn_sin, cn_exp, cn_sqrt, cn_round

PI = math.pi


# ============================================================
# Вспомогательный класс чёрного ящика (дублируем из lab_01)
# ============================================================

class BlackBox:
    def __init__(self, func, grad, name, n_args, bounds=None):
        self._func = func
        self._grad = grad
        self.name = name
        self.n_args = n_args
        self.bounds = bounds          # [(lo, hi), ...] или None
        self.n_func_calls = 0
        self.n_grad_calls = 0

    def __call__(self, x):
        self.n_func_calls += 1
        return self._func(x)

    def grad(self, x):
        if self._grad is None:
            raise NotImplementedError(f"{self.name}: градиент не определён")
        self.n_grad_calls += 1
        return self._grad(x)

    def reset_counters(self):
        self.n_func_calls = 0
        self.n_grad_calls = 0

    def __repr__(self):
        return (f"BlackBox('{self.name}', dim={self.n_args}, "
                f"f_calls={self.n_func_calls}, ∇f_calls={self.n_grad_calls})")


# ============================================================
# 1. Функция Растригина (2D)
# ============================================================
# f(x,y) = 20 + x² - 10cos(2πx) + y² - 10cos(2πy)
# Глобальный минимум: f(0,0) = 0
# Сильно мультимодальная, ~(2*5.12/0.5)² ≈ 400 локальных минимумов

def _rastrigin_f(x):
    result = float(10 * len(x))
    for xi in x:
        result = result + xi * xi - 10 * cn_cos(2 * PI * xi)
    return result


def _rastrigin_grad(x):
    return [2 * float(xi) + 20 * PI * math.sin(2 * PI * float(xi))
            for xi in x]


def make_rastrigin(n=2):
    return BlackBox(
        func=_rastrigin_f,
        grad=_rastrigin_grad,
        name=f"Rastrigin-{n}",
        n_args=n,
        bounds=[(-5.12, 5.12)] * n,
    )


# ============================================================
# 2. Функция Экли (2D)
# ============================================================
# f(x,y) = -20·exp(-0.2·√(0.5(x²+y²))) - exp(0.5(cos2πx+cos2πy)) + e + 20
# Глобальный минимум: f(0,0) = 0
# Почти плоская внешняя область + резкая яма в центре

def _ackley_f(x):
    n = len(x)
    sum_sq = x[0] * x[0]
    sum_cos = cn_cos(2 * PI * x[0])
    for xi in x[1:]:
        sum_sq = sum_sq + xi * xi
        sum_cos = sum_cos + cn_cos(2 * PI * xi)

    term1 = -20 * cn_exp(-0.2 * cn_sqrt(sum_sq / n))
    term2 = -cn_exp(sum_cos / n)
    return term1 + term2 + math.e + 20


def _ackley_grad(x):
    xf = [float(xi) for xi in x]
    n = len(xf)
    s1 = sum(xi ** 2 for xi in xf)
    s2 = sum(math.cos(2 * PI * xi) for xi in xf)
    r = math.sqrt(s1 / n + 1e-12)
    grad = []
    for xi in xf:
        d1 = 20 * 0.2 * (xi / n) / r * math.exp(-0.2 * r)
        d2 = -2 * PI * math.sin(2 * PI * xi) / n * math.exp(s2 / n)
        grad.append(d1 + d2)
    return grad


def make_ackley(n=2):
    return BlackBox(
        func=_ackley_f,
        grad=_ackley_grad,
        name=f"Ackley-{n}",
        n_args=n,
        bounds=[(-5.0, 5.0)] * n,
    )


# ============================================================
# 3. Функция из Desmos (2D, НЕ непрерывная)
# ============================================================
# z = ((x·(round(sin(10y))+2))² + y − 10)² + (x + (y·(round(sin(7x))+2))² − 7)²
# Разрывна из-за round(). Градиент не определён.
# Приближённый глобальный минимум: z ≈ 0 около (x,y) ~ (2-3, 7-10)

def _desmos_f(x):
    xv, yv = x[0], x[1]
    r1 = cn_round(cn_sin(10 * yv)) + 2   # CN или float
    r2 = cn_round(cn_sin(7 * xv)) + 2
    a = xv * r1                            # x·(round(sin(10y))+2)
    b = yv * r2                            # y·(round(sin(7x))+2)
    t1 = a * a + yv - 10                   # (...)² + y - 10
    t2 = xv + b * b - 7                   # x + (...)² - 7
    return t1 * t1 + t2 * t2


def make_desmos():
    return BlackBox(
        func=_desmos_f,
        grad=None,
        name="Desmos",
        n_args=2,
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
    )
