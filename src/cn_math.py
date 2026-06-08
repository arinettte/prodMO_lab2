"""
Математические функции, совместимые с ConstructiveNumber.

Для вещественного аргумента — стандартные math.*
Для CN — консервативные интервальные оценки:
    |d/dx sin(x)| = |cos(x)| ≤ 1  →  sin([a,b]) ⊆ [sin(mid) - ε, sin(mid) + ε]
    |d/dx cos(x)| = |sin(x)| ≤ 1  →  аналогично
    exp — монотонна → [exp(a_low), exp(a_high)]
    sqrt — монотонна → [sqrt(a_low), sqrt(a_high)]
    round — разрывна → если интервал пересекает 0.5+k, возвращаем расширенный интервал
"""
import math
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from constructive_number import ConstructiveNumber

_CN = ConstructiveNumber


def cn_sin(x):
    if isinstance(x, _CN):
        mid, eps = x.midpoint, x.eps
        return _CN.from_real(math.sin(mid), eps)
    return math.sin(float(x))


def cn_cos(x):
    if isinstance(x, _CN):
        mid, eps = x.midpoint, x.eps
        return _CN.from_real(math.cos(mid), eps)
    return math.cos(float(x))


def cn_exp(x):
    if isinstance(x, _CN):
        # exp монотонна — точные границы
        return _CN(math.exp(x.a_low), math.exp(x.a_high))
    return math.exp(float(x))


def cn_sqrt(x):
    if isinstance(x, _CN):
        lo = math.sqrt(max(0.0, x.a_low))
        hi = math.sqrt(max(0.0, x.a_high))
        return _CN(lo, hi)
    return math.sqrt(max(0.0, float(x)))


def cn_round(x):
    """
    round() — разрывная функция.
    Если интервал пересекает полуцелое число (k + 0.5),
    возвращаем интервал из двух соседних целых значений.
    """
    if isinstance(x, _CN):
        mid = x.midpoint
        r = round(mid)
        # Проверяем ближайшие два полуцелых
        for half in (r - 0.5, r + 0.5):
            if x.a_low < half < x.a_high:
                # Интервал пересекает разрыв
                return _CN(float(min(round(x.a_low), round(x.a_high))),
                           float(max(round(x.a_low), round(x.a_high))))
        return _CN(float(r), float(r))
    return float(round(float(x)))
