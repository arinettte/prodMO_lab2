"""
Реализация типа данных "Конструктивное число".

Конструктивное число -- пара рациональных чисел (a_low, a_high),
представляющая интервал [a_low, a_high], внутри которого "живёт"
искомое действительное число.

Арифметика построена по правилам интервальной арифметики:
    α + β := (a↓ + b↓,  a↑ + b↑)
    α − β := (a↓ − b↑,  a↑ − b↓)
    α · β := (min{...}, max{...})  -- все четыре произведения
    α / β := α · (1/b↑, 1/b↓)
"""


class ConstructiveNumber:
    """
    Конструктивное число [a_low, a_high].

    Параметры конструктора:
        a_low  -- нижняя граница (рац. число / float)
        a_high -- верхняя граница (рац. число / float)
    """

    def __init__(self, a_low: float, a_high: float):
        self.a_low = float(a_low)
        self.a_high = float(a_high)

        # маленькие числовые ошибки -- терпим, большие -- ошибка
        if self.a_low > self.a_high + 1e-12:
            raise ValueError(
                f"a_low должен быть <= a_high, получено [{a_low}, {a_high}]"
            )
        # гарантируем корректность после округления
        if self.a_low > self.a_high:
            self.a_low, self.a_high = self.a_high, self.a_low

    # ---------- фабричные методы ----------

    @classmethod
    def from_real(cls, x: float, eps: float) -> "ConstructiveNumber":
        """Создать конструктивное число из x ∈ R и ε ∈ R.
        Результат: [x - |ε|, x + |ε|]
        """
        eps = abs(eps)
        return cls(x - eps, x + eps)

    @classmethod
    def from_pair(cls, a: float, b: float) -> "ConstructiveNumber":
        """Создать конструктивное число из пары a, b ∈ Q.
        Автоматически сортирует, чтобы a_low <= a_high.
        """
        return cls(min(a, b), max(a, b))

    # ---------- свойства ----------

    @property
    def eps(self) -> float:
        """Полуширина интервала -- текущая погрешность ε."""
        return (self.a_high - self.a_low) / 2.0

    @property
    def width(self) -> float:
        """Ширина интервала (= 2ε)."""
        return self.a_high - self.a_low

    @property
    def midpoint(self) -> float:
        """Центр интервала."""
        return (self.a_low + self.a_high) / 2.0

    def get_value(self, alpha: float = 0.5) -> float:
        """Получить желаемое вещественное число по параметру α ∈ [0,1].

        α = 0  →  a_low
        α = 1  →  a_high
        α = 0.5 → midpoint (по умолчанию)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha должен быть в [0,1], получено: {alpha}")
        return self.a_low + alpha * (self.a_high - self.a_low)

    # ---------- арифметические операции ----------

    def __add__(self, other):
        if isinstance(other, ConstructiveNumber):
            return ConstructiveNumber(
                self.a_low + other.a_low,
                self.a_high + other.a_high,
            )
        other = float(other)
        return ConstructiveNumber(self.a_low + other, self.a_high + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ConstructiveNumber):
            return ConstructiveNumber(
                self.a_low - other.a_high,
                self.a_high - other.a_low,
            )
        other = float(other)
        return ConstructiveNumber(self.a_low - other, self.a_high - other)

    def __rsub__(self, other):
        other = float(other)
        return ConstructiveNumber(other - self.a_high, other - self.a_low)

    def __mul__(self, other):
        if isinstance(other, ConstructiveNumber):
            products = [
                self.a_low * other.a_low,
                self.a_low * other.a_high,
                self.a_high * other.a_low,
                self.a_high * other.a_high,
            ]
            return ConstructiveNumber(min(products), max(products))
        other = float(other)
        if other >= 0:
            return ConstructiveNumber(self.a_low * other, self.a_high * other)
        else:
            return ConstructiveNumber(self.a_high * other, self.a_low * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ConstructiveNumber):
            if other.a_low <= 0.0 <= other.a_high:
                raise ZeroDivisionError(
                    "Деление на интервал, содержащий нуль"
                )
            # α/β = α * (1/b_high, 1/b_low)
            inv = ConstructiveNumber(1.0 / other.a_high, 1.0 / other.a_low)
            return self * inv
        other = float(other)
        if other == 0.0:
            raise ZeroDivisionError("Деление на нуль")
        if other > 0:
            return ConstructiveNumber(self.a_low / other, self.a_high / other)
        else:
            return ConstructiveNumber(self.a_high / other, self.a_low / other)

    def __rtruediv__(self, other):
        if self.a_low <= 0.0 <= self.a_high:
            raise ZeroDivisionError(
                "Деление на интервал, содержащий нуль"
            )
        other = float(other)
        # other / self = other * (1/a_high, 1/a_low)
        if other >= 0:
            return ConstructiveNumber(other / self.a_high, other / self.a_low)
        else:
            return ConstructiveNumber(other / self.a_low, other / self.a_high)

    def __neg__(self):
        return ConstructiveNumber(-self.a_high, -self.a_low)

    def __pow__(self, n: int):
        """Целочисленная степень."""
        if n == 0:
            return ConstructiveNumber(1.0, 1.0)
        if n == 1:
            return ConstructiveNumber(self.a_low, self.a_high)
        if n == 2:
            # чуть оптимальнее через прямое вычисление
            return self * self
        # для больших степеней -- итеративно
        result = ConstructiveNumber(1.0, 1.0)
        base = ConstructiveNumber(self.a_low, self.a_high)
        for _ in range(abs(n)):
            result = result * base
        if n < 0:
            return 1.0 / result
        return result

    def __abs__(self):
        if self.a_low >= 0:
            return ConstructiveNumber(self.a_low, self.a_high)
        if self.a_high <= 0:
            return ConstructiveNumber(-self.a_high, -self.a_low)
        # интервал пересекает нуль
        return ConstructiveNumber(0.0, max(-self.a_low, self.a_high))

    # ---------- сравнения ----------

    def __lt__(self, other):
        """Строгое: self < other, если a_high < other.a_low."""
        if isinstance(other, ConstructiveNumber):
            return self.a_high < other.a_low
        return self.a_high < float(other)

    def __le__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.a_high <= other.a_low
        return self.a_high <= float(other)

    def __gt__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.a_low > other.a_high
        return self.a_low > float(other)

    def __ge__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.a_low >= other.a_high
        return self.a_low >= float(other)

    def __eq__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.a_low == other.a_low and self.a_high == other.a_high
        return False

    def probably_less(self, other) -> bool:
        """Нежёсткое сравнение по центру интервала (для случая перекрытия)."""
        if isinstance(other, ConstructiveNumber):
            return self.midpoint < other.midpoint
        return self.midpoint < float(other)

    # ---------- преобразования ----------

    def __float__(self) -> float:
        return self.midpoint

    def __repr__(self) -> str:
        return f"CN([{self.a_low:.8g}, {self.a_high:.8g}], ε={self.eps:.3e})"

    def __str__(self) -> str:
        return self.__repr__()
