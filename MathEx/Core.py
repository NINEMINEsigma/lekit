from typing                         import *
from lekit.Internal                 import *
import numpy                        as     np
import scipy                        as     sci_base
from scipy          import optimize as     sci_opt

BasicIntFloatNumber     = Union[int, float]
NpSignedIntNumber       = Union[
    np.int8,
    np.int16,
    np.int32,
    np.int64,
]
NpUnsignedIntNumber     = Union[
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
NpFloatNumber           = Union[
    np.float16,
    np.float32,
    np.float64,
]
NumberTypeOf_Float_Int_And_More = Union[
    BasicIntFloatNumber,
    NpSignedIntNumber,
    NpUnsignedIntNumber,
    NpFloatNumber
]
NumberLike              = NumberTypeOf_Float_Int_And_More
class left_number_reference(left_value_reference[NumberLike]):
    def __add__(self, value:NumberLike):
        return self.ref_value + value
    def __sub__(self, value:NumberLike):
        return self.ref_value - value
    def __mul__(self, value:NumberLike):
        return self.ref_value * value
    def __truediv__(self, value:NumberLike):
        return self.ref_value / value
    def __floordiv__(self, value:NumberLike):
        return self.ref_value // value
    def __mod__(self, value:NumberLike):
        return self.ref_value % value
    def __pow__(self, value:NumberLike):
        return self.ref_value ** value
    def __radd__(self, value:NumberLike):
        return value + self.ref_value
    def __rsub__(self, value:NumberLike):
        return value - self.ref_value
    def __rmul__(self, value:NumberLike):
        return value * self.ref_value
    def __rtruediv__(self, value:NumberLike):
        return value / self.ref_value
    def __rfloordiv__(self, value:NumberLike):
        return value // self.ref_value
    def __rmod__(self, value:NumberLike):
        return value % self.ref_value
    def __rpow__(self, value:NumberLike):
        return value ** self.ref_value
    def __neg__(self):
        return -self.ref_value
    def __pos__(self):
        return self.ref_value
    def __abs__(self):
        return abs(self.ref_value)
    def __round__(self, ndigits:Optional[int]=None):
        return round(self.ref_value, ndigits)
    def __eq__(self, value:NumberLike):
        return self.ref_value == value
    def __ne__(self, value:NumberLike):
        return self.ref_value != value
    def __lt__(self, value:NumberLike):
        return self.ref_value < value
    def __le__(self, value:NumberLike):
        return self.ref_value <= value
    def __gt__(self, value:NumberLike):
        return self.ref_value > value
    def __ge__(self, value:NumberLike):
        return self.ref_value >= value
    def __bool__(self):
        return bool(self.ref_value)
    def __hash__(self):
        return hash(self.ref_value)
    def __repr__(self):
        return str(self.ref_value)
    def __str__(self):
        return str(self.ref_value)


NumberInstanceOrContainer = Union[
    NumberLike,
    Sequence[NumberLike],
    Dict[Any, NumberLike]
]

NumberInside = NumberLike

def clamp_without_check(
    value:  NumberLike,
    left:   NumberLike,
    right:  NumberLike
    ) -> NumberLike:
    return max(left, min(value, right))
def clamp(
    value:  NumberLike,
    a:      NumberLike,
    b:      NumberLike
    ) -> NumberLike:
    if a<b:
        return clamp_without_check(value, a, b)
    else:
        return clamp_without_check(value, b, a)
def clamp_sequence(
    values: Sequence[NumberLike],
    a:      NumberLike,
    b:      NumberLike
    ) -> Sequence[NumberLike]:
    if a<b:
        return [clamp_without_check(value, a, b) for value in values]
    else:
        return [clamp_without_check(value, b, a) for value in values]
def clamp_dict(
    values: Dict[Any, NumberLike],
    a:      NumberLike,
    b:      NumberLike
    ) -> Dict[str, NumberLike]:
    if a<b:
        return {key: clamp_without_check(value, a, b) for key, value in values.items()}
    else:
        return {key: clamp_without_check(value, b, a) for key, value in values.items()}
def make_clamp(
    value_or_values:    NumberInstanceOrContainer,
    a:                  NumberLike,
    b:                  NumberLike
    ) -> Union[NumberLike, Sequence[NumberLike], Dict[str, NumberLike]]:
    if isinstance(value_or_values, NumberLike):
        return clamp(value_or_values, a, b)
    elif isinstance(value_or_values, Sequence):
        return clamp_sequence(value_or_values, a, b)
    elif isinstance(value_or_values, Dict):
        return clamp_dict(value_or_values, a, b)
    else:
        raise TypeError("value_or_values must be NumberLike, Sequence or Dict")

NumberBetween01 = Union[float, NpFloatNumber]
def clamp01(value: NumberLike) -> NumberBetween01:
    return make_clamp(value, 0, 1)

EquationsLike = Callable[[Sequence[float]], Sequence[float]]
class EquationsCallable(EquationsLike):
    def __init__(self, equations:EquationsLike, args_size):
        self.__equations = equations
        self.param_size = args_size
    def __call__(self, args:Sequence[float]) -> Sequence[float]:
        return self.__equations(args)

def make_equations(equations:Sequence[Callable[[float], float]]) -> EquationsCallable:
    def equations_solution(args:Sequence[float]) -> Sequence[float]:
        result = []
        for index, equation in enumerate(equations):
            result.append(equation(args[index]))
        return result
    # Closer
    return EquationsCallable(equations_solution, len(equations))

class solver:
    def __init__(self, **kwargs):
        self.__config = kwargs

    def opt_fsolve(
        self,
        equations:      EquationsLike,
        initer:         Optional[Union[Sequence[float], float]] = None,
        **kwargs
        ) -> Sequence[float]:
        if initer is None:
            equations:EquationsCallable = equations
            initer = [0.001 for _ in range(equations.param_size)]
        elif isinstance(initer, float):
            equations:EquationsCallable = equations
            initer = [initer for _ in range(equations.param_size)]
        return sci_opt.fsolve(equations, initer, **kwargs)


if __name__ == "__main__":
    solver_i = solver()
    equs = make_equations([lambda x: x**2 - 1])
    print(solver_i.opt_fsolve(equs))
