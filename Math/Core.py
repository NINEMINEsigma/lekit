from typing                         import *
import numpy                        as     base
import scipy                        as     sci
from scipy          import optimize as     sci_opt

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
