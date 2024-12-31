from types import NoneType
from typing import *
from lekit.Internal import *

# string

string = str
def to_string(target) -> str:
    return str(target)

# make
def make_tuple(*args) -> tuple:
    return args
def make_pair(first, second) -> tuple:
    return (first, second)
def make_list(*args) -> list:
    result:list = []
    for i in args:
        result.append(i)
    return result
def make_dict(*args, **kwargs) -> dict:
    result:dict = {}
    index = 0
    for i in args:
        result[index] = i
        index += 1
    for key in kwargs:
        result[key] = kwargs[key]
    return result
def make_map(*args, **kwargs) -> Dict[str, Any]:
    result:dict = {}
    index = 0
    for i in args:
        result[to_string(index)] = i
        index += 1
    for key in kwargs:
        result[to_string(key)] = kwargs[key]
    return result

# LightDiagram::ld::instance<_Ty>
class ld_instance:
    def __init__(self, target, *, constructor_func:Callable[[object], NoneType]=None, destructor_func:Callable[[object], NoneType]=None):
        self.target = target
        if constructor_func:
            constructor_func(self.target)
        self.destructor_func = destructor_func
    def __del__(self):
        if self.destructor_func:
            self.destructor_func(self.target)
    def __getitem__(self, key):
        return self.target[key]
    def __setitem__(self, key, value):
        self.target[key] = value
    def __delitem__(self, key):
        del self.target[key]
    def __iter__(self):
        return iter(self.target)
    def __len__(self):
        return len(self.target)
    def __contains__(self, key):
        return key in self.target
    def __str__(self):
        return str(self.target)

    def get_ref(self):
        return self.target