from typing import *
from lekit.Str.Core import UnWrapper

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
        result[UnWrapper(index)] = i
        index += 1
    for key in kwargs:
        result[UnWrapper(key)] = kwargs[key]
    return result