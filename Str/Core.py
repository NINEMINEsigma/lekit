from pathlib import Path
from typing                         import *
from lekit.Lang.Reflection          import light_reflection

def limit_str(data, max_length=50):
    s:str = data if data is str else str(data)
    if len(s) <= max_length:
        return s
    else:
        inside_str = "\n...\n...\n"
        # 计算头尾部分的长度
        head_length = max_length // 2
        tail_length = max_length - head_length - len(inside_str)  # 3 是省略号的长度
        
        # 截取头尾部分并连接
        return s[:head_length] + inside_str + s[-tail_length:]

def link(symbol:str, strs:list):
    return symbol.join(strs)

class light_str:
    def __init__(self, s:str=""):
        self._str = s
    
    def length(self):
        return len(self._str)
    
    def append(self, s):
        self._str += s
    
    def clear(self):
        self._str = ""
    
    def insert(self, pos, s):
        if pos < 0 or pos > len(self._str):
            raise IndexError("Position out of range")
        self._str = self._str[:pos] + s + self._str[pos:]
    
    def erase(self, pos, length):
        if pos < 0 or pos + length > len(self._str):
            raise IndexError("Position and length out of range")
        self._str = self._str[:pos] + self._str[pos + length:]
    
    def find(self, s, pos=0):
        return self._str.find(s, pos)
    
    def substr(self, pos, length):
        if pos < 0 or pos + length > len(self._str):
            raise IndexError("Position and length out of range")
        return self._str[pos:pos + length]
    
    def replace(self, old, new):
        self._str = self._str.replace(old, new)
    
    def split(self, sep):
        return self._str.split(sep)
    
    def join(self, seq):
        return self._str.join(seq)
    
    def lower(self):
        return self._str.lower()
    
    def upper(self):
        return self._str.upper()
    
    def strip(self):
        return self._str.strip()
    
    def __str__(self):
        return self._str

def UnWrapper(from_) -> str:
    if isinstance(from_, str):
        return from_
    elif isinstance(from_, Path):
        return str(from_)
    elif isinstance(from_, IO):
        return from_.name
    
    ReEx = light_reflection(from_)
    if ReEx.contains_method("to_string"):
        return from_.to_string()
    elif ReEx.contains_method("__str__"):
        return str(from_)
    
    else:
        raise ValueError("Unsupport instance")
    
def Able_UnWrapper(from_) -> bool:
    if isinstance(from_, str):
        return True
    elif isinstance(from_, Path):
        return True
    
    ReEx = light_reflection(from_)
    if ReEx.contains_method("to_string"):
        return True
    elif ReEx.contains_method("__str__"):
        return True
    else:
        return False
    
def Combine(*args) -> str:
    result:str = ""
    if len(args) == 1:
        if isinstance(args[0], Sequence):
            for current in args:
                result += UnWrapper(current)
                result += ","
        else:
            result = UnWrapper(args[0])
    else:
        for current in args:
            result += UnWrapper(current)
    
def list_byte_to_list_string(lines:List[bytes], encoding='utf-8') -> List[str]:
    return [line.decode(encoding) for line in lines]
    
def list_byte_to_string(lines:List[bytes], encoding='utf-8') -> str:
    return "".join(list_byte_to_list_string(lines, encoding))
    
    
    
    