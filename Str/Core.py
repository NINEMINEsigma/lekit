

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