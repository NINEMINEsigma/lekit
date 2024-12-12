

def limit_str(data, max_length=50):
    s:str = data if data is str else str(data)
    if len(s) <= max_length:
        return s
    else:
        # 计算头尾部分的长度
        head_length = max_length // 2
        tail_length = max_length - head_length - 3  # 3 是省略号的长度
        
        # 截取头尾部分并连接
        return s[:head_length] + "..." + s[-tail_length:]
