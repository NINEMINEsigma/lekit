# -*- coding: utf-8 -*-

from pydantic import BaseModel as Base

#不要继承这个类, 直到pydantic将私有变量相关的__pydantic_private__错误修复为止
#除非继承的类不打算使用私有或者保护变量
class BaseBehaviour(Base):
    def __init__(self):
        pass
    def __del__(self):
        pass