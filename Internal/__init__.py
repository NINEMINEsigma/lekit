from typing import *
from abc    import *

def ImportingThrow(
    ex:             ImportError,
    moduleName:     str,
    requierds:      Sequence[str],
    *,
    messageBase:    str = "{module} Module requires {required} package.",
    installBase:    str = "\tpip install {name}"
    ):
        requierds_str = ",".join([f"<{r}>" for r in requierds])
        print(messageBase.format_map(dict(module=moduleName, required=requierds_str)))
        print('Install it via command:')
        for i in requierds:
            print(installBase.format_map(name=i))
        if ex:
            raise ex

def InternalImportingThrow(
    moduleName:     str,
    requierds:      Sequence[str],
    *,
    messageBase:    str = "{module} Module requires internal lekit package: {required}.",
    ):
        requierds_str = ",".join([f"<{r}>" for r in requierds])
        print(f"Internal lekit package is not installed.\n{messageBase.format_map(dict(module=moduleName, required=requierds_str))}")

def static_cast[_T](from_) -> _T:
    return _T(from_)
def dynamic_cast[_T](from_) -> Optional[_T]:
    if isinstance(from_, _T):
        return from_
    return None
def reinterpret_cast[_T](from_) -> _T:
    raise NotImplementedError("Python does not support reinterpret_cast anyways")

type Action[_T] = Callable[[], _T]

class type_class(ABC):
    def GetType(self):
        return type(self)
    def SymbolName(self) -> str:
        return self.GetType().__name__
    def ToString(self) -> str:
        return str(self.GetType())
class base_value_reference[_T](type_class):
    def __init__(self, ref_value:_T):
        self._ref_value = ref_value
        self.__real_type = type(ref_value)
    def _clear_ref_value(self):
        self._ref_value = None
    @override
    def GetType(self):
        return self.__real_type
    @override
    def SymbolName(self) -> str:
        if self._ref_value is None:
            return "null"
        return f"{self.GetType().__name__}&"
    @override
    def ToString(self) -> str:
        if self._ref_value is None:
            return "null"
        return str(self._ref_value)
class left_value_reference[_T](base_value_reference):
    def __init__(self, ref_value:_T):
        super().__init__(ref_value)
    @property
    def ref_value(self) -> _T:
        return self._ref_value
    @ref_value.setter
    def ref_value(self, value) -> _T:
        if value is None or isinstance(value, self.GetType()):
            self._ref_value = value
        else:
            raise TypeError(f"Cannot assign {type(value)} to {self.GetType()}")
        return value
class right_value_refenence[_T](type_class):
    def __init__(self, ref_value:_T):
        super().__init__(ref_value)
    @property
    def ref_value(self) -> _T:
        result = self.__ref_value
        self.__ref_value = None
        return result
    @property
    def const_ref_value(self) -> _T:
        return self.__ref_value
class any_class(type_class, ABC):
    def AsRef[_T](self):
        return dynamic_cast[_T](self)
    def AsValue[_T](self):
        '''
        warning: this will be a real transform, it is not a reference to the object
        '''
        return static_cast[_T](self)
    def Fetch[_T](self, value:_T) -> _T:
        return value
    def Share[_T](self, out_value:left_value_reference[_T]) -> Self:
        if out_value is None:
            raise ValueError("out_value cannot be None")
        out_value.ref_value = self.AsRef[_T]()
        return self
    def Is[_T](self) -> bool:
        return isinstance(self, _T)
    def IfIam[_T, _Ret](self, action:Callable[[_T], _Ret]) -> Union[Self, _Ret]:
        if self.Is[_T]() is False:
            return self
        if _Ret == type(None):
            return action(self)
        else:
            action(self)
        return self
    def AsIam[_T, _Ret](self, action:Callable[[_T], _Ret]) -> Union[Self, _Ret]:
        if _Ret == type(None):
            action(self.AsRef[_T]())
            return self
        return action(self.AsRef[_T]())



if __name__ == "__main__":
    ref = left_value_reference[int](5.5)
    print(ref.ToString())
    print(ref.GetType())
    print(ref.SymbolName())
    print(ref.ref_value)
    print(ref)



