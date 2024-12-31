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
        return self.GetType().__name__
class left_value_reference[_T](type_class):
    def __init__(self, ref_value:_T):
        self.__ref_value = ref_value
    @property
    def ref_value(self):
        return self.__ref_value
    @ref_value.setter
    def ref_value(self, value):
        if value is None or isinstance(value, _T):
            self.__ref_value = value
        else:
            raise TypeError(f"Cannot assign {type(value)} to {_T}")
    def _clear_ref_value(self):
        self.__ref_value = None
    @override
    def GetType(self):
        return _T
    @override
    def SymbolName(self) -> str:
        if self.__ref_value is None:
            return "null"
        return f"{self.GetType().__name__}&"
class right_value_refenence[_T](type_class):
    def __init__(self, ref_value):
        self.__ref_value = ref_value
    @property
    def ref_value(self):
        result = self.__ref_value
        self.__ref_value = None
        return result
    @override
    def GetType(self):
        return _T
    @override
    def SymbolName(self) -> str:
        if self.__ref_value is None:
            return "null"
        return f"{self.GetType().__name__}&&"
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
    print(type(None))



