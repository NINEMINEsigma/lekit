from types          import TracebackType
from typing         import *
from abc            import *
import                     sys
from pydantic       import BaseModel
import                     threading
import                     traceback

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
            print(installBase.format_map({"name":i}))
        if ex:
            print(ex)
            #raise ex

def InternalImportingThrow(
    moduleName:     str,
    requierds:      Sequence[str],
    *,
    messageBase:    str = "{module} Module requires internal lekit package: {required}.",
    ):
        requierds_str = ",".join([f"<{r}>" for r in requierds])
        print(f"Internal lekit package is not installed.\n{messageBase.format_map(dict(module=moduleName, required=requierds_str))}")

false = False
true = True

type Typen[_T] = type

def static_cast[_T](typen:Typen[_T], from_) -> _T:
    return typen(from_)
def dynamic_cast[_T](typen:Typen[_T], from_) -> Optional[_T]:
    if isinstance(from_, typen):
        return from_
    return None
def reinterpret_cast[_T](typen:Typen[_T], from_) -> _T:
    raise NotImplementedError("Python does not support reinterpret_cast anyways")

def any_if[_T](iter:Iterator[_T], pr:Callable[[_T], bool]) -> bool:
    for i in iter:
        if pr(i):
            return True
    return False
def all_if[_T](iter:Iterator[_T], pr:Callable[[_T], bool]) -> bool:
    for i in iter:
        if not pr(i):
            return False
    return True

type Action[_T] = Callable[[_T], None]
type Action2[_T1, _T2] = Callable[[_T1, _T2], None]
type Action3[_T1, _T2, _T3] = Callable[[_T1, _T2, _T3], None]
type Action4[_T1, _T2, _T3, _T4] = Callable[[_T1, _T2, _T3, _T4], None]
type Action5[_T1, _T2, _T3, _T4, _T5] = Callable[[_T1, _T2, _T3, _T4, _T5], None]
type ActionW = Callable[[Sequence[Any]], None]
type ClosuresCallable[_T] = Union[Callable[[Optional[None]], _T], Typen[_T]]

def format_traceback_info():
    return ''.join(traceback.format_stack()[:-1])

class type_class(ABC):
    def __init__(self):
        self.generate_trackback = format_traceback_info()
    def GetType(self):
        return type(self)
    def SymbolName(self) -> str:
        return self.GetType().__name__
    def ToString(self) -> str:
        return str(self.GetType())
    def AsRef[_T](self, typen:Typen[_T]) -> _T:
        return dynamic_cast(self, typen)
    def AsValue[_T](self, typen:Typen[_T]) -> _T:
        '''
        warning: this will be a real transform, it is not a reference to the object
        '''
        return static_cast[_T](self, typen)
    def Fetch[_T](self, value:_T) -> _T:
        return value
    def Is[_T](self, typen:Typen[_T]) -> bool:
        return isinstance(self, typen)
    def IfIam[_T](self, typen:Typen[_T], action:Action[_T]) -> Self:
        if self.Is(typen):
            action(self)
        return self
    def AsIam[_T](self, typen:Typen[_T], action:Action[_T]) -> Self:
        action(self.AsRef(typen))
        return self
    def __enter__(self) -> Self:
        return self
    def __exit__(
        self,
        exc_type:   Optional[type],
        exc_val:    Optional[BaseException],
        exc_tb:     Optional[TracebackType]
        ) -> bool:
        if exc_val is None:
            return True
        else:
            return False
class base_value_reference[_T](type_class):
    def __init__(self, ref_value:_T):
        super().__init__()
        self._reinit_ref_value(ref_value)
    def _reinit_ref_value(self, value):
        if isinstance(value, base_value_reference):
            value = value._ref_value
        self._ref_value = value
        self.__real_type = type(value)
    def __getattr__(self, name):
        return self._ref_value.__getattribute__(name)
    def __setattr__(self, name, value):
        self._ref_value.__setattr__(name, value)
    def _clear_ref_value(self):
        self._ref_value = None
    @override
    def GetRealType(self):
        return self.__real_type
    @override
    def SymbolName(self) -> str:
        if self._ref_value is None:
            return f"{self.GetRealType()}<None, generate_on\n{self.generate_trackback}\n>"
        return f"{self.GetRealType().__name__}<generate on\n{self.generate_trackback}\n>&"
    @override
    def ToString(self) -> str:
        if self._ref_value is None:
            return "None"
        return str(self._ref_value)

    def __repr__(self):
        if self._ref_value is None:
            return "None"
        return repr(self._ref_value)
    def __str__(self):
        if self._ref_value is None:
            return "None"
        return str(self._ref_value)
class left_value_reference[_T](base_value_reference):
    def __init__(self, ref_value:_T):
        super().__init__(ref_value)
    @property
    def ref_value(self) -> _T:
        return self._ref_value
    @ref_value.setter
    def ref_value(self, value) -> _T:
        if value is None or isinstance(value, self.GetRealType()):
            self._ref_value = value
        elif self.GetRealType() == type(None):
            self._reinit_ref_value(value)
        else:
            raise TypeError(f"Cannot assign {type(value)} to {self.GetRealType()}")
        return value
    def __bool__(self):
        return self._ref_value is not None
    def is_empty(self):
        return self._ref_value is None
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
    def __init__(self):
        super().__init__()
    def Share[_T](self, out_value:left_value_reference[_T]) -> Self:
        if out_value is None:
            raise ValueError("out_value cannot be None")
        if isinstance(self, out_value.GetRealType()):
            out_value.ref_value = self
        else:
            out_value.ref_value = out_value.GetRealType()(self)
        return self

def UnwrapperInstance2Ref[_T](instance:Union[
    _T,
    base_value_reference
    ]) -> _T:
    if isinstance(instance, base_value_reference):
        return instance.ref_value
    else:
        return instance

class invoke_callable(any_class):
    def __init__(self):
        super().__init__()
    def __call__(self, *args, **kwargs):
        if "invoke" in dir(self):
            return self.invoke(*args, **kwargs)
        else:
            raise NotImplementedError(f"self<{self.SymbolName()}> invoke not implemented")
class null_package[_T](left_value_reference[_T]):
    @override
    def __init__(self, ref_value:_T):
        super().__init__(ref_value)

    def Try[_T](
        self,
        typen:Typen[_T],
        call:Action[_T]
        ) -> Self:
        if self.GetRealType() == typen:
            call(self.ref_value)
        return self
class closures[_T](left_value_reference[_T], invoke_callable):
    @override
    def __init__(self, ref_value:_T, callback:Action[_T]):
        super().__init__(ref_value)
        self.callback = callback
    def invoke(self):
        if self.callback is None:
            return
        self.callback(self.ref_value)
        self.callback = None
class release_closures[_T](closures[_T]):
    @override
    def __init__(self, ref_value:_T, callback:Action[_T]):
        super().__init__(ref_value)
        self.callback = callback
    def __del__(self):
        self.invoke()

# LightDiagram::ld::instance<_Ty>
class restructor_instance[_Ty](left_value_reference[_Ty]):
    def __init__(
        self,
        target:             _Ty,
        *,
        constructor_func:   Optional[Callable[[_Ty], None]] = None,
        destructor_func:    Optional[Callable[[_Ty], None]] = None
        ):
        super().__init__(target)
        if constructor_func:
            constructor_func(self.ref_value)
        self.destructor_func = destructor_func
    def __del__(self):
        if self.destructor_func:
            self.destructor_func(self.ref_value)

    def get_ref(self):
        return self.ref_value
    def is_empty(self):
        return self.ref_value is None

class iter_builder[_T](any_class):
    def __init__(self,  pr:Callable[[], bool], returner:Callable[[], _T]):
        self.pr = pr
        self.returner = returner
    def __iter__(self):
        return self
    def __next__(self):
        if self.pr():
            return self.returner()
        raise StopIteration
class iter_callable_range(Callable[[], bool], any_class):
    def __init__(
        self,
        start:  Union[int, left_value_reference[int]],
        stop:   Union[int, left_value_reference[int]],
        step=1
        ):
        self.__start = start
        self.__stop = stop
        self.__start_is_real_value = isinstance(start, int)
        self.__stop_is_real_value = isinstance(stop, int)
        self.step = step
    @property
    def start(self):
        return self.__start if self.__start_is_real_value else self.__start.ref_value
    @start.setter
    def start(self, value:int):
        if self.__start_is_real_value:
            self.__start = value
        else:
            self.__start.ref_value = value
    @property
    def stop(self):
        return self.__stop if self.__stop_is_real_value else self.__stop.ref_value
    @stop.setter
    def stop(self, value:int):
        if self.__stop_is_real_value:
            self.__stop = value
        else:
            self.__stop.ref_value = value
    def __call__(self):
        start = self.start
        stop = self.stop
        result = start<stop
        self.start = start + self.step
        return result

# using as c#: func(out var obj)
_out_static_value_reference = []
class out_value_reference[_T](left_value_reference[_T]):
    '''
    用于作为输出参数，使用时需要使用out_value_reader来读取输出值,
    顺序压参数入栈
    '''
    def __init__(self, value:Optional[_T] = None):
        super().__init__(value)
    def __del__(self):
        _out_static_value_reference.append(self.ref_value)
class out_value_reader[_T](right_value_refenence[_T]):
    '''
    用于读取对应out_value_reference的输出值,
    顺序弹参数出栈
    '''
    def __init__(self):
        if len(_out_static_value_reference) != 0:
            super().__init__(_out_static_value_reference.pop())
        else:
            super().__init__(None)

# using as c#: event
class ActionEvent[_Call:Callable](invoke_callable):
    def __init__(self, actions:Sequence[_Call]):
        super().__init__()
        self.__actions:     List[Callable]  = [action for action in actions]
        self.call_indexs:   List[int]       = [i for i in range(len(actions))]
        self.last_result:   List[Any]       = []
    def call_func_without_call_index_control(self, index:int, *args, **kwargs) -> Union[Any, Exception]:
        try:
            return self.__actions[index](*args, **kwargs)
        except Exception as ex:
            return ex
    def call_func(self, index:int, *args, **kwargs) -> Union[Any, Exception]:
        return self.call_func_without_call_index_control(self.call_indexs[index], *args, **kwargs)
    def _inject_invoke(self, *args, **kwargs):
        result:List[Any] = []
        for index in range(self.call_max_count):
            result.append(self.call_func(index, *args, **kwargs))
        return result
    def invoke(self, *args, **kwargs) -> Union[Self, bool]:
        self.last_result = self._inject_invoke(*args, **kwargs)
        return self
    def init_call_index(self):
        self.call_indexs = [i for i in range(len(self.__actions))]
    def add_action(self, action:_Call):
        self.__actions.append(action)
        self.call_indexs.append(len(self.__actions)-1)
        return self
    def add_actions(self, actions:Sequence[_Call]):
        for action in actions:
            self.add_action(action)
        return self
    def _internal_remove_action(self, action:_Call):
        if action in self.__actions:
            index = self.__actions.index(action)
            self.__actions.remove(action)
            self.call_indexs.remove(index)
            for i in range(len(self.call_indexs)):
                if self.call_indexs[i] > index:
                    self.call_indexs[i] -= 1
            return True
        return False
    def remove_action(self, action:_Call):
        while self._internal_remove_action(action):
            pass
        return self

    def is_valid(self):
        return not any_if(self.last_result, lambda x: isinstance(x, Exception))
    def __bool__(self):
        return self.is_valid()
    @property
    def call_max_count(self):
        return len(self.call_indexs)
    @property
    def action_count(self):
        return len(self.__actions)

# region instance

# threads
InternalGlobalLocker = threading.Lock()
class lock_guard(any_class):
    def __init__(
        self,
        lock:Optional[Union[threading.RLock, threading.Lock]] = None
        ):
        if lock is None:
            lock = InternalGlobalLocker
        self.__locker = lock
        self.__locker.acquire()
    def __del__(self):
        self.__locker.release()
class global_lock_guard(lock_guard):
    def __init__(self):
        super().__init__(None)
class thread_instance(threading.Thread, any_class):
    def __init__(
        self,
        call:           Action[None],
        *,
        is_del_join:    bool = True
        ):
        super().__init__(target=call)
        self.is_del_join = is_del_join
        self.start()
    def __del__(self):
        if self.is_del_join:
            self.join()
class atomic[_T](any_class):
    def __init__(
        self,
        value:  Optional[_T] = None,
        locker: Optional[threading.Lock] = None,
        ):
        self.__value:   _T = value
        self.__is_in_with: bool = False
        if locker is None:
            locker = InternalGlobalLocker
        self.locker:    threading.Lock = locker
    def fetch_add(self, value):
        with lock_guard(self.locker):
            self.__value += value
        return self.__value
    def fetch_sub(self, value):
        with lock_guard(self.locker):
            self.__value -= value
        return self.__value
    def load(self) -> _T:
        with lock_guard(self.locker):
            return self.__value
    def store(self, value: _T):
        with lock_guard(self.locker):
            self.__value = value
    def __add__(self, value):
        return self.fetch_add(value)
    def __sub__(self, value):
        return self.fetch_sub(value)
    def __iadd__(self, value):
        self.fetch_add(value)
        return self
    def __isub__(self, value):
        self.fetch_sub(value)
        return self
    @override
    def __enter__(self) -> Self:
        self.__is_in_with = True
        self.locker.acquire()
        return self
    @override
    def __exit__(
        self,
        exc_type:   Optional[type],
        exc_val:    Optional[BaseException],
        exc_tb:     Optional[TracebackType]
        ) -> Optional[bool]:
        self.__is_in_with = False
        self.locker.release()
        if exc_type is None:
            return True
        else:
            raise exc_val
    @property
    def value(self) -> _T:
        if self.__is_in_with:
            return self.__value
        raise NotImplementedError("This method can only be called within a with statement")
    @value.setter
    def value(self, value:_T) -> _T:
        if self.__is_in_with:
            self.__value = value
        raise NotImplementedError("This method can only be called within a with statement")

# region end

def create_py_file(path:str):
    with open(path, "w") as f:
        f.write("# -*- coding: utf-8 -*-\n")
        f.write("from lekit.lazy import *\n")
        f.write("\n")
        f.write("def run():\n")
        f.write("\tpass\n")
        f.write("\n")
        f.write(f"if __name__ == \"__main__\":\n")
        f.write("\trun()\n")

def WrapperConfig2Instance[_TargetType](
    typen_or_generater:             Union[type, Callable[[Any], _TargetType]],
    datahead_of_config_or_instance: Optional[Union[
            Dict[str, Any], # kwargs
            Sequence[Any],  # args
            _TargetType,            # typen or datahead
        ]],
    *args,
    datahead_typen_:                 Optional[type] = None,
    **kwargs
    ) -> _TargetType:
    if typen_or_generater is None:
        raise ValueError("typen cannt be none")
    if datahead_of_config_or_instance is None:
        return typen_or_generater(*args, **kwargs)
    elif datahead_typen_ is not None and isinstance(datahead_of_config_or_instance, datahead_typen_):
        return datahead_of_config_or_instance
    elif isinstance(typen_or_generater, type) and isinstance(datahead_of_config_or_instance, typen_or_generater):
        return datahead_of_config_or_instance
    elif isinstance(datahead_of_config_or_instance, dict):
        return typen_or_generater(*args, **datahead_of_config_or_instance, **kwargs)
    elif isinstance(datahead_of_config_or_instance, Sequence):
        return typen_or_generater(*datahead_of_config_or_instance, *args, **kwargs)
    else:
        return typen_or_generater(datahead_of_config_or_instance, *args, **kwargs)

def remove_none_value[_T:Union[
        dict, tuple, list
    ]](data:_T) -> _T:
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if v is not None}
    elif isinstance(data, (tuple, list)):
        return type(data)(v for v in data if v is not None)
    else:
        raise ValueError(f"remove_none_value not support this type<{type(data)}>")
def to_list[_DataTy, _T:Sequence[_DataTy]](data:_T) -> List[_DataTy]:
    return data if isinstance(data, list) else list(data)
def to_tuple[_DataTy, _T:Sequence[_DataTy]](data:_T) -> Tuple[_DataTy, ...]:
    return data if isinstance(data, tuple) else tuple(data)

if __name__ == "__main__":
    ref = left_value_reference[int](5.5)
    print(ref.ToString())
    print(ref.GetType())
    print(ref.SymbolName())
    print(ref.ref_value)
    print(ref)



