from lekit.Internal import *

# Import BaseClass.py
try:
    from pydantic   import BaseModel as _
except ImportError as ex:
    ImportingThrow(ex, "Lang-Base", ["pydantic"])    
from lekit.Lang.BaseClass import *

# Import CppLike.py
try:
    pass
except ImportError as ex:
    ImportingThrow(ex, "CppLike", [])
from lekit.Lang.CppLike import *

# Import Reflection.py
try:
    import importlib as _
    import inspect as _
except ImportError as ex:
    ImportingThrow(ex, "Reflection", ["importlib", "inspect"])
from lekit.Lang.Reflection import *