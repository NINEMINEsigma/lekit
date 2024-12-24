from lekit.Internal import *

try:
    from lekit.Lang.Reflection import light_reflection as _
except ImportError:
    InternalImportingThrow("String", ["Lang"])
    
# Import RE.py
try:
    import re as _
except ImportError:
    InternalImportingThrow("Regular-Expression", ["re"])
from lekit.Str.RE import *

# Import Core.py
try:
    from pathlib import Path as _
except ImportError as ex:
    ImportingThrow(ex, "String-Core", ["pathlib"])
from lekit.Str.Core import *