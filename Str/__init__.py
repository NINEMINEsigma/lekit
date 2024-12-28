from lekit.Internal import *

try:
    from lekit.Lang.Reflection import light_reflection as _
except ImportError:
    InternalImportingThrow("String", ["Lang"])
    
# Import RE.py
try:
    import re as _
    from lekit.Str.RE import *
except ImportError:
    InternalImportingThrow("Regular-Expression", ["re"])

# Import Core.py
try:
    from pathlib import Path as _
    import jieba as _
    from lekit.Str.Core import *
except ImportError as ex:
    ImportingThrow(ex, "String-Core", ["pathlib"])