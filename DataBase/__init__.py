from lekit.Internal import *
from abc import *

try:
    import lekit.Str.Core as _
except ImportError:
    InternalImportingThrow("DataBase", ["Str"])
    

# Import Core.py
try:
    from lekit.DataBase.Core import *
except ImportError as ex:
    ImportingThrow(ex, "DataBase Core", [])


# Import Core.py
try:
    import sqlite3 as _
    from lekit.DataBase.light_sqlite import *
except ImportError as ex:
    ImportingThrow(ex, "light_sqlite", ["sqlite3"])