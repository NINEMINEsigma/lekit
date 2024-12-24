from lekit.Internal import *
from abc import *

try:
    import lekit.Str.Core as _
except ImportError:
    InternalImportingThrow("DataBase", ["Str"])
    

# Import Core.py
try:
    pass
except ImportError as ex:
    ImportingThrow(ex, "DataBase Core", [])
from lekit.DataBase.Core import *


# Import Core.py
try:
    import sqlite3 as _
except ImportError as ex:
    ImportingThrow(ex, "light_sqlite", ["sqlite3"])
from lekit.DataBase.light_sqlite import *