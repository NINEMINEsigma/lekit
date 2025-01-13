from lekit.Internal import *

try:
    from lekit.Str.Core import UnWrapper as _
    from lekit.File.Core import tool_file as _
    from lekit.MathEx.Transform import Rect as _
except ImportError:
    InternalImportingThrow("Web", ["String", "File", "Math"])

# Import Core.py
try:
    from http import server as _
    from lekit.Web.Core import *
except ImportError as ex:
    ImportingThrow(ex, "Web-Core", ["http"])

# Import BeautifulSoup.py
try:
    import bs4 as _
    from lekit.Web.BeautifulSoup import *
except ImportError as ex:
    ImportingThrow(ex, "BeautifulSoup", ["bs4"])

# Import Requests.py
try:
    import requests as _
    import urllib3 as _
    from lekit.Web.Requests import *
except ImportError as ex:
    ImportingThrow(ex, "Requests", ["requests", "urllib3"])

# Import Selunit.py
try:
    import selenium as _
    from lekit.Web.Selunit import *
except ImportError as ex:
    ImportingThrow(ex, "Selunit", ["selenium"])