from lekit.Internal import *

try:
    from lekit.Str.Core import UnWrapper as _
except ImportError:
    InternalImportingThrow("Web", ["String"])

# Import Core.py
try:
    from http import server as _
except ImportError as ex:
    ImportingThrow(ex, "Web-Core", ["http"])
from lekit.Web.Core import *

# Import BeautifulSoup.py
try:
    import bs4 as _
except ImportError as ex:
    ImportingThrow(ex, "BeautifulSoup", ["bs4"])
from lekit.Web.BeautifulSoup import *

# Import Requests.py
try:
    import requests as _
    import urllib3 as _
except ImportError as ex:
    ImportingThrow(ex, "Requests", ["requests", "urllib3"])
from lekit.Web.Requests import *

print("xx")