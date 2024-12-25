from lekit.Internal import *

try:
    import lekit.Str.Core as _
    import lekit.Lang as _
except ImportError:
    InternalImportingThrow("File", ["Str", "Lang"])

# Import Core.py
try:
    import json as _
    import shutil as _
    import pandas as _
    import os as _
    import sys as _
    import pickle as _
    from pathlib                                        import Path as _
    from pydub                                          import AudioSegment as _
    from PIL                                            import Image as _, ImageFile as _
    from docx                                           import Document as _
    from docx.document                                  import Document as _
    from lekit.File.Core import *
except ImportError as ex:
    InternalImportingThrow("File Core", ["json", "shutil", "pandas", "os", "sys", "pickle", "pathlib", "pydub", "PIL", "docx"], ex)