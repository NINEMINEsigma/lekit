from lekit.Internal import *

# Import Tkinter.py
try:
    import tkinter as _
    from lekit.UI.Tkinter import *
except ImportError as ex:
    ImportingThrow(ex, "TkinterUI", ["tkinter"])

