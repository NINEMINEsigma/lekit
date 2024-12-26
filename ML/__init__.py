from lekit.Internal import *

try:
    import sklearn as _
except ImportError as ex:
    ImportingThrow(ex, "Core", ["scikit-learn"])


