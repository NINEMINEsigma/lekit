from lekit.Internal import *

try:
    from lekit import MathEx as _
except ImportError:
    InternalImportingThrow("Machine-Learning", "Math")

try:
    import sklearn as _
except ImportError as ex:
    ImportingThrow(ex, "Core", ["scikit-learn"])

try:
    from keras import api as _
    from lekit.ML.Keras import *
except ImportError as ex:
    ImportingThrow(ex, "Keras", ["keras(or tensorflow)"])
