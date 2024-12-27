from lekit.Internal import *

try:
    import sklearn as _
except ImportError as ex:
    ImportingThrow(ex, "Core", ["scikit-learn"])

try:
    from keras import api as _
    from lekit.ML.Keras import *
except ImportError as ex:
    ImportingThrow(ex, "Keras", ["keras(or tensorflow)"])
