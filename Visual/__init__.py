from lekit.Internal import *

try:
    from lekit.File.Core        import tool_file as _
except ImportError:
    InternalImportingThrow("Visual", ["File"])

# Import Core.py
try:
    import matplotlib.pyplot    as     _
    import seaborn              as     _
    import pandas               as     _
    import cv2 as _
    import numpy                as _
    from lekit.Visual.Core import *
except ImportError as ex:
    ImportingThrow(ex, "Visual-Core", ["matplotlib", "seaborn", "pandas", "opencv-python", "numpy"])

# Import OpenCV.py
try:
    import numpy            as     _
    from PIL                import ImageFile as _
    from lekit.Visual.OpenCV import *
except ImportError as ex:
    ImportingThrow(ex, "OpenCV", ["numpy", "Pillow"])

# Import WordCloud.py
try:
    from pyecharts.charts import WordCloud as _
    from pyecharts import options as _
    from lekit.Visual.WordCloud import *
except ImportError as ex:
    ImportingThrow(ex, "WordCloud", ["wordcloud"])
