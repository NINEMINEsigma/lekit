from lekit.Internal import *

try:
    from lekit.File.Core import tool_file as _
except ImportError:
    InternalImportingThrow("Audio", ["File"])
    
# Import Core.py
try:
    import sounddevice as _
    import soundfile as _
except ImportError as ex:
    ImportingThrow(ex, "Audio Core", ["sounddevice", "soundfile"])
from lekit.Audio.Core import *

# Import Microphone.py
try:
    import numpy as _
    import wave as _
    import keyboard as _
except ImportError as ex:
    ImportingThrow(ex, "Microphone", ["numpy", "wave", "keyboard"])
from lekit.Audio.Microphone import *
    
# Import Vocal.py
try:
    import speech_recognition as _
except ImportError as ex:
    ImportingThrow(ex, "Vocal", ["speech_recognition"])
from lekit.Audio.Vocal import *