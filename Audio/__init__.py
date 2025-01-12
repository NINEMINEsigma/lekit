from lekit.Internal import *

try:
    from lekit.File.Core import tool_file as _
except ImportError:
    InternalImportingThrow("Audio", ["File"])
    
# Import Core.py
try:
    import sounddevice as _
    import soundfile as _
    from lekit.Audio.Core import *
except ImportError as ex:
    ImportingThrow(ex, "Audio Core", ["sounddevice", "soundfile"])

# Import Microphone.py
try:
    import numpy as _
    import wave as _
    import keyboard as _
    from lekit.Audio.Microphone import *
except ImportError as ex:
    ImportingThrow(ex, "Microphone", ["numpy", "wave", "keyboard"])
    
# Import Vocal.py
try:
    import speech_recognition as _
    from lekit.Audio.Vocal import *
except ImportError as ex:
    ImportingThrow(ex, "Vocal", ["SpeechRecognition"])