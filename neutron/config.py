import numpy as np

SAMPLERATE = 48000
NCHANNELS = 2
BUFFERSIZE = 500
BYTEDEPTH = 32
SLEEPTIME = BUFFERSIZE / SAMPLERATE / 1000.
MASTER = 0.1
PRELOADTIME = 5

if BYTEDEPTH == 32:
    CAST = np.int32
elif BYTEDEPTH == 16:
    CAST = np.int16
elif BYTEDEPTH == 8:
    CAST = np.int8
else:
    raise Exception('bad DEPTH')

#USB_IN = 'VMPK Output:VMPK Output'
#USB_IN = 'CASIO USB-MIDI:CASIO USB-MIDI MIDI 1 24:0'
# pulseaudio -k && sudo alsa force-reload
USB_IN = 'Midi Through:Midi Through Port-0'
#USB_IN = 'SuperCollider:out0 129:16'

# DEVICE = 'HDA Intel PCH: ALC293 Analog (hw:0,0)'
DEVICE = 'Steinberg UR22mkII: USB Audio (hw:1,0)'
