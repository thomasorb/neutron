import numpy as np

SAMPLERATE = 44100
NCHANNELS = 2
BUFFERSIZE = 600
BYTEDEPTH = 32
SLEEPTIME = BUFFERSIZE / SAMPLERATE / 1000.
MASTER = 2
DIRTY = 700
DATATUNE = 36

if BYTEDEPTH == 32:
    CAST = np.int32
elif BYTEDEPTH == 16:
    CAST = np.int16
elif BYTEDEPTH == 8:
    CAST = np.int8
else:
    raise Exception('bad DEPTH')

# pulseaudio -k && sudo alsa force-reload

MIDI_OUT = 'Midi Through:Midi Through Port-0'

#MIDI_IN = 'VMPK Output:VMPK Output'
MIDI_IN = 'Midi Through:Midi Through Port-0'
#MIDI_IN = 'CASIO USB-MIDI:CASIO USB-MIDI MIDI 1'

#DEVICE = 'HDA Intel PCH: ALC293 Analog (hw:0,0)'
DEVICE = 'Steinberg UR22mkII: USB Audio'
