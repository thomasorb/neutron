import numpy as np

SAMPLERATE = 44100
NCHANNELS = 2
BUFFERSIZE = 2000
BYTEDEPTH = 32
SLEEPTIME = BUFFERSIZE / SAMPLERATE / 1000.
MASTER = 0.1
DATA_MASTER = 10000
DIRTY = 700
DATA_UPDATE_TIME = 0.2

DATATUNE = 12
BASENOTE = 102
A_MIDIKEY = 45
INTEGSIZE = 20

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
#MIDI_IN = 'Midi Through:Midi Through Port-0'
MIDI_IN = 'CASIO USB-MIDI:CASIO USB-MIDI MIDI 1'

#DEVICE = 'HDA Intel PCH: ALC293 Analog (hw:0,0)'
DEVICE = 'Steinberg UR22mkII: USB Audio'


ATTACK = 0.05
RELEASE = 0.5

CHANNEL_NB = 5
TEMPO = 120
INSTRUMENT = 0
VELOCITY = 127
MEASURE_BEATS = 4
MEASURE_NOTE = 4
OCTAVE = 5
MODE = 0
