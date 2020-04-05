import numpy as np
import multiprocessing

import astropy.io.fits as pyfits
import orb.utils.io

import logging
import warnings

logging.getLogger().setLevel('INFO')

from . import config
from . import gluon
from . import quark

class Core(object):

    def __init__(self, cubepath):


        if '.fits' in cubepath:
            self.data = pyfits.open(cubepath)[0].data.astype(np.float32)
        else:
            self.data = orb.utils.io.read_hdf5(cubepath, dtype=np.complex64)
        self.data[np.isnan(self.data)] = 0.
        self.data /= np.nanmax(self.data) * 0.5
        self.data[self.data > 1] = 1.
        self.data[self.data < -1] = -1.
        self.data *= 1000
        logging.info('data shape: {}'.format(self.data.shape))
        self.mgr = multiprocessing.Manager()
        self.out_bufferL = multiprocessing.RawArray(
            'f', np.zeros(config.BUFFERSIZE, dtype=np.float32))
        self.out_bufferR = multiprocessing.RawArray(
            'f', np.zeros(config.BUFFERSIZE, dtype=np.float32))
        
        self.out_i = multiprocessing.RawValue('i', 0)
        self.outlock = multiprocessing.Lock()

        self.notes = multiprocessing.Array('d', np.zeros(256, dtype=float))
    
        self.p = multiprocessing.RawArray('d', np.zeros(4, dtype=float))
        self.p[0] = 0.05 # attack
        self.p[1] = 2 # release
        self.p[2] = self.data.shape[0]/2 # posx
        self.p[3] = self.data.shape[1]/2 # posy

        self.midi_server_process = multiprocessing.Process(
            name='midiserver',
            target=quark.MidiServer, 
            args=())

        self.midi_server_process.start()
        
        self.audio_player_process = multiprocessing.Process(
            name='audioplayer',
            target=gluon.AudioPlayer, 
            args=(self.out_bufferL, self.out_bufferR, self.out_i, self.outlock))

        self.audio_player_process.start()
                
        self.midi_player_process = multiprocessing.Process(
            name='midiplayer',
            target=quark.MidiPlayer, 
            args=(self.out_bufferL, self.out_bufferR, self.out_i,
                  self.notes, self.p, self.outlock, self.data))

        self.midi_player_process.start()

        self.midi_server_process.join()
        self.audio_player_process.join()
        self.midi_player_process.join()
        
