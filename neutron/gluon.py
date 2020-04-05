import numpy as np
import multiprocessing
import time

import sounddevice as sd

import warnings
import logging

from . import config
from . import ccore

        
class AudioPlayer(object):

    def __init__(self, out_bufferL, out_bufferR, out_i, outlock):

        self.out_bufferL = out_bufferL
        self.out_bufferR = out_bufferR
        
        self.out_i = out_i
        self.zeros = np.zeros(config.BUFFERSIZE, dtype=np.float32)
        self.outlock = outlock
        sd.default.device = config.DEVICE
        sd.default.samplerate = config.SAMPLERATE
        sd.default.latency = 'low'
        
        logging.info('>> AUDIO OUTPUTS:\n{}'.format(sd.query_devices()))
        logging.info('>> AUDIO OUTPUT: {}'.format(config.DEVICE))
        
        self.last_looptime = 0
        #self.sine = ccore.SineWave(32)
        with sd.OutputStream(
                dtype=config.CAST, channels=config.NCHANNELS,
                callback=self.callback, blocksize=config.BUFFERSIZE) as self.stream:
        
            while True:
                time.sleep(config.SLEEPTIME)
                
                
    def callback(self, outdata, frames, timing, status):
        stime = time.time()
        if status:
            warnings.warn('callback status: {} \n > (callback loop time: {})'.format(status, self.last_looptime))

        self.outlock.acquire()    
        try:
            self.out_i.value += 1
            outdata[:] = ccore.ndarray2buffer(np.array((self.out_bufferL, self.out_bufferR), dtype=np.float32).T)
            self.out_bufferL[:] *= self.zeros
            self.out_bufferR[:] *= self.zeros
            # for testing pure sine
            #outdata[:] = self.sine.get_buffer(config.BUFFERSIZE)
        except Exception as err:
            warnings.warn('callback error {}'.format(err))
            raise
        finally:
            self.outlock.release()

                
        self.last_looptime = time.time() - stime

    def __del__(self):
        try:
            self.stream.close()
        except: pass 
                        



        

