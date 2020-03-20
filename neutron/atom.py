import numpy as np
import scipy.fft
import multiprocessing
import time

import sounddevice as sd
import mido

import warnings
import logging
logging.getLogger().setLevel('INFO')

from . import config
from . import utils
from . import ccore

class Core(object):

    def __init__(self):

        self.mgr = multiprocessing.Manager()
        self.out_bufferL = multiprocessing.RawArray(
            'f', np.zeros(config.BUFFERSIZE, dtype=np.float32))
        self.out_bufferR = multiprocessing.RawArray(
            'f', np.zeros(config.BUFFERSIZE, dtype=np.float32))
        
        self.out_i = multiprocessing.RawValue('i', 0)
        self.outlock = multiprocessing.Lock()

        self.notes = multiprocessing.Array('d', np.zeros(256, dtype=float))
    
        self.p = multiprocessing.RawArray('d', np.zeros(2, dtype=float))
        self.p[0] = 0.3 # attack
        self.p[1] = 1 # release
        
        self.player_process = multiprocessing.Process(
            name='player',
            target=Player, 
            args=(self.out_bufferL, self.out_bufferR, self.out_i, self.outlock))

        self.player_process.start()
                
        self.midi_process = multiprocessing.Process(
            name='midiplayer',
            target=MidiPlayer, 
            args=(self.out_bufferL, self.out_bufferR, self.out_i, self.notes, self.p, self.outlock))

        self.midi_process.start()

        self.player_process.join()
        self.midi_process.join()
        

class Player(object):

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
                        

class MidiPlayer(object):

    def __init__(self, out_bufferL, out_bufferR, out_i, notes, p, outlock):

        loopcount = 0
        longloop = 100
        timing = 0
        
        logging.info('>> MIDI INPUTS:\n   {}'.format('\n   '.join(mido.get_input_names())))
        logging.info('>> MIDI INPUT: {}'.format(config.USB_IN))
        inport = mido.open_input(config.USB_IN)

        sounds = list()

        release = p[1]
        attack = p[0]

        while True:
            loopcount += 1
            if loopcount > longloop:
                release = p[1]
                attack = p[0]
                for isound in sounds:
                    if not isound[0].is_alive():
                        del isound

            time.sleep(config.SLEEPTIME)

            rawmsgs = [msg for msg in inport.iter_pending()]
            if len(rawmsgs) == 0: continue
            # integrate other messages sent right after the first ones
            time.sleep(config.SLEEPTIME)
            rawmsgs += [msg for msg in inport.iter_pending()] 

            msgs = list()
            msgs.append(rawmsgs[0])
            for msg in rawmsgs[1:]:
                if msg not in msgs:
                    msgs.append(msg)

            # note off are considered first
            msgs = sorted(msgs, key= lambda elem: elem.type == 'note_on')

            for msg in msgs:

                if msg.type == 'note_on':
                    timing = time.time()

                    notes[int(msg.note)] = timing

                    #logging.debug('MIDI msg: {}'.format(msg))

                    sounds.append(
                        (multiprocessing.Process(
                            name='sound', 
                            target=ccore.sound,
                            args=(out_bufferL, out_bufferR, out_i, notes, msg, outlock,
                                  timing, attack, release,
                                  config.BUFFERSIZE, config.MASTER, config.SLEEPTIME,
                                  config.PRELOADTIME)),
                         msg.note))

                    sounds[-1][0].start()


                else:
                    notes[int(msg.note)] = 0
