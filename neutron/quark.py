import numpy as np
import multiprocessing
import multiprocessing.connection
        
import time
import datetime
import logging
import warnings
import soundfile as sf

import sounddevice as sd
import soundfile
import mido

import astropy.io.fits as pyfits
import orb.utils.io

logging.getLogger().setLevel('INFO')

from . import config
from . import ccore
from . import utils

class Core(object):

    def __init__(self, cubepath):

        self.manager = multiprocessing.Manager()
        self.sampler = self.manager.dict()
        
        if '.fits' in cubepath:
            self.data = pyfits.open(cubepath)[0].data.astype(np.float32)
        else:
            self.data = orb.utils.io.read_hdf5(cubepath, dtype=np.complex64)
        self.data[np.isnan(self.data)] = 0.
        self.data /= np.nanmax(self.data) * 0.5
        self.data[self.data > 1] = 1.
        self.data[self.data < -1] = -1.
        self.data *= config.DATA_MASTER
        logging.info('data shape: {}'.format(self.data.shape))
        self.out_bufferL = multiprocessing.RawArray(
            'f', np.zeros(config.BUFFERSIZE, dtype=np.float32))
        self.out_bufferR = multiprocessing.RawArray(
            'f', np.zeros(config.BUFFERSIZE, dtype=np.float32))
        
        self.out_i = multiprocessing.RawValue('i', 0)
        self.outlock = multiprocessing.Lock()

        
        self.notes = multiprocessing.Array('d', np.zeros(256, dtype=float))
    
        self.p = multiprocessing.RawArray('d', np.zeros(4, dtype=float))
        self.p[0] = config.ATTACK # attack
        self.p[1] = config.RELEASE # release
        self.p[2] = self.data.shape[0]/2 # posx
        self.p[3] = self.data.shape[1]/2 # posy

        self.audio_player_process = multiprocessing.Process(
            name='audioplayer',
            target=AudioPlayer, 
            args=(self.out_bufferL, self.out_bufferR, self.out_i, self.outlock))

        self.audio_player_process.start()
                
        self.midi_player_process = multiprocessing.Process(
            name='midiplayer',
            target=MidiPlayer, 
            args=(self.out_bufferL, self.out_bufferR, self.out_i,
                  self.notes, self.p, self.outlock, self.data, self.sampler))

        self.midi_player_process.start()


    def load_sample(self, note, path):
        try:
            with open(path, 'rb') as f:
                data, samplerate = soundfile.read(f)
                if len(data.shape) == 1:
                    data = np.array([data, data]).T
                if len(data.shape) != 2:
                    raise Exception('bad sample format: {}'.format(data.shape))
                if data.shape[1] != 2:
                    raise Exception('bad sample format: {}'.format(data.shape))
        except Exception as e:
            logging.info('error reading sample', path, e)
        else:
            self.sampler[str(note)] = data
        
    def __del__(self):
        try:
            self.manager.close()
            self.audio_player_process.terminate()
            self.audio_player_process.join()
            
            self.midi_player_process.terminate()
            self.midi_player_process.join()

        except Exception:
            pass
                            
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
        self.lastoutbuffer = None
        if config.AUTORECORD:
            self.soundfile = sf.SoundFile(
                '{}.wav'.format(datetime.datetime.timestamp(datetime.datetime.now())), 'w',
                samplerate=config.SAMPLERATE, channels=config.NCHANNELS)
        
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
            lastoutbuffer = ccore.ndarray2buffer(np.array((self.out_bufferL, self.out_bufferR), dtype=np.float32).T)
            outdata[:] = lastoutbuffer
            if config.AUTORECORD:
                self.soundfile.write(lastoutbuffer)
            
            self.out_bufferL[:] *= self.zeros
            self.out_bufferR[:] *= self.zeros
        except Exception as err:
            warnings.warn('callback error {}'.format(err))
            raise
        finally:
            self.outlock.release()

                
        self.last_looptime = time.time() - stime

    def __del__(self):
        try:
            self.stream.close()
            if config.AUTORECORD:
                self.soundfile.close()
        except: pass 
                        


class MidiPlayer(object):

    def __init__(self, out_bufferL, out_bufferR, out_i, notes, p, outlock, data, sampler):


        self.sampler = sampler
        
        loopcount = 0
        longloop = 100
        timing = 0
        view = ccore.data2view(np.ascontiguousarray(data.astype(np.complex64)))

        self.listener = multiprocessing.connection.Listener(
            ('localhost', config.PORT), authkey=b'neutron')
        self.connection = self.listener.accept()
        #print('connection accepted from', self.listener.last_accepted)
        
        sounds = list()

        release = p[1]
        attack = p[0]
        posx = int(p[2])
        posy = int(p[3])
        
        while True:
            loopcount += 1
            if loopcount > longloop:
                release = p[1]
                attack = p[0]
                for isound in sounds:
                    if not isound[0].is_alive():
                        isound[0].join()
                        del(isound)

            time.sleep(config.SLEEPTIME)

            msgs = [utils.Message(*self.connection.recv())]
            
            # note off are considered first
            msgs = sorted(msgs, key= lambda elem: elem.category == 'note_on')

            for msg in msgs:
                logging.debug('MIDI msg: {}'.format(str(msg)))

                if msg.category == 'note_on':
                    timing = time.time()

                    notes[int(msg.note)] = timing
                                        
                    # sounds.append(
                    #     (multiprocessing.Process(
                    #         name='sound', 
                    #         target=ccore.sound,
                    #         args=(out_bufferL, out_bufferR, out_i, notes,
                    #               msg.note, msg.velocity, msg.channel, outlock,
                    #               timing, attack, release,
                    #               config.BUFFERSIZE, config.MASTER, config.SLEEPTIME,
                    #               view, 'sine', config.DIRTY, config.DATATUNE, posx, posy,
                    #               config.DATA_UPDATE_TIME)),
                    #      msg.note))

                    try:
                        sample = self.sampler[str(msg.note)]
                    except KeyError:
                        print('no sound registered as', msg.note)
                    else:
                        sounds.append(
                            (multiprocessing.Process(
                                name='play_sample', 
                                target=play_sample,
                                args=(out_bufferL, out_bufferR, out_i, notes, msg.note, outlock,
                                      timing, attack, release, 
                                      sample,
                                      config.BUFFERSIZE)),
                            msg.note))


                        sounds[-1][0].start()


                else:
                    notes[int(msg.note)] = 0

    def __del__(self):
        try:
            self.connection.close()
            self.listener.close()
        except Exception as e:
            warnings.warn('exception during MidiPlayer closing: {}'.format(e))


def play_sample(out_bufferL, out_bufferR, out_i, notes, note, outlock, timing, attack, release, data, buffersize):
    i = 0
    lastbuffer= 0
    stop = False
    att_stime = 0
    rel_stime = 0
    
    while not stop:
        now = time.time()        

        if att_stime == 0:
            att_stime = now
            
        svr = 1
        evr = 1
        if rel_stime > 0:
            svr, evr = ccore.release_values(now, rel_stime, release, buffersize)
        sva, eva = ccore.attack_values(now, att_stime, attack, buffersize)
        sv = sva * svr
        ev = eva * evr
        
        outlock.acquire()
        try:
            if out_i.value > lastbuffer:
                lastbuffer = out_i.value
                env = np.arange(buffersize, dtype=float) / buffersize * (ev - sv) + sv
                out_bufferL[:] += data[i:i+buffersize, 0] * env
                out_bufferR[:] += data[i:i+buffersize, 1] * env
                i += buffersize

        except Exception as e:
            print('error playing sample: {}'.format(e))

        finally:
            outlock.release()

        # note off, starting release
        if notes[int(note)] != timing: 

            if rel_stime == 0:
                rel_stime = now

            if now - rel_stime >= release:
                stop = True


        if i >= data.shape[0] - buffersize:
            stop = True

    
