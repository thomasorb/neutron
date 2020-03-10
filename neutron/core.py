import astropy.io.fits as pyfits
import numpy as np
import scipy.fftpack
#import pyaudio
import sounddevice as sd
import wave
import sys
import pylab as pl
import multiprocessing
import matplotlib.backend_bases
import matplotlib.widgets
import matplotlib.animation
import time
import neutron.utils
print(sd.query_devices())

SAMPLERATE = 44100
NCHANNELS = 2
BUFFERSIZE = 2000
MAX_NOTECHANNELS = 10
DEPTH = 32
if DEPTH == 32:
    CAST = np.int32
elif DEPTH == 16:
    CAST = np.int16
elif DEPTH == 8:
    CAST = np.int8
else:
    raise Exception('bad DEPTH')
    
sd.default.samplerate = SAMPLERATE
sd.default.latency = 'low'

#sd.default.device = 'HDA Intel PCH: ALC293 Analog (hw:0,0)'
sd.default.device = 'Steinberg UR22mkII: USB Audio (hw:1,0)'

#USB_IN = 'VMPK Output:VMPK Output 128:0'
USB_IN = 'CASIO USB-MIDI:CASIO USB-MIDI MIDI 1 24:0'
# pulseaudio -k && sudo alsa force-reload


import mido
print(mido.get_input_names())



class Machine(object):

    def __init__(self, cubepath, dfpath):

        self.data = pyfits.open(cubepath)[0].data.T
        self.shape = np.array(self.data.shape)
        
        
        self.mgr = multiprocessing.Manager()
        self.p = self.mgr.dict()
        self.p['harm_number'] = (1, 1, 20, int)
        self.p['harm_step'] = (2, 1, 15, int)
        self.p['center'] = ((self.shape / 2).astype(int), None, None, None)
        self.p['size'] = (10, 1, 30, int)
        self.p['worm_move_scale'] = (3, 1, 50, int)
        self.p['harm_scale'] = (0., -2., 2., float)
        self.p['norm_perc_scale'] = (4, 1, 7, float)
        self.p['shift'] = (600,100,1000, int)
        self.p['norm'] = (None, None, None, None)
        self.p['attack'] = (0.1, 0, 1, float)
        self.p['release'] = (0.5, 0, 5, float)
        for i in range(MAX_NOTECHANNELS):
            self.p[str(i)] = (None, None, None, None)
        for i in range(MAX_NOTECHANNELS):
            self.p[str(i) + 'out'] = (None, None, None, None)
        
        self.p['out'] = (None, None, None, None)
        

            
        
        self.player_process = multiprocessing.Process(
            name='player', 
            target=Player, 
            args=(self.data, self.p))


        self.player_process.start()
        
        
        self.viewer_process = multiprocessing.Process(
            name='viewer', 
            target=Viewer, 
            args=(dfpath, self.p))

        
        self.viewer_process.start()

        self.midi_process = multiprocessing.Process(
            name='midi',
            target=Midi, 
            args=(self.p, self.data))

        self.midi_process.start()

        self.viewer_process.join()
        self.player_process.join()        
        self.midi_process.join()
        
        
        
class Base(object):

    def getp(self, key):
        if self.p[key][1] is not None:
            return self.p[key][3](np.clip(self.p[key][0], self.p[key][1], self.p[key][2]))
        return self.p[key][0]

    def setp(self, key, value):
        vals = list(self.p[key])
        vals[0] = value
        self.p[key] = vals
        
class Worm(Base):

    def __init__(self, p):
        self.p = p
        self.old_center = self.getp('center')
        self.center = np.copy(self.old_center)
        self.move()
        
    def move(self):
        if np.any(self.getp('center') != self.old_center):
            self.center = np.copy(self.getp('center'))
            self.old_center = np.copy(self.center)
            
        self.center += np.random.randint(
            -self.getp('worm_move_scale'), self.getp('worm_move_scale')+1, size=self.center.size)

    def get(self):
        self.move()
        return np.copy(self.center)


class Worms(Base):

    def __init__(self, p):
        self.p = p
        self.update()
        
    def update(self):
        if not hasattr(self, 'worms'):
            self.worms = [Worm(self.p) for i in range(self.getp('harm_number') * NCHANNELS)]
            
        if len(self.worms) == self.getp('harm_number') * NCHANNELS:
            return

        while len(self.worms) > self.getp('harm_number') * NCHANNELS:
            self.worms.pop(0)

        while len(self.worms) < self.getp('harm_number') * NCHANNELS:
            self.worms.append(Worm(self.p))

    def get(self):
        self.update()
        return [iworm.get() for iworm in self.worms]
    
class Player(Base):

    def __init__(self, data, p, sample_width=2):

        print('starting player')
        self.data = data
        self.shape = np.array(self.data.shape)
        self.data_max = np.max(self.data)
        print('data loaded')
        #self.data_sub = self.data.flatten()
        #np.random.shuffle(self.data_sub)
        #self.data_sub = self.data_sub[:int(0.0002*self.data_sub.size)]
        #print(self.data_sub.shape)
        self.p = p
        
        self.old_norm_perc_scale = None
    
        with sd.OutputStream(
                dtype=CAST, channels=NCHANNELS,
                callback=self.callback, blocksize=BUFFERSIZE,
                latency='low'):
        
            while True:
                time.sleep(0.00001)
                
    def callback(self, outdata, frames, time, status):
        if status:
            print('callback status:', status)

        try:
            self.reset_norm()
        
            sample = np.zeros((BUFFERSIZE, NCHANNELS), dtype=float)
            #stime = time.time()
            for i in range(MAX_NOTECHANNELS):
                iout = str(i) + 'out'
                if self.getp(iout) is not None:
                    sample += self.getp(iout)
                    self.setp(iout, None)
            #print(time.time() - stime)
            if sample[0,0] != 0.:
                
                sample *= 2**(DEPTH-2) - 1
                sample = np.ascontiguousarray(sample.astype(CAST))
                self.setp('out', sample)
                #self.stream.write(sample)
            outdata[:] = sample
    
        except Exception as err:
            print('callback error', err)
        
        
    def reset_norm(self):
        norm_perc_scale = float(self.getp('norm_perc_scale'))
        if norm_perc_scale != self.old_norm_perc_scale:
            norm = self.data_max * 1.-10**(1-norm_perc_scale)
            print('norm:', norm)
            self.old_norm_perc_scale = norm_perc_scale
            self.setp('norm', norm)
            
        

class Slider(Base):

    def __init__(self, ax, key, p):
        self.p = p
        self.key = str(key)
        self.slider = matplotlib.widgets.Slider(
            ax, self.key,
            self.p[self.key][1], self.p[self.key][2],
            valinit=self.p[self.key][0])
        self.slider.on_changed(self.sliders_onchanged)

    def sliders_onchanged(self, val):
        self.setp(self.key, val)
        
            
class Viewer(Base):


    def __init__(self, path, p):
        self.data = pyfits.open(path)[0].data.T
        self.p = p
        self.show()
        
    def show(self):
        self.fig = pl.figure(figsize=((7,8)), constrained_layout=True)
        slider_nb = 0
        while len(self.p) == 0:
            time.sleep(0.1)
        for key in self.p.keys():
            if self.p[key][1] is not None:
                slider_nb += 1
        height_ratios = list([1]) + list([0.1]) * slider_nb + list([0.2])
        gs = self.fig.add_gridspec(ncols=1, nrows=2 + slider_nb, height_ratios=height_ratios)
        self.axes = [self.fig.add_subplot(gs[i,0]) for i in range(2 + slider_nb)]
        if self.data.ndim == 2:
            self.axes[0].imshow(self.data, vmin=np.nanpercentile(self.data, 5), vmax=np.nanpercentile(self.data, 95))
        else: raise NotImplementedError('not implemented')

        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_onmove)

        # add sliders
        iindex = 1
        self.sliders = list()
        for key in self.p.keys():
            if self.p[key][1] is not None:
                self.sliders.append(Slider(self.axes[iindex], key, self.p))
                iindex += 1

        self.graph, = self.axes[-1].plot(np.zeros(BUFFERSIZE))
        self.axes[-1].set_ylim((-2**DEPTH, 2**DEPTH))
        ani = matplotlib.animation.FuncAnimation(self.fig,
                                      self.update_graph,
                                      interval=10,
                                      blit=True)

        pl.show()

    def update_graph(self, arg):
        out = self.getp('out')
        if out is not None:
            self.graph.set_ydata(out[:,0])
        return self.graph,
        
    def mouse_onmove(self, event):
        if event.button is matplotlib.backend_bases.MouseButton.LEFT:
            if event.inaxes is self.axes[0]:
                if event.xdata is not None and event.ydata is not None:
                    center = self.getp('center')
                    center[0] = int(event.xdata)
                    center[1] = int(event.ydata)
                    self.setp('center', center)
                    


class Sound(Base):
    
    def __init__(self, p, msg, data, channel):
        self.p = p
        self.data = data
        self.shape = np.array(self.data.shape)
        
        self.msg = msg
        self.worms = Worms(self.p)
        key_note = channel
        key_noteout = channel + 'out'
        
        sample = None

        stop = False
        stime = None
        release = self.getp('release')
        attack = self.getp('attack')
        
        while not stop:
            time.sleep(BUFFERSIZE/float(SAMPLERATE)/10.)

            # if nothing on the output feed it 
            if self.getp(key_noteout) is None:
                
                if sample is None:
                    sample = self.get_sample()
                    sample *= np.reshape(
                        neutron.utils.envelope(
                            sample.shape[0], (attack * SAMPLERATE)/sample.shape[0], 0, 1, 0.),
                        (sample.shape[0],1))
            
                    
                while sample.shape[0] <= max(release * SAMPLERATE, BUFFERSIZE) + BUFFERSIZE * 2.:
                    sample = np.concatenate((sample, self.get_sample()))

                if not self.getp(key_note): # not off, starting release
                    
                    if stime is None:
                        stime = time.time()
                        while sample.shape[0] <= release * SAMPLERATE + BUFFERSIZE:
                            sample = np.concatenate((sample, self.get_sample()))
                            
                        releasef = 1 - np.arange(sample.shape[0], dtype=float) / (release * SAMPLERATE)
                        releasef = np.reshape(np.where(releasef > 0, releasef, 0), (sample.shape[0], 1))
                        sample *= releasef
                        
                    
                    if time.time() - stime > release:
                        stop = True

                if not stop and sample is not None:
                    if sample.shape[0] > BUFFERSIZE:
                        last_sample = sample[:BUFFERSIZE,:] * msg.velocity / 64.
                        #print(last_sample[:-5,:])
                        self.setp(key_noteout, last_sample)
                        sample = sample[BUFFERSIZE:,:]
            
        self.setp(key_note, None)
        
    def _get_box(self, pos, size, harm):
        def transform(a, shift):
            a -= np.mean(a)
            #b = scipy.fftpack.fft(a, n=int(shift*a.size))
            b = np.fft.fft(a, n=int(shift*a.size))
            b = b[b.size//20:-b.size//20].real
            return b
        
        if len(pos) != self.data.ndim:
            raise Exception('center must be a tuple of length {}'.format(self.data.ndim))
        center = np.clip(pos, size, self.shape - size)
        box = self.data[center[0]-size:center[0]+size+1,
                        center[1]-size:center[1]+size+1,100:-100]
        #box = np.mean(np.mean(box, axis=0), axis=0)
        #box = np.median(box, axis=(0,1))
        box = np.min(box, axis=(0,1))
        #box = np.nanpercentile(box, 30, axis=(0,1))
        
        box = transform(box, harm)
        return box

    def get_sample(self):
        s = int(self.getp('size'))
        harm_scale = self.getp('harm_scale')
        note = self.msg.note
        shift = self.getp('shift')
        harm_step = shift / 2**((note) / 12) * 2#self.getp('harm_step')
        centers = self.worms.get()
        boxes = list()
        norm = self.getp('norm')
        nharm = len(centers) // NCHANNELS
        for ichan in range(NCHANNELS):
            box = self._get_box(centers[ichan*nharm], s, harm_step)
            for i in range(1, nharm):
                iscale = (nharm - i)/nharm * (1-harm_scale) + harm_scale
                box += (self._get_box(
                    centers[i+ichan*nharm], s, harm_step * 2**i)[:box.size]) * iscale
            box /= norm

            
            #box = box**(1/3)
            #box -= np.mean(box)
            #box *= neutron.utils.envelope(box.size, 0.1, 0, 1, 0.)
            boxes.append(box)
        
        boxes = np.array(boxes).T
        return boxes
        
        #print('gtep')
        #return neutron.utils.loop(boxes, self.getp('sample_size'), start=boxes.shape[0]//4, end=boxes.shape[0]//4, merge=boxes.shape[0]//8)



class Midi(Base):

    def __init__(self, p, data):
        self.p = p
        self.data = data

        self.inport = mido.open_input(USB_IN)
        self.sounds = list()
        self.notes = dict()
        for i in range(256):
            self.notes[i] = None

        clean_index = 0
        for msg in self.inport:
            print(msg)
            clean_index += 1
            if clean_index > 10:
                for isound in self.sounds:
                    if not isound[0].is_alive():
                        del isound
                    
                clean_index = 0
                
            time.sleep(0.001)
            if msg.type == 'note_on':
                if self.notes[msg.note] is not None: continue
                
                for i in range(MAX_NOTECHANNELS):
                    ichannel = str(i)
                    
                    # None : channel released
                    # False : note off
                    # True : note on
                    if self.getp(ichannel) is None:
                        self.notes[msg.note] = ichannel
                
                        self.setp(ichannel, True)
                        self.sounds.append(
                            (multiprocessing.Process(
                                name='sound', 
                                target=Sound,
                                args=(self.p, msg, self.data, ichannel)),
                             msg.note, ichannel))
                        
                        self.sounds[-1][0].start()
                        
                        break
                    
            elif msg.type == 'note_off':
                if self.notes[msg.note] is None: continue
                
                self.setp(self.notes[msg.note], False)
                self.notes[msg.note] = None
                
    def __del__(self):
        try:
            self.inport.close()
        except: pass
