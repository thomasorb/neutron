import astropy.io.fits as pyfits
import numpy as np
#import pyaudio
import sounddevice as sd
import wave
import sys
import pylab as pl
import multiprocessing
import matplotlib.backend_bases
import matplotlib.widgets
import time
import neutron.utils
print(sd.query_devices())
sd.default.samplerate = 48000
sd.default.device = 'HDA Intel PCH: ALC293 Analog (hw:0,0)'

NCHANNELS = 2

class Machine(object):

    def __init__(self, path, dfpath=None):

        if dfpath is None:
            dfpath = path

        self.mgr = multiprocessing.Manager()
        self.p = self.mgr.dict()
        
        self.player_process = multiprocessing.Process(
            name='player', 
            target=Player, 
            args=(path, self.p))


        self.player_process.start()
        
        
        self.viewer_process = multiprocessing.Process(
            name='viewer', 
            target=Viewer, 
            args=(dfpath, self.p))

        self.viewer_process.start()


        self.viewer_process.join()
        self.player_process.join()
        

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

    def __init__(self, path, p, sample_width=2):
        self.data = pyfits.open(path)[0].data.T
        self.shape = np.array(self.data.shape)
        self.data_sub = self.data.flatten()
        np.random.shuffle(self.data_sub)
        self.data_sub = self.data_sub[:int(0.002*self.data_sub.size)]
        print(self.data_sub.shape)
        
        self.p = p
        self.p['harm_number'] = (1, 1, 20, int)
        self.p['harm_step'] = (100, 10, 300, int)
        self.p['center'] = ((self.shape / 2).astype(int), None, None, None)
        self.p['size'] = (10, 1, 30, int)
        self.p['stop'] = (False, None, None, None)
        self.p['worm_move_scale'] = (10, 1, 100, int)
        self.p['harm_scale'] = (0., -2., 2., float)
        self.p['norm_perc_scale'] = (4, 0, 7, float)
        self.old_norm_perc_scale = None
        self.get_norm()
        
        self.worms = Worms(self.p)
        
        self.stream = sd.OutputStream(dtype=np.int16, channels=NCHANNELS)
        self.stream.start()
        self.play()
                
        
    def _get_box(self, pos, size, harm):
        def transform(a, shift):
            a -= np.mean(a)
            b = np.fft.fft(a, n=shift*a.size)
            b = b[b.size//8:b.size//3].real
            return b
        
        if len(pos) != self.data.ndim:
            raise Exception('center must be a tuple of length {}'.format(self.data.ndim))
        center = np.clip(pos, size, self.shape - size)
        print(center)
        if self.data.ndim == 3:
            box = self.data[center[0]-size:center[0]+size+1,
                            center[1]-size:center[1]+size+1,10:-10]
            box = np.mean(np.mean(box, axis=0), axis=0)
            
        else:
            slices = tuple([slice(c - size, c + size + 1) for c in center])
            box = self.data.__getitem__(slices)
            box = box.flatten()
        
        box = transform(box, harm)
        return box

    def get_norm(self):
        if float(self.getp('norm_perc_scale')) != self.old_norm_perc_scale:
            self.norm = np.nanpercentile(self.data_sub, 100-10**(1-self.getp('norm_perc_scale')))
            self.old_norm_perc_scale = float(self.getp('norm_perc_scale'))
        return self.norm
            
    def get_sample(self):
        s = int(self.getp('size'))
        harm_scale = self.getp('harm_scale')
        centers = self.worms.get()
        boxes = list()
        nharm = len(centers) // NCHANNELS
        for ichan in range(NCHANNELS):
            box = self._get_box(centers[ichan*nharm], s, self.getp('harm_step'))
            for i in range(1, nharm):
                iscale = (nharm - i)/nharm * (1-harm_scale) + harm_scale
                box += (self._get_box(
                    centers[i+ichan*nharm], s, self.getp('harm_step') * 2**i)[:box.size]) * iscale
            box /= self.get_norm()

            box -= np.mean(box)
            box *= neutron.utils.envelope(box.size, 0.01, 0, 1, 0.)
            boxes.append(box)
        return np.array(boxes).T
        
    def play_sample(self):
        sample = self.get_sample()
        nmix = int(sample.shape[0]*0.05)
        if hasattr(self, 'next_sample'):
            if self.next_sample.size == nmix:
                sample[:nmix] *= np.atleast_2d(neutron.utils.envelope(nmix,1,0,0,0)).T
                sample[:nmix] += self.next_sample * np.atleast_2d(neutron.utils.envelope(nmix,0,1,0,0)).T
            

        self.next_sample = sample[-nmix:]
        sample = sample[:-nmix]
        
        
        sample *= 2**15 - 1
        sample = np.ascontiguousarray(sample.astype(np.int16))
        self.stream.write(sample)

    def play(self):
        while True:
            self.play_sample()
        

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
        width_ratios = list([1]) + list([0.1]) * slider_nb
        gs = self.fig.add_gridspec(ncols=1, nrows=1 + slider_nb, height_ratios=width_ratios)
        self.axes = [self.fig.add_subplot(gs[i,0]) for i in range(1 + slider_nb)]
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
        
        pl.show()
        
        
    def mouse_onmove(self, event):
        if event.button is matplotlib.backend_bases.MouseButton.LEFT:
            if event.inaxes is self.axes[0]:
                if event.xdata is not None and event.ydata is not None:
                    center = self.getp('center')
                    center[0] = int(event.xdata)
                    center[1] = int(event.ydata)
                    self.setp('center', center)
                    
    
