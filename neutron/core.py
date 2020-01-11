import astropy.io.fits as pyfits
import numpy as np
#import pyaudio
import sounddevice as sd
import wave
import sys
import pylab as pl
import multiprocessing
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Slider
import time

print(sd.query_devices())
sd.default.samplerate = 48000
sd.default.device = 'HDA Intel PCH: ALC293 Analog (hw:0,0)'


class Machine(object):

    def __init__(self, path):

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
            args=(path, self.p))

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
            self.worms = [Worm(self.p) for i in range(self.getp('harm_number'))]
            
        if len(self.worms) == self.getp('harm_number'):
            return

        while len(self.worms) > self.getp('harm_number'):
            self.worms.pop(0)

        while len(self.worms) < self.getp('harm_number'):
            self.worms.append(Worm(self.p))

    def get(self):
        self.update()
        return [iworm.get() for iworm in self.worms]
    
class Player(Base):

    def __init__(self, path, p, nchannels=2, rate=44100, sample_width=2):
        self.data = pyfits.open(path)[0].data
        self.reset_norm()
        self.shape = np.array(self.data.shape)
        
        self.p = p
        self.p['harm_number'] = (5, 1, 20, int)
        self.p['harm_step'] = (30, 1, 100, int)
        self.p['center'] = ((self.shape / 2).astype(int), None, None, None)
        self.p['size'] = (60, 1, 200, int)
        self.p['stop'] = (False, None, None, None)
        self.p['worm_move_scale'] = (10, 1, 100, int)
        self.p['harm_scale'] = (0., 0., 2., float)

        self.worms = Worms(self.p)
        
        self.stream = sd.OutputStream(dtype=np.int16, channels=1)

        self.stream.start()
        self.play()
        

        
    def reset_norm(self, perc=99.9):
        self.norm = np.nanpercentile(self.data, perc)

    def _get_box(self, pos, size):
        if len(pos) != self.data.ndim:
            raise Exception('center must be a tuple of length {}'.format(self.data.ndim))
        center = np.clip(pos, size, self.shape - size)
        slices = tuple([slice(c - size, c + size + 1) for c in center])
        box = self.data.__getitem__(slices).flatten()
        return box
    
    def get_sample(self):
        s = int(self.getp('size'))
        harm_scale = self.getp('harm_scale')
        centers = self.worms.get()
        box = self._get_box(centers[0], s)
        for i in range(1, len(centers)):
            iscale = (len(centers) - i)/len(centers) * (1-harm_scale) +  harm_scale
            box += (self._get_box(centers[i], s + i * self.getp('harm_step'))[:box.size]) * iscale
        box/= self.norm
        box -= np.mean(box)
        return box
        
    def play_sample(self):  
        sample = self.get_sample()
        sample *= 2**15 - 1
        sample = sample.astype(np.int16)
        self.stream.write(sample)

    def play(self):
        while True:
            self.play_sample()
        

class ViewerSlider(Base):
    pass
    
            
class Viewer(Base):


    def __init__(self, path, p):
        self.data = pyfits.open(path)[0].data
        self.p = p
        self.show()
        
    def show(self):
        self.fig = pl.figure(constrained_layout=True)
        slider_nb = 0
        while len(self.p) == 0:
            time.sleep(0.1)
        for key in self.p.keys():
            if self.p[key][1] is not None:
                slider_nb += 1
        #self.fig.subplots_adjust(left=0.25, bottom=0.25)
        width_ratios = list([1]) + list([0.1]) * slider_nb
        gs = self.fig.add_gridspec(ncols=1, nrows=1 + slider_nb, height_ratios=width_ratios)
        self.axes = [self.fig.add_subplot(gs[i,0]) for i in range(1 + slider_nb)]
        if self.data.ndim == 2:
            self.axes[0].imshow(self.data)
        else: raise NotImplementedError('not implemented for a cube')

        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_onmove)

        # add sliders
        iindex = 1
        self.sliders = list()
        for key in self.p.keys():
            if self.p[key][1] is not None:
                islider = Slider(self.axes[iindex], key,
                                 self.p[key][1], self.p[key][2],
                                 valinit=self.p[key][0])
                islider.on_changed(self.sliders_onchanged)
                self.sliders.append(islider)
                iindex += 1
        
        pl.show()

    def sliders_onchanged(self, val):
        for islider in self.sliders:
            islider.val)
        
        
    def mouse_onmove(self, event):
        if event.button is MouseButton.LEFT:
            if event.inaxes is self.axes[0]:
                if event.xdata is not None and event.ydata is not None:
                    self.setp('center', np.array((int(event.xdata),int(event.ydata))))
                    
    
