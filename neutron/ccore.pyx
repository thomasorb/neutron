cimport cython
import cython

cimport numpy as np
import numpy as np
import scipy.fft
import time
import logging

from cpython cimport bool

## Import functions from math.h (faster than python math.py)
cdef extern from "math.h" nogil:
    double cos(double theta)
    double sin(double theta)
    double exp(double x)
    double sqrt(double x)
    double log(double x)
    double ceil(double x)
    double floor(double x)
    double M_PI
    double isnan(double x)

# define long double for numpy arrays
ctypedef long double float128_t

cdef int SAMPLERATE = 44100
cdef int BYTEDEPTH = 32
cdef int A_MIDIKEY = 45

@cython.boundscheck(False)
@cython.wraparound(False)
def min_along_z(np.ndarray[np.float32_t, ndim=3] a):
    
    cdef int n = a.shape[2]
    cdef np.ndarray[np.float32_t, ndim=1] b = np.empty(n, dtype=np.float32)
    cdef int i, j, k, si, sj
    cdef float minval
    si = a.shape[0]
    sj = a.shape[1]
    with nogil:
        for ik in range(n):
            minval = a[0,0,ik]
            for ii in range(si):
                for ij in range(sj):
                    if a[ii,ij,ik] < minval:
                        minval = a[ii,ij,ik]
            b[ik] = minval
    return b
    

def transform3d(np.ndarray[np.float32_t, ndim=3] a, double shift):
    cdef np.ndarray[np.float32_t, ndim=1] b = transform1d(min_along_z(a), shift)
    b -= np.mean(b)
    return b

def transform1d(np.ndarray[np.float32_t, ndim=1] a, int note, int basenote):
    cdef double shift = note2shift(note, basenote)
    shift = max(1, shift)

    cdef int n = <int> (2 * shift * <float> a.size)
    if n&1: n+=1
    
    cdef np.ndarray[np.float32_t, ndim=1] afft = np.empty(n, dtype=np.float32)
    cdef int border
    
    n = scipy.fft.next_fast_len(n)
    afft = scipy.fft.irfft(a, n=n, overwrite_x=True).real * shift * 2
    border = afft.size//20
    return np.array(afft[border:-border]), afft.size - 2 * border

def ndarray2buffer(np.ndarray[np.float32_t, ndim=2] arr):
    global BYTEDEPTH
    arr *= 2**(BYTEDEPTH - 2) - 1
    return np.ascontiguousarray(arr.astype(np.int32))

def sine(float f, int n, int srate):
    cdef np.ndarray[np.float32_t, ndim=1] x = np.arange(n, dtype=np.float32) / <float> srate
    return np.cos(f * x * 2. * np.pi)

def note2f(int note):
    return 440. / 2**((A_MIDIKEY - <float> note) / 12.) / 2.

def note2shift(int note, int basenote):
    return note2f(basenote) / note2f(note)

def release_values(double time, double stime, double rtime, int buffersize):
    cdef double sv, ev, deltat
    global SAMPLERATE
    deltat = rtime - time + stime
    sv = deltat / rtime
    ev = (deltat - <double> buffersize / <double> SAMPLERATE) / rtime
    ev = max(0, ev)
    return sv, ev

def attack_values(double time, double stime, double atime, int buffersize):
    cdef double sv, ev, deltat
    global SAMPLERATE
    deltat = time - stime
    sv = deltat / atime
    sv = min(1, max(0, sv))
    ev = (deltat + <double> buffersize / <double> SAMPLERATE) / atime
    ev = min(1, ev)
    return sv, ev


cdef class NullWave(object):

    cdef public int index, nbuf
    cdef public np.float32_t[:] sampleL
    cdef public np.float32_t[:] sampleR
    cdef public int N, BASENOTE
    
    def __init__(self, int nbuf, int note, int BASENOTE):
        self.index = 0
        self.nbuf = nbuf
        self.BASENOTE = BASENOTE
        cdef np.ndarray[np.float32_t, ndim=1] base_sampleL
        cdef np.ndarray[np.float32_t, ndim=1] base_sampleR
        
        base_sampleL, base_sampleR = self.get_base_sample()
        
        self.sampleL, self.N = transform1d(base_sampleL[:], note, BASENOTE)
        self.sampleR, self.N = transform1d(base_sampleR[:], note, BASENOTE)
        

    def get_base_sample(self):
        global SAMPLERATE
        cdef np.ndarray[np.float32_t, ndim=1] zeros = np.zeros(
            self.nbuf, dtype=np.float32)
        return zeros, zeros

    def get_buffers(self, int BUFFERSIZE, int velocity,
                    float sv=1, float ev=1, float volume=0.5):

        cdef int i
        cdef float ienv
        cdef np.float32_t[:] bufL = np.empty(BUFFERSIZE, dtype=np.float32)
        cdef np.float32_t[:] bufR = np.empty(BUFFERSIZE, dtype=np.float32)
        
        bufL = np.copy(self.sampleL[self.index:self.index + BUFFERSIZE])
        bufR = np.copy(self.sampleR[self.index:self.index + BUFFERSIZE])

        self.index += BUFFERSIZE
        if self.index + BUFFERSIZE >= self.N:
            self.index = 0

        for i in range(BUFFERSIZE):
            ienv = i / <float> BUFFERSIZE * (ev - sv) + sv
            ienv *= <float> velocity / 64. * volume
            bufL[i] = bufL[i] * ienv 
            bufR[i] = bufR[i] * ienv
        return bufL, bufR
    
cdef class SineWave(NullWave):

    
    def get_base_sample(self):
        global SAMPLERATE
        cdef np.ndarray[np.float32_t, ndim=1] s = sine(
            note2f(self.BASENOTE),
            self.nbuf * 2, SAMPLERATE)
        cdef int n = scipy.fft.next_fast_len(s.size)
        s = scipy.fft.rfft(s, n=n, overwrite_x=True).real
        return np.array([s, s])

cdef class DataWave(NullWave):

    cdef public np.float32_t[:,:,:] data
    
    def __init__(self, np.ndarray[np.float32_t, ndim=3] data, int nbuf, int note, int BASENOTE):
        self.data = data
        NullWave.__init__(self, nbuf, note, BASENOTE)

    def get_base_sample(self):
        return np.array([self.data[230,230,:], self.data[231,237,:]])


@cython.boundscheck(False)
@cython.wraparound(False)
def sound(out_bufferL, out_bufferR, out_i, notes, int note, int velocity, int channel,
          outlock, double timing, float attack, float release,
          int BUFFERSIZE, float MASTER, float SLEEPTIME, int PRELOADSAMPLES, int BASENOTE,
          np.ndarray[np.float32_t, ndim=3] data):
    
    cdef bool stop = False
    cdef double rel_stime = 0
    cdef double att_stime = 0
    cdef long lastbuffer = 0
    cdef double sva, eva, svr, evr, now
    cdef np.float32_t[:] bufL
    cdef np.float32_t[:] bufR
    cdef int i
    
    if channel == 0:
        wave = DataWave(data, PRELOADSAMPLES, note, BASENOTE)
    else:
        wave = SineWave(PRELOADSAMPLES, note, BASENOTE)

    while not stop:
        now = time.time()        

        if att_stime == 0:
            att_stime = now
            
        svr = 1
        evr = 1
        if rel_stime > 0:
            svr, evr = release_values(now, rel_stime, release, BUFFERSIZE)
        sva, eva = attack_values(now, att_stime, attack, BUFFERSIZE)

        outlock.acquire()
        try:
            if out_i.value > lastbuffer:
                lastbuffer = out_i.value
                bufL, bufR = wave.get_buffers(BUFFERSIZE,
                                              velocity,
                                              sv=sva * svr,
                                              ev=eva * evr,
                                              volume=MASTER)

                for i in range(BUFFERSIZE):
                    out_bufferL[i] += bufL[i]
                    out_bufferR[i] += bufR[i]
                
        finally:
            outlock.release()

        # note off, starting release
        if notes[int(note)] != timing: 

            if rel_stime == 0:
                #logging.debug('release sound {}'.format(note))
                rel_stime = now

            if now - rel_stime >= release:
                stop = True

        time.sleep(SLEEPTIME)

    #logging.debug('end sound {}'.format(note))


