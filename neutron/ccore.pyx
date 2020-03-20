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


cdef int SAMPLERATE = 48000
cdef int BYTEDEPTH = 32

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
    

def transform(np.ndarray[np.float32_t, ndim=3] a, double shift):
    
    cdef int n = <int> shift * a.shape[2]
    cdef np.ndarray[np.float32_t, ndim=1] amin = np.empty(n, dtype=np.float32)
    amin = min_along_z(a)
    n = scipy.fft.next_fast_len(n)
    amin -= np.mean(amin)
    amin = scipy.fft.rfft(amin, n=n, overwrite_x=True).real
    return amin[amin.size//20:-amin.size//20]


def ndarray2buffer(np.ndarray[np.float32_t, ndim=2] arr):
    global BYTEDEPTH
    arr *= 2**(BYTEDEPTH - 2) - 1
    return np.ascontiguousarray(arr.astype(np.int32))

def sine(float f, int n, int srate):
    cdef np.ndarray[np.float32_t, ndim=1] x = np.arange(n, dtype=np.float32) / <float> srate
    return (np.sin(f * x * 2 * np.pi))

def note2f(int note):
    return 440. / 2**((49 - <float> note) / 12.)

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

    cdef public int note, index, t, velocity
    cdef public np.float32_t[:] sampleL
    cdef public np.float32_t[:] sampleR
    
    def __init__(self, int note, int t, int velocity):
        self.note = note
        self.index = 0
        self.t = t
        self.sampleL, self.sampleR = self.get_base_sample()
        
        self.velocity = velocity

    def get_base_sample(self):
        global SAMPLERATE
        cdef np.ndarray[np.float32_t, ndim=1] zeros = np.zeros(
            (SAMPLERATE * self.t), dtype=np.float32)
        return zeros, zeros
    
    def get_buffers(self, int BUFFERSIZE, float sv=1, float ev=1, float volume=0.5):
        bufL = self.sampleL[self.index:self.index + BUFFERSIZE]
        bufR = self.sampleR[self.index:self.index + BUFFERSIZE]
        cdef int i
        cdef float ienv
        
        self.index += BUFFERSIZE
        if self.index + BUFFERSIZE >= SAMPLERATE * self.t:
            self.index = 0

        for i in range(BUFFERSIZE):
            ienv = i / <float> BUFFERSIZE * (ev - sv) + sv
            ienv *= <float> self.velocity / 64. * volume
            bufL[i] = bufL[i] * ienv 
            bufR[i] = bufR[i] * ienv
        return bufL, bufR
    
cdef class SineWave(NullWave):

    def get_base_sample(self):
        global SAMPLERATE
        cdef np.ndarray[np.float32_t, ndim=1] s = sine(
            note2f(self.note),
            SAMPLERATE * self.t,
            SAMPLERATE)
        return np.array([s, s])

@cython.boundscheck(False)
@cython.wraparound(False)
def sound(out_bufferL, out_bufferR, out_i, notes, msg, outlock, double timing, float attack, float release, int BUFFERSIZE, float MASTER, float SLEEPTIME, float PRELOADTIME):
    cdef float velocity = msg.velocity
    cdef bool stop = False
    cdef double rel_stime = 0
    cdef double att_stime = 0
    cdef long lastbuffer = 0
    cdef double sva, eva, svr, evr, now
    cdef np.float32_t[:] bufL
    cdef np.float32_t[:] bufR
    cdef int i
    
    #logging.debug('start sound {}'.format(msg.note))

    if msg.channel == 1:
        wave = SineWave(msg.note, PRELOADTIME, msg.velocity)
    else:
        wave = NullWave(msg.note, PRELOADTIME, msg.velocity)

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
                                              sv=sva * svr,
                                              ev=eva * evr,
                                              volume=MASTER)

                for i in range(BUFFERSIZE):
                    out_bufferL[i] += bufL[i]
                    out_bufferR[i] += bufR[i]
                
        finally:
            outlock.release()

        # note off, starting release
        if notes[int(msg.note)] != timing: 

            if rel_stime == 0:
                #logging.debug('release sound {}'.format(msg.note))
                rel_stime = now

            if now - rel_stime >= release:
                stop = True

        time.sleep(SLEEPTIME)

    #logging.debug('end sound {}'.format(msg.note))


