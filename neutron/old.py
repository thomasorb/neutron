"""PyAudio Example: Play a wave file."""

def get_sound(im, x,y,s,zp=1):
    box = get_box(im, x, y, s)
    box = box.flatten()
    #box = np.nanmean(box, axis=1)
    box -= np.mean(box)
    ifft = box
    #x = np.arange(box.size)
    #SINF = 0.08
    #ifft *= np.sin(SINF * x)# + np.sin(SINF * 2 * x) + np.sin(SINF * 4 * x)
    #ifft = np.fft.fft(box, n=zp*box.size)
    #ifft = ifft[:box.size]
    #ifft = np.fft.ifft(ifft, n=1*ifft.size)
    #pl.plot(ifft)
    return ifft

import pyaudio
import wave
import sys

CHUNK = 1024


wf = wave.open('sound.wav', 'rb')

# instantiate PyAudio (1)



# play stream (3)

nharm = 10
ii = np.array(list([1000]) * nharm * 2)
ij = np.array(list([1000]) * nharm * 2)
ii += (np.random.standard_normal(size = ii.size) * 3).astype(int)
ij += (np.random.standard_normal(size = ij.size) * 3).astype(int)
print(ii)
scale = 5
size = 60
HARMP = 30
sound_norm = np.nanpercentile(im,99.9)
path = list()
for _ in range(5000):
    ii += int(np.random.standard_normal() * scale)
    ij += int(np.random.standard_normal() * scale)
    path.append((ii[0],ij[0]))
    isound_l = get_sound(im, ii[0], ij[0], size)
    #isound_r = get_sound(im, ii[1], ij[1], size)
    for iharm in range(1,nharm):
        inorm = (nharm-iharm/2)/nharm
        isound_l += get_sound(im, ii[iharm*2], ij[iharm*2], size+iharm*HARMP)[:isound_l.size] * inorm
    #    isound_r += get_sound(im, ii[iharm*2+1], ij[iharm*2+1], size+iharm*HARMP)[:isound_l.size] * inorm
    sound_l = isound_l
    #sound_r.append(isound_r)
    arr = sound_l
    arr /= sound_norm
    #arr[arr > 1] = 1.
    #arr[arr <-1] = -1.
    arr *= 32767
    arr = arr.astype(np.int16)
    stream.write(arr)

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()
