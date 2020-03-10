import numpy as np

def envelope(n, a, d, s, r):
    x = np.arange(n, dtype=float) / (n-1)
    env = np.ones_like(x)
    if a > 0:
        env *= np.clip(x/a, 0, 1)
    if d > 0:
        env *= np.clip((1.-s) * (1.-((x-a)/d)), 0, (1-s)) + s
    if r > 0:
        env *= np.clip((-x+1)/r, 0, 1)
    return env

def loop(a, final_size, start=0, end=1, merge=0):
    b = np.copy(a[int(start):-int(end)])
    if b.size == 0: raise ValueError('oups, invalid parameters')
    while a.size < final_size:
        if merge > 0:
            a = np.concatenate((a[:-merge], (a[-merge:] + b[:merge])/2., b[merge:]))
        else:
            a = np.concatenate((a, b))
            
    return a[:final_size]
    
    
def delay(a, d, n, feedback):
    b = np.zeros((a.shape[0] + d * n, a.shape[1]), dtype=a.dtype)
    b[:a.shape[0],:] = a
    for i in range(n):
        b[(i+1)*d:a.shape[0]+((i+1)*d)] += a * feedback**i
    return b
    
