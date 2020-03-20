import neutron.atom

#machine = neutron.core.Machine('/home/thomas/data/M1/m1.final.deep_frame.fits')
#machine = neutron.core.Machine('M1_subscube_small.fits', 'M1_subdf_small.fits')
core = neutron.atom.Core()#'M1_subscube.fits', 'M1_subdf.fits')


# import astropy.io.fits as pyfits
# import neutron.ccore
# import time
# import numpy as np

# from functools import wraps
# import errno
# import os
# import signal

# from multiprocessing import Process

# class TimeoutError(Exception):
#     pass

# def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
#     def decorator(func):
#         def _handle_timeout(signum, frame):
#             raise TimeoutError(error_message)

#         def wrapper(*args, **kwargs):
#             signal.signal(signal.SIGALRM, _handle_timeout)
#             signal.setitimer(signal.ITIMER_REAL, seconds)
#             try:
#                 result = func(*args, **kwargs)
#             finally:
#                 signal.alarm(0)
#             return result

#         return wraps(func)(wrapper)

#     return decorator

# data = pyfits.open('M1_subscube.fits')[0].data.T
# shift = 300
# size = 5

    

# for i in range(1000):
#     center = np.random.randint(100, data.shape[0]-100, size=2)
#     note  = np.random.randint(10,60)
#     box = data[center[0]-size:center[0]+size+1,
#                center[1]-size:center[1]+size+1,100:-100]
#     box = np.min(box, axis=(0,1))
    
#     harm = shift / 2**((note) / 12) * 2
#     box = box.astype(np.float32)
#     size = box.size
#     stime = time.time()
#     neutron.ccore.transform(box, harm)
#     timing = time.time() - stime
#     if timing > 0.005:
#         #print(neutron.ccore.fft_length(size, harm))
#         print(timing, note)
        
        
