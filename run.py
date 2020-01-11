import neutron.core

machine = neutron.core.Machine('/home/thomas/data/M1/m1.final.deep_frame.fits')



# import multiprocessing
# import time
# import pylab as pl
# import sys
# import numpy as np

# def read_param(p, t):
#     while True:
#         time.sleep(t)
#         print('pouet_lock', p['a'])
            

# class Change(object):

#     def __init__(self, p, t):
#         self.p = p
#         fig, ax = pl.subplots()
#         ax.plot(np.arange(25))
#         fig.canvas.mpl_connect('motion_notify_event', self.mouse_onmove)
#         pl.show()

#     def mouse_onmove(self, event):
#         self.p['a'] = event.xdata
    
        

# mgr = multiprocessing.Manager()
# param = mgr.dict()
# param['a'] = 1

# w1 = multiprocessing.Process(name='read', 
#                              target=read_param,
#                              args=(param, 0.3))

# w2 = multiprocessing.Process(name='change', 
#                              target=Change, 
#                              args=(param, 0.001))
# w1.start()
# w2.start()
# w1.join()
# w2.join()


