import neutron.atom

#machine = neutron.core.Machine('/home/thomas/data/M1/m1.final.deep_frame.fits')
#machine = neutron.core.Machine('M1_subscube_small.fits', 'M1_subdf_small.fits')
core = neutron.atom.Core('../data/M1_subscube.float32.fits')#, 'M1_subdf.fits')

#import astropy.io.fits as pyfits
#import numpy as np
#data = pyfits.open('../data/M1_subscube.fits')[0].data.T.astype(np.float32)
#pyfits.writeto('../data/M1_subscube.float32.fits', data=data)

# import neutron.ccore
# import numpy as np
# a = neutron.ccore.SineWave()
# print(a.get_samples())

