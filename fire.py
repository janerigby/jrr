import numpy as np
import re
from   astropy.io import fits

'''Routines for FIRE/Magellan spectra'''
        
def FIRE_load_fits(filename) :
    ''' Load a FIRE spectrum in .fits format, from MIT pipeline.
    Should return np arrays of wavelength (angstroms), flux (flam?),
    and uncertainty spectrum'''
    flux, header = fits.getdata(filename, header=True)
    sig_filename = re.sub('F.fits', 'E.fits', filename) 
    sig = fits.getdata(filename)
    pixels = np.arange(start=1, stop=flux.shape[0]+1)  #first pix is 1, not 0
    wave = 10**(header['crval1'] + header['cdelt1'] * (pixels - header['crpix1']))
    return(wave, flux, sig)  
