import numpy as np
import re
from   astropy.io import fits

'''Routines for various telescopes and instruments.'''
        
def fire_load_fits(filename) :
    ''' Load a FIRE spectrum in .fits format, from MIT pipeline.
    Should return np arrays of wavelength (angstroms), flux (flam?),
    and uncertainty spectrum'''
    flux, header = fits.getdata(filename, header=True)
    sig_filename = re.sub('F.fits', 'E.fits', filename) 
    sig = fits.getdata(filename)
    pixels = np.arange(start=1, stop=flux.shape[0]+1)  #first pix is 1, not 0
    wave = 10**(header['crval1'] + header['cdelt1'] * (pixels - header['crpix1']))
    return(wave, flux, sig)  

def load_Klaus_sensitivities(pandir) :   # Load Klaus's sensitivities for JWST
    if pandir == None:  pandir = '/Users/jrrigby1/MISSIONS/JWST/Sens/pandeia_sensitivities_1.5/'
    filenames = [ basename(x) for x in glob.glob(pandir + '*npz')]
    pan = {} # empty dictionary of pandeia files (in weird numpy format)
    df = {}  # empty dict of dataframes
    for filename in filenames :
        (rootname, suffix) = filename.split('.')
        pan[rootname] = np.load(pandir + filename, allow_pickle=True)

        uglyintermed = []
        for ii in range(0, len(list(pan[rootname].items()))) :
            uglyintermed.append( list(pan[rootname].items())[ii][1].flatten())  #oh god it burns
        tempdf = pandas.DataFrame(data=uglyintermed)
        df[rootname] = tempdf.transpose()
        df[rootname].columns = list(pan[rootname].keys())
    return(df)
