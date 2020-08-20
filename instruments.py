import numpy as np
import re
import glob
from os.path import basename
from astropy.io    import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import pandas

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

def load_Klaus_sensitivities(pandir=None) :   # Load Klaus's sensitivities for JWST
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
        # configus column is super ugly, a dict, w missing values.  splitting it into columns
        df[rootname]['configs'] = df[rootname]['configs'].apply(lambda x: {} if pandas.isna(x) else x)      
        df[rootname] = pandas.concat((df[rootname], pandas.json_normalize(df[rootname]['configs'], errors='ignore')), axis=1) #the pain
    return(df)

def load_Klaus_dispersions(pandir=None) :
    # Try grabbing dispersion, so that for spectroscopy I can plot limiting line flux for an unresolved line, rather than fnu
    if pandir == None:  pandir = '/Users/jrrigby1/MISSIONS/JWST/Sens/pandeia_data-1.5rc2/jwst/'
    df = {}
    fig, ax = plt.subplots()
    filenames = [x for x in glob.glob(pandir + '*/dispersion/*fits')]
    for filename in filenames :
        rootname = re.split('_disp', basename(filename))[0]
        df[rootname] = Table(fits.getdata(filename)).to_pandas()   # I'm not dealing with astropy fits_rec crap. Converting to pandas. Ah.
        if   'WAVELENGTH' in df[rootname].keys()  : wavecol = 'WAVELENGTH'
        elif 'Wavelength' in df[rootname].keys()  : wavecol = 'Wavelength'
        else : raise Exception("ERROR: cannot find the wavelength key in", rootname)
        df[rootname].plot(x=wavecol, y='R', ax=ax, label=rootname)
        plt.xlabel ("Wavelength (micron)")
        plt.ylabel("Spectral resolution R")
    return(df)
