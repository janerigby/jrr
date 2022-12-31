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

def load_Pandeia_sensitivities(pandir=None) :   # Load Klaus's sensitivities for JWST
    if pandir == None:  pandir = '/Users/jrrigby1/MISSIONS/JWST/Sens/Pandeia_v2.0-sensitivity/'
    print("DEBUG, pandir was", pandir)
    filenames = [ basename(x) for x in glob.glob(pandir + '*npz')]
    pan = {} # empty dictionary of pandeia files (in weird numpy format)
    df = {}  # empty dict of dataframes
    for filename in filenames :
        (rootname, suffix) = filename.split('.')
        pan[rootname] = np.load(pandir + filename, fix_imports=True, encoding='latin1', allow_pickle=True)
        uglyintermed = []
        for ii in range(0, len(list(pan[rootname].items()))) :
            uglyintermed.append( list(pan[rootname].items())[ii][1].flatten())  #oh god it burns
        tempdf = pandas.DataFrame(data=uglyintermed)
        df[rootname] = tempdf.transpose()
        df[rootname].columns = list(pan[rootname].keys())
        # configus column is super ugly, a dict, w missing values.  splitting it into columns
        df[rootname]['configs'] = df[rootname]['configs'].apply(lambda x: {} if pandas.isna(x) else x)      
        df[rootname] = pandas.concat((df[rootname], pandas.json_normalize(df[rootname]['configs'], errors='ignore')), axis=1) #the pain
        # the above is screwing up MIRI LRS at the conversion to dataframes.  How to fix?
    return(df)

def load_Pandeia_dispersions(pandir=None) :
    # Grab dispersions from Pandeia.  Needed to calculate limiting spectroscopic line flux, rather than limiting continuum fnu
    # Must download reference files from: https://stsci.app.box.com/v/pandeia-refdata-v2p0-jwst
    # Installation instructions: https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+Installation
    if pandir == None:  pandir = '/Users/jrrigby1/MISSIONS/JWST/Sens/pandeia_data-2.0/jwst/'
    df = {}
    fig, ax = plt.subplots()
    filenames = [x for x in glob.glob(pandir + '*/dispersion/*fits')]
    for filename in filenames :
        rootname = re.split('_disp', basename(filename))[0]
        df[rootname] = Table(fits.getdata(filename)).to_pandas()   # I'm not dealing with astropy fits_rec crap. Converting to pandas. Ah.
        df2 =  df[rootname]  # pointer, keep this shorter
        if   'WAVELENGTH' in df2.keys()  : wavecol = 'WAVELENGTH'
        elif 'Wavelength' in df2.keys()  : wavecol = 'Wavelength'
        else : raise Exception("ERROR: cannot find the wavelength key in", rootname)
        df2.plot(x=wavecol, y='R', ax=ax, label=rootname)
        plt.xlabel ("Wavelength (micron)")
        plt.ylabel("Spectral resolution R")
    return(df)

def import_Pandeia_miri_lrs_special(filename='../Pandeia_v2.0-sensitivity/miri_lrs_sensitivity.npz'):
    # MIRI LRS pickles have different format than rest of Pandeia sensitivities.  It's awful.  Here is
    # a script that works around the ugliness.  Just treat LRS special and move on with life
    miri_lrs = dict(np.load(filename, fix_imports=True, encoding='latin1', allow_pickle=True))
    uglyintermed_slitless = []
    uglyintermed_slit  = []
    keys_not_configs = ['wavelengths', 'sns', 'lim_fluxes', 'source_rates_per_njy', 'sat_limits', 'line_limits'] # these have same # of elements
    for key in keys_not_configs:
        uglyintermed_slitless.append(miri_lrs[key][0])
        uglyintermed_slit.append(  miri_lrs[key][1]) 
    df0 = pandas.DataFrame(data=uglyintermed_slitless).transpose()
    df1 = pandas.DataFrame(data=uglyintermed_slit).transpose()
    df0.columns = keys_not_configs
    df1.columns = keys_not_configs
    return(df0, df1)   # slitless, slit
