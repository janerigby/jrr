''' Some basic photometry not covered elsewhere.  jrigby Nov 2021 '''

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.io.fits import getdata
import pyregion


def basic_photstats_ma(subset) :  # helper function for pyregion_photometry
    # subset is a numpy.ma masked array, w mask from the region file
    result = {
        'npix': ma.count(subset), 'mean': ma.mean(subset),
        'median': ma.median(subset), 'thesum': ma.sum(subset)
        }
    return(result)


def basic_photstats(subset) :  # helper function for pyregion_photometry
    # subset is a numpy.ma masked array, w mask from the region file
    result = {
        'npix': np.count_nonzero(subset),      'mean': np.nanmean(subset),
        'median': np.nanmedian(subset), 'thesum': np.nansum(subset)
        }
    return(result)



# These approaches should give same results.  They use different logic to apply the mask.  Both are slow.
# no time now to speed them up

def pyregion_photometry_ma(imagefile, regfile) :
    # Note, regions files need to be in IMAGE coordinates.  Pyregions not working w regions that use WCS
    f = fits.open(imagefile)
    results = {} # a dict
    dum = f[0].data  # pyregions wants hdu in this picky way
    im = dum.astype(np.float64)
    reg = pyregion.open(regfile).as_imagecoord(f[0].header)
    #mask: everything inside first region in regions file is True, and outside is False
    mask   = reg.get_mask(f[0])  # masked array wants the opposite sign convention from pyregions
    subset = ma.MaskedArray(data=im, mask=np.invert(mask)) 
    photreg  = basic_photstats_ma(subset)
    # In progress.  write background later, and do net photometry. steal from S0033_photometry_v2.ipynb
    return(photreg)

def pyregion_photometry(imagefile, regfile) :
    # Note, regions files need to be in IMAGE coordinates.  Pyregions not working w regions that use WCS
    f = fits.open(imagefile)
    results = {} # a dict
    dum = f[0].data  # pyregions wants hdu in this picky way
    im = dum.astype(np.float64)
    reg = pyregion.open(regfile).as_imagecoord(f[0].header)
    #mask: everything inside first region in regions file is True, and outside is False
    mask   = reg.get_mask(f[0])
    mask2 = mask.astype(float)
    mask2[mask2 == 0] = np.nan
    subset = mask2 * im
    photreg  = basic_photstats(subset)
    return(photreg)
    # In progress.  write background later, and do net photometry. steal from S0033_photometry_v2.ipynb

