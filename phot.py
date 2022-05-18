''' Some basic photometry not covered elsewhere.  jrigby Nov 2021 '''

import pandas
import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.io.fits import getdata
import pyregion
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import regions # Maybe astropy regions will work better than pyregion


def basic_photstats_ma(subset) :  # helper function for pyregion_photometry
    # subset is a numpy.ma masked array, w mask from the region file
    result = {
        'npix': ma.count(subset), 'mean': ma.mean(subset), \
        'median': ma.median(subset), 'thesum': ma.sum(subset), 'stddev': ma.std(subset)
        }
    return(result)


def basic_photstats(subset, clip_sigma=5, clip_grow=0, clip_maxiters=2) :  # helper function for pyregion_photometry
    # subset is a numpy.ma masked array, w mask from the region file
    result = {
        'npix': np.count_nonzero(subset),      'mean': np.nanmean(subset), \
        'median': np.nanmedian(subset), 'thesum': np.nansum(subset), 'stddev': np.nanstd(subset),
        'clipped_median': sigma_clipped_stats(data=subset, mask_value=np.nan, sigma=clip_sigma, grow=clip_grow)[1]
        }
    return(result)



# These approaches should give same results.  They use different logic to apply the mask.  Both are slow.
# No time now to speed them up

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


def photometry_loop_regions(imagefile, regfile, debug=False, override_label=None, clip_sigma=5, clip_grow=0, clip_maxiters=2) :   # Uses Regions, not pyregions
    # This works for ds9 regions files in either image coords or WCS coords
    # This does not yet use background subtraction.  It should!
    reg = regions.Regions.read(regfile, format='ds9')
    imdata, header = fits.getdata(imagefile, header=True)
    thewcs = WCS(header)
    results = {}  #empty dict
    for thisreg in reg:
        if 'label' in thisreg.meta.keys() :  label = thisreg.meta['label']
        elif override_label : label = override_label
        else : label = 'foo'
        if hasattr(thisreg, 'to_pixel') : # If it has a to_pixel method, then assume it's in sky coords
            if debug : print("Making a pixel region from WCS fk5 region")
            pixreg = thisreg.to_pixel(thewcs)
        else :  pixreg = thisreg  # otherwise assume region
        # check whether region overlaps w data
        mask = pixreg.to_mask(mode='subpixels')
        masked_data = mask.multiply(imdata, fill_value=np.nan)
        if masked_data.sum() <= 0 : raise Exception("Error: Region and image may not overlap.")
        if debug : print("DEBUG!!", masked_data, masked_data.sum())
        # fill_value keeps it from using default=0, which messes up medians
        photreg = basic_photstats(masked_data, clip_sigma=clip_sigma, clip_grow=clip_grow, clip_maxiters=clip_maxiters)
        results[label] = photreg
    return(results)

def wrap_simple_region_photometry(imagefile, regionfile, prefix='foo_', drop=['stddev'], override_label=None, clip_sigma=5, clip_grow=0, clip_maxiters=2):
    tmp_results = photometry_loop_regions(imagefile, regionfile, override_label=override_label, clip_sigma=clip_sigma, clip_grow=clip_grow, clip_maxiters=clip_maxiters)
    df_tmp = pandas.DataFrame.from_dict(tmp_results).T
    df_tmp.drop(drop, inplace=True, axis=1)
    newkeys = [prefix + x   for x in df_tmp.keys()]
    df_tmp.columns= newkeys
    return(df_tmp)

def get_median_of_imagefile(imagefile) :
    hdu = fits.open(imagefile)
    median = np.median(hdu[1].data)
    return(median)

