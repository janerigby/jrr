''' Helper tools for fitting lines and other functions. jrigby feb 2025'''
from __future__ import print_function

import numpy as np
import re
from jrr.jrjwst import getwave_for_filter

# Started as helper functions to fit ramp + sinusoid to JWST background trending data

def straightline(slope, intercept, x):  # obvious but decent coding
    return(slope * x + intercept )  # y = b + mx

def get_uncert_lines_given_dslope(slope, dslope, intercept, xmidpt, ymidpt):
    # For a linear fit with an uncertainty in the slope dslope, and the x,y coords of
    # at the midpoint of the original line, calculate the parameters of the two lines
    # that cross this line at xmidpt, ymidpt, with slopes +dslope and -dslope
    newintercept1 = ymidpt - (slope + dslope) * xmidpt     #b = y - mx
    newintercept2 = ymidpt - (slope - dslope) * xmidpt     #b = y - mx
    return((slope + dslope, newintercept1), (slope - dslope, newintercept2))

def get_covariance(fitter, thefit):
    cov = fitter.fit_info['param_cov']
    return(dict(zip(thefit.param_names, np.diag(cov)**0.5)))

def grab_the_slope(thefit) :
    # deal w fact that there may be different keywords for slope, depending on whether
    # model is a line, or a line + another component
    if   hasattr(thefit, 'slope_0'): slope = thefit.slope_0.value
    elif hasattr(thefit, 'slope'):   slope = thefit.slope.value
    else: raise Exception('Cannot find keys slope_0 or slope in the fit')
    return(slope)

def get_dslope_from_cov(cov): # get uncertainty in slope from covariance matrix
    # Fits to 6616 use key slope, but other fits (with sinusoid) put it in slope_0
        if 'slope_0' in cov.keys() :    return(np.round(cov['slope_0'], 4))
        elif 'slope' in cov.keys() :    return(np.round(cov['slope'], 4))
        else: raise Exception('Cannot find keys slope_0 or slope in covariance matrix')
            
def evaluate_line(thefit, thetime, cov=None, xmidpt=None) :
    if hasattr(thefit, 'slope_0'):
        slope     = thefit.slope_0.value
        intercept = thefit.intercept_0.value
    elif hasattr(thefit, 'slope'):
        slope     = thefit.slope
        intercept = thefit.intercept
    else:  raise Exception('Cannot find keys slope_0 or slope in covariance matrix')
    myline = straightline(slope, intercept, thetime)  # y = mx+b
    if cov == None:   return(myline)
    if cov and xmidpt: 
        ymidpt = straightline(slope, intercept, xmidpt)  # y = mx+b
        dslope = get_dslope_from_cov(cov)  # if passed covariance matrix, use uncert in slope
        ((s1, b1), (s2, b2)) = get_uncert_lines_given_dslope(slope, dslope, intercept, xmidpt, ymidpt)
        myline_psig = straightline(s1, b1, thetime)
        myline_msig = straightline(s2, b2, thetime)
        return(myline, myline_psig, myline_msig)
    else: raise Exception('Either specify both midpoint of x and covariance matrix, or neither')

    
# these metrics are specific to the ramp + sinusoid fitting of JWST background data
def compute_metrics_of_fit(bestfits, cov, aname, filenames, time1, time0, outfile, xmidpt, measured_median):
    for handle in filenames:
        result = compute_one_fit_metric(handle, bestfits, cov, aname, time1, time0, xmidpt, measured_median)
        print(*result, file=outfile, sep='\t')

def compute_one_fit_metric(handle, bestfits, cov, aname, time1, time0, xmidpt, measured_median):
    (filtname, wavelength) = get_filtername_from_filename(handle)
    y0, y0p, y0m   = evaluate_line(bestfits[handle], time0, cov=cov[handle], xmidpt=xmidpt)   # just the ramp component
    y1, y1p, y1m   = evaluate_line(bestfits[handle], time1, cov=cov[handle], xmidpt=xmidpt)
    z0    = np.round(bestfits[handle](time0), 3)    # the best fit value (includes sinusoid)
    z1    = np.round(bestfits[handle](time1), 3)
    if hasattr(bestfits[handle], 'amplitude_1'):
        amp   = np.round(bestfits[handle].amplitude_1.value, 4)
    else: amp = np.nan
    frac_amp       = np.round(amp/y0, 3)
    frac_amp_total = np.round(amp / measured_median[filtname], 3)
    frac_increase  = np.round(((y1 - y0)/y0), 3)
    frac_increase_peryr = np.round(frac_increase / (time1 - time0), 3)
    rawslope = grab_the_slope(bestfits[handle])
    slope = np.round(rawslope, 4)  # calculate slope of line and uncert from covariance matrix    
    dslope = get_dslope_from_cov(cov[handle])
    dfrac_increase_peryr = np.round((dslope / slope * frac_increase_peryr), 3)  # scale from slope uncert
    result = (filtname, aname, np.round(time0, 4), np.round(time1, 4), z0, z1, np.round(y0,4), np.round(y1,4), \
          amp, frac_amp, frac_amp_total, slope, dslope, frac_increase_peryr, \
          dfrac_increase_peryr)
    return(result)

def get_filtername_from_filename(filename) :  # very specific convenience function for compute_one_fit_metric
    if 'bksub' in filename:    filtname   = re.sub('bksub.csv', '', filename)
    else                  :    filtname   = re.sub('.csv',      '', filename)
    wavelength = getwave_for_filter(filtname)
    return((filtname, wavelength))
