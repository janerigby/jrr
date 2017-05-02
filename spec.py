''' General-purpose functions to convert and deal with spectra.  Nothing
instrument-specific should go here.  jrigby, May 2016'''

from jrr import util
from astropy.wcs import WCS
from astropy.io import fits
from re import sub
import pandas
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from astropy import constants

def fnu2flam(wave, fnu) :
    '''Convert fnu (in erg/s/cm^2/Hz) to flambda (in erg/s/cm^2/A). Assumes wave in Angstroms.'''
    A_c = constants.c.to('cm/s').value
    flam = fnu *  A_c/(wave * 1E-8 ) / wave
    return(flam)

def flam2fnu(wave, flam) :
    '''Convert from flambda (in erg/s/cm^2/A) to fnu (in erg/s/cm^2/Hz).  Assumes wave in Angstroms.'''
    A_c = constants.c.to('cm/s').value
    fnu = flam * (wave * 1E-8 ) * wave / A_c;
    return(fnu)

def convert2restframe(wave, f, f_u, zz, units) :
    if units == 'flam' :
        rest_wave    = wave / (1.0+zz)      # convert to rest frame
        rest_flam    = f    * (1.0+zz)      # deal w bandwith compression
        rest_flam_u  = f_u  * (1.0+zz)
        return(rest_wave, rest_flam, rest_flam_u)
    elif units == 'fnu' :
        rest_wave    = wave / (1.0+zz)      # convert to rest frame
        rest_fnu     = f    / (1.0+zz)      # deal w bandwith compression
        rest_fnu_u   = f_u  / (1.0+zz)
        return(rest_wave, rest_fnu, rest_fnu_u)
    else :
        raise Exception('ERROR: last argument must specify whether  units of f, f_u are fnu or flam.  Must be same units for input and output')

def convert2restframe_df(df, zz, units='fnu', colwave='wave', colf='fnu', colf_u='fnu_u') : # same, for a data fram
    (rest_wave, rest_f, rest_f_u) = convert2restframe(df[colwave], df[colf],  df[colf_u], zz, units)
    df['rest_wave']      = pandas.Series(rest_wave)
    df['rest_' + colf]   = pandas.Series(rest_f)
    df['rest_' + colf_u] = pandas.Series(rest_f_u)
    return(0)  # acts on df
    
def convert2obsframe(rest_wave, rest_f, rest_f_u, zz, units) :
    if units == 'flam' :
        wave    = rest_wave * (1.0+zz)      # convert to rest frame
        flam    = rest_f    / (1.0+zz)      # deal w bandwith compression
        flam_u  = rest_f_u  / (1.0+zz)
        return(wave, flam, flam_u)
    elif units == 'fnu' :
        wave    = rest_wave * (1.0+zz)      # convert to rest frame
        fnu       = rest_f    * (1.0+zz)      # deal w bandwith compression
        fnu_u     = rest_f_u  * (1.0+zz)
        return(wave, fnu, fnu_u)
    else :
        raise Exception('ERROR: last argument must specify whether  units of f, f_u are fnu or flam.  Must be same units for input and output')

def convert2obsframe_df(df, zz, units='fnu', colrw='rest_wave', colrf='rest_fnu', colrf_u='rest_fnu_u') :
    (wave, f, f_u) =  convert2obsframe(df[colrw], df[colf], df[colf_u], zz, units)
    colwave = re.sub('rest_', '', colrw)
    colf    = re.sub('rest_', '', colrf)
    colf_u  = re.sub('rest_', '', colrf_u)     
    df[colwave] = pandas.Series(wave)
    df[colf]    = pandas.Series(f)
    df[colf_u]  = pandas.Series(f_u)
    return(0)  # acts on df
 
        
def calc_EW(flam, flam_u, cont, cont_u, disp, redshift) :   # calculate equivalent width as simple sum over the region of interest.
    # bare-bones simple.
    # Inputs are flambda, uncertainty, continuum, uncertainty, dispersion in Angstroms
    # Sign convention is that negative EW is emission, positive is absorption
    unity = np.zeros( np.shape(flam)) + 1.0
    EW =  np.sum((unity - flam / cont) * disp) / (1. + redshift) 
    EW_u = np.sqrt(np.sum( (flam_u / cont * disp)**2  +  (cont_u * flam / cont**2 * disp)**2 ))  / (1. + redshift)
    return(EW, EW_u)  # return rest-frame equivalent width and uncertainty

def onegaus(x, aa, bb, cc, cont): # Define a Gaussian with linear continuum
    #y = np.zeros_like(x)
    y = (aa * np.exp((x - bb)**2/(-2*cc**2))  + cont)
    return y
    
def fit_quick_gaussian(sp, guess_pars, colwav='wave', colf='flam', zz=0.) : # Gaussian fit to emission line.  Uses Pandas
    # guess_pars are the initial guess at the Gaussian.
    popt, pcov = curve_fit(onegaus, sp[colwav], sp[colf], p0=guess_pars)
    fit = onegaus(sp[colwav], *popt)
    return(popt, fit)
    
def rebin_spec_new(wave, specin, new_wave, fill=np.nan):
    f = interp1d(wave, specin, bounds_error=False, fill_value=fill)  # With these settings, writes NaN to extrapolated regions
    new_spec = f(new_wave)
    return(new_spec)

def velocity_offset(z1, dz1, z2, dz2) :
    ''' Compute the relative velocity offset (in km/s) between two similar redshifts.
    Only valid for small redshift offsets.  Returns velocity offset and uncertainty.
    call as:  (vel_offset, d_vel_offset) = jrr.spectra.velocity_offset(z1,dz1, z2,dz2)'''
    A_c = constants.c.to('km/s').value  # speed of light
    vel_offset  = (z2 - z1) * A_c / z1
    d_vel_offset = np.sqrt(( dz1 * A_c / (1.+z1)**2 )**2 + (dz2 * A_c / (1.+dz1))**2)
    return(vel_offset, d_vel_offset)

def convert_restwave_to_velocity(restwave, line_center) :
    ''' Utility, convert rest-frame wavelength array to velocity array, relative to v=0 at line_center '''
    A_c =  constants.c.to('km/s').value
    vel =  (restwave - line_center)/line_center * A_c      # km/s     
    return(vel) 

def get_waverange_spectrum(sp, wavcol='wave') :
    # Get the first and last wavelengths of input spectrum sp (assumed to be a pandas data frame).
    # Assumes spectrum is already ordered by wavelength.
    return (np.float(sp[0:1][wavcol].values), np.float(sp[-1:][wavcol].values))

def stack_observed_spectra(spectra, do_sum=True, colwav='wave', colf='fnu', colfu='fnu_u', stacklo=False, stackhi=False, disp=False, sigmaclip=3) :
    ''' General-purpose function to stack spectra.
    Rebins wavelength.  Does NOT convert to rest-frame.  
    Input is a dictionary of pandas data frames that contains the spectra.
    Output wavelength array will be taken from first spectrum in list,
    unless overriden by stacklo, stackhi, disp.
    if do_sum, output colf is straight sum and errors summed in quadrature
    if not do_sum, output colf is weighted average and uncertainty in weighted avg
    '''
    if stacklo and stackhi and disp :
        print "Caution: overriding the default wavelength range and dispersion!"
        nbins = int((stackhi - stacklo)/disp)
        stacked = pandas.DataFrame(data = np.linspace(stacklo, stackhi, num=nbins), columns=(colwav,))
    else :
        stacked = pandas.DataFrame(data=spectra[spectra.keys()[0]][colwav])  # Get output wavelength array from first spectrum
    nbins = stacked.shape[0]  #N of pixels
    nfnu    = np.zeros(shape=(len(spectra), nbins))   # temp array that will hold the input spectra
    nfnu_u  = np.zeros(shape=(len(spectra), nbins))
    Nfiles  = np.ones_like(nfnu, dtype=np.int)
    jackknife     = np.zeros(shape=(len(spectra), nbins))
    for ii, spec in enumerate(spectra.itervalues()):   # Load the spectra into big fat arrays.
        nfnu[ii]   = rebin_spec_new(spec[colwav], spec[colf],  stacked[colwav]) # fnu rebinned
        nfnu_u[ii] = rebin_spec_new(spec[colwav], spec[colfu], stacked[colwav])  # uncertainty on above

    if do_sum :  # SUM the spectra
        stacked[colf]    = np.sum(nfnu, axis=0)
        stacked[colfu]   = util.add_in_quad(nfnu_u, axis=0)
        stacked['Nfiles'] = np.sum(Nfiles, axis=0)
        stacked[colf + '_median_xN'] = np.median(nfnu, axis=0) * np.sum(Nfiles, axis=0) 
        
    if not do_sum :  # Compute the weighted average
        weights = nfnu_u ** -2
        (stacked[colf], sumweight)   = np.average(nfnu, axis=0, weights=weights, returned=True) # weighted avg
        stacked[colfu] =  sumweight**-0.5
        nfnu_clip  = sigma_clip(nfnu, sig=sigmaclip, iters=None, axis=0)
        stacked['clipavg'], sumweight2   = np.average(nfnu_clip, axis=0, weights=weights, returned=True)
        stacked['clipavg_u'] = sumweight2**-0.5   
        stacked['median'] = np.median(nfnu, axis=0)
    return(stacked)
    

    
