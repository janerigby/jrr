''' General-purpose functions to convert and deal with spectra.  Nothing
instrument-specific should go here.  jrigby, May 2016'''

from astropy.wcs import WCS
from astropy.io import fits
from re import sub
import pandas
import numpy as np
from   scipy.interpolate import interp1d

A_c=2.997925e10   # cm/s

    
def fnu2flam(wave, fnu) :
    '''Convert fnu (in erg/s/cm^2/Hz) to flambda (in erg/s/cm^2/A). Assumes wave in Angstroms.'''
    flam = fnu *  A_c/(wave * 1E-8 ) / wave
    return(flam)

def flam2fnu(wave, flam) :
    '''Convert from flambda (in erg/s/cm^2/A) to fnu (in erg/s/cm^2/Hz).  Assumes wave in Angstroms.'''
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
    
def calc_EW(flam, flam_u, cont, cont_u, disp, redshift) :   # calculate equivalent width as simple sum over the region of interest.
    # bare-bones simple.
    # Inputs are flambda, uncertainty, continuum, uncertainty, dispersion in Angstroms
    # Sign convention is that negative EW is emission, positive is absorption
    unity = np.zeros( np.shape(flam)) + 1.0
    EW =  np.sum((unity - flam / cont) * disp) / (1. + redshift) 
    EW_u = np.sqrt(np.sum( (flam_u / cont * disp)**2  +  (cont_u * flam / cont**2 * disp)**2 ))  / (1. + redshift)
    return(EW, EW_u)  # return rest-frame equivalent width and uncertainty

def rebin_spec_new(wave, specin, new_wave, fill=np.nan):
    f = interp1d(wave, specin, bounds_error=False, fill_value=fill)  # With these settings, writes NaN to extrapolated regions
    new_spec = f(new_wave)
    return(new_spec)

def velocity_offset(z1, dz1, z2, dz2) :
    ''' Compute the relative velocity offset between two similar redshifts.
    Only valid for small redshift offsets.  Returns velocity offset and uncertainty.
    call as:  (vel_offset, d_vel_offset) = jrr.spectra.velocity_offset(z1,dz1, z2,dz2)'''
    const = A_c / 1E5
    vel_offset  = (z2 - z1) * const / z1
    d_vel_offset = np.sqrt(( dz1 * const / (1.+z1)**2 )**2 + (dz2 * const / (1.+dz1))**2)
    return(vel_offset, d_vel_offset)

def convert_restwave_to_velocity(restwave, line_center) :
    ''' Utility, convert rest-frame wavelength array to velocity array, relative to v=0 at line_center '''
    vel =  (restwave - line_center)/line_center * A_c/1E5     # km/s     
    return(vel) 

def iraf_1D_spectrum_to_pandas(infile, uncert_file=None, outfile=None) :
    ''' Takes a 1D spectrum with the wavelength stuck in the WCS header,
    gets the wavelength array out, and packages the spectrum into a nice 
    pandas data frame.  Returns DF, and dumps a pickle file too.
    This was written to reach Chuck Steidel's stacked spectrum, and may
    not yet be general enough, for example the wcs.dropaxis munge.'''
    sp = fits.open(infile)
    header = sp[0].header
    wcs = WCS(header)
    wcs2 = wcs.dropaxis(1)  # Kill the WCS's dummy 2nd dimension
    index = np.arange(header['NAXIS1'])
    temp =  (np.array(wcs2.wcs_pix2world(index, 0))).T
    wavelength = (10**temp)[:,0]
    fnu = sp[0].data
    if uncert_file :
        sp2 = fits.open(uncert_file)      # Get the uncertainty
        fnu_u = sp2[0].data
    else : fnu_u = np.zeros_like(fnu)

    # Make a pandas data frame
    foo = np.array((wavelength, fnu, fnu_u))
    df = pandas.DataFrame(foo.T, columns=("wave", "fnu", "fnu_u"))
    if not outfile :
        outfile = sub(".fits", ".p", infile)
    df.to_pickle(outfile)
    return(df)
