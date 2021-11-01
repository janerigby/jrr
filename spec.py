''' General-purpose functions to convert and deal with spectra.  Nothing
instrument-specific should go here.  jrigby, May 2016'''
from __future__ import print_function

#from builtins import range
import operator  # Needed to get absorption/emission signs right for find_edges_of_line()
import warnings
from jrr import util, peakdet
from astropy.wcs import WCS
from astropy.io import fits
from re import sub
import pandas
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
import astropy.convolution
from astropy import constants
from astropy import units 
from astropy.stats import gaussian_fwhm_to_sigma 
import extinction
from matplotlib import pyplot as plt
from scipy import asarray as ar,exp


def calz_unred(wave, flux, ebv, R_V=4.05):    # pythonified version of idlutil calz_unred
    # rest wave units are Angstroms.  ebv is for the stars, following note in idutil calz_unred, which implements Calzetti 2000 attenuation
    w1 = np.where((wave >= 6300.) * (wave <= 22000.))[0]
    w2 = np.where((wave >= 912.) * (wave < 6300.))[0]
    x = 10000.0/wave # Wavelength in inverse microns
    print("DEBUGGING", len(w1), len(w2), len(wave), len(w1)+len(w2))
    if len(w1) + len(w2) != len(wave):
        warnings.warn('Warning - some elements of wavelength vector outside valid domain')
        print('Warning - some elements of wavelength vector outside valid domain')
    flux[(wave < 912) + (wave > 22000)] = np.NaN
    klam = flux*0.0
    if len(w1) > 0:   klam[w1] = 2.659*(-1.857 + 1.040*x[w1]) + R_V
    if len(w2) > 0:   klam[w2] = 2.659 * np.poly1d([-2.156, 1.509, -0.198, 0.011][::-1])(x[w2]) + R_V
    funred = flux * 10.0**(0.4 * klam * ebv)
    return(funred)

def calz_unred_df(sp, ebv, R_V=4.05, colwave='rest_wave', colf='rest_fnu', coldered='') :  #pandas wrapper for calz_unred
    funred = calz_unred(sp[colwave], sp[colf], ebv, R_V)
    if len(coldered)==0 : coldered=colf + "_dered"
    sp[coldered] = funred
    return(0)
    

def calc_dispersion(sp, colwave='wave', coldisp='disp') :
    '''Calculate the dispersion for a pandas dataframe
    Inputs: spectrum dataframe, column to write dispersion, column to read wavelength'''
    sp[coldisp] = sp[colwave].diff().abs()  # dispersion, in Angstroms
    sp[coldisp].iloc[0] = sp[coldisp][1] # first value will be nan
    return(0)

def boxcar_smooth(sp, win=21, colwave='wave', colf='flam', outcol='flam_smooth', func='median') :
    # Applies a boxcar window, and then takes the median within the boxcar.  May want to generalize to other funcs
    # Calculates a value for every pixel; does not boxcar
    if func == 'median' :   sp[outcol] =  sp[colf].rolling(window=win, center=True).median()
    elif func == 'mean' :   sp[outcol] =  sp[colf].rolling(window=win, center=True).mean()
    elif func == 'sum'  :   sp[outcol] =  sp[colf].rolling(window=win, center=True).sum()
    else : raise Exception("ERROR in boxcar_smooth: func not recognized (should be median, mean, or sum")
    return(0)

def bin_boxcar(df, bins, bincol='wave', fcol='flam', func='mean') :  #Makes a new df, bc bining made it smaller
    df['binned'] = pandas.cut(df[bincol], bins)
    if   func=='mean'   : newdf = df.groupby(['binned'],)[bincol, fcol].mean()
    elif func=='median' : newdf = df.groupby(['binned'],)[bincol, fcol].median()
    # Should probably treat the uncertainty column too, but I haven't done that yet.  
    return(newdf)

def bin_boxcar_better(df, bins, how_to_comb, bincol='wave') :
    # Bin the data, binning bincol by bins, and then doing math to the other columns following the how_to_comb dict
    # https://stackoverflow.com/questions/33217702/groupby-in-pandas-with-different-functions-for-different-columns
    newdf = df.groupby(pandas.cut(df[bincol], bins)).agg(how_to_comb)  # This seems to be onto something...
    # where how_to_comb = {'flam' : 'mean', 'flam_u' : 'sum', 'wave' : 'mean'}
    newdf.set_index(bincol, inplace=True, drop=False)
    return(newdf)

def find_edges_of_line(df, colwave, colf, colcont, Nover_blue, Nover_red, linecen, Nredmax=600, isabs=True) :
    # Find the edges of an absorption line, when it crosses the continuumm for the Nover-th time.
    # INPUTS:  df, pandas data frame containing the spectrum
    # colwave, colf, colcont:  names of columns that contain the wavelength, flux/fnu/flam, and continuum
    # Nover _blue/_red: Use the Nth pixel that crosses the continuum as the edge on blue/red side. Sensible values are 1st or 2nd
    # isabs (Optional): Is this an absorption line? If false, emission line
    # OUTPUTS:
    # (blue_edge, red_edge) are wavelengths of the extent of the line.  Blue_edge is also vmax as I have defined it in stacked paper
    A_c = constants.c.to('km/s').value  # speed of light
    if not test_wave_in_spectrum(df, linecen, colwave) : # if line not covered by spectrum
        print("WARNING:", linecen, "is outside spectrum range", get_waverange_spectrum(df, colwave))
        return(np.nan, np.nan)
    if isabs: comp = operator.gt  # absorption line, edge of line is where flux exceeds continuum
    else :    comp = operator.lt  # emission line,   edge of line is where flux drops below continuum
    blue_edge = df.loc[ comp((df[colf] - df[colcont]), 0) & (df[colwave] < linecen)].iloc[-1*Nover_blue][colwave]  # blueside
    red_edge  = df.loc[ comp((df[colf] - df[colcont]), 0) & (df[colwave] > linecen)].iloc[Nover_red - 1][colwave]  # blueside
    if convert_restwave_to_velocity(red_edge, linecen)  > Nredmax :  # If redmax exceeded the boundary
        red_edge = convert_velocity_to_restwave(Nredmax, linecen)
    return(blue_edge, red_edge)

def calc_vmean_vmax(df, colrwave, colf, colcont, Nover_blue, Nover_red, linecen, Nredmax=600, scalecont=1.0, isabs=True, plotit=False, label=False) :
    # Calculate the absorption-weighted mean velocity, and the max velocities, of a feature.  Default is an absorption line
    # INPUTS:  df, pandas data frame containing the spectrum.  Should have wavelength in REST frame
    # colrwave, colf, colcont:  names of columns that contain the REST wavelength, flux/fnu/flam, and continuum
    # Nover _blue, _red: Use the Nth pixel that crosses the continuum as the edge on the blue/red side. Sensible values are 1st or 2nd
    # Nredmax: Don't allow red edge to be >500km/s from center.  
    # scalecont: scale the continuum by a small amt?  Can see effect of uncertaint continuum
    # isabs (Optional): Is this an absorption line? If false, emission line
    # OUTPUTS:
    if not test_wave_in_spectrum(df, linecen, colrwave) : # if line not covered by spectrum
        print("WARNING:", linecen, "is outside spectrum range", get_waverange_spectrum(df, colrwave))
        return(np.nan, np.nan, np.nan, np.nan)
    df['vel'] = convert_restwave_to_velocity(df[colrwave], linecen)
    df['tempcont']  = df[colcont] * scalecont  # temporary scaling of continuum.  Can be used to gauge effect of cont. uncertainty
    (blue_edge, red_edge) = find_edges_of_line(df, colrwave, colf, 'tempcont', Nover_blue, Nover_red, linecen, Nredmax=Nredmax, isabs=isabs)
#    print "DEBUGGING", blue_edge, red_edge
    subset = df.loc[df[colrwave].between(blue_edge, red_edge)][1:-1]  # The subset of the spectrum between the edges.  Drop the edge pixels, since MgII emission caused trouble
    subset['dv']   = subset['vel'].diff()
    subset['fa'] = (subset['tempcont'] - subset[colf]) / np.sum((subset['tempcont'] - subset[colf]) * subset['dv'])
    vmean = np.sum((subset['vel'] * subset['fa'] * subset['dv']))
    vmax_blue = convert_restwave_to_velocity(blue_edge, linecen)
    vmax_red = convert_restwave_to_velocity(red_edge, linecen)
    if plotit:
        plt.plot(df['vel'], df[colf], color='b', drawstyle="steps-post")
        plt.plot(df['vel'], df['tempcont'], color='g',  drawstyle="steps-post")
        plt.xlim(-3000, 1000)
        plt.ylim(-0.1,1.3)
        plt.scatter((vmean, vmax_blue, vmax_red), (subset[colf].min(), subset['tempcont'].median(), subset['tempcont'].median()), color='red', s=40)
        plt.xlabel("velocity")
        plt.ylabel("flux")
        if label:  plt.annotate(label, xy=(0.8,0.2), xycoords='axes fraction', fontsize=12)
    return(vmean, vmax_blue, vmax_red)

def make_wavearray_constant_resoln(wavelo, wavehi, R, p=2, asSeries=False) :
    # Following Ayan's math, make a wavelength array with
    # Inputs:
    # wavelo, wavehi, lowest and highest wavelength in array
    # R, desired resolution.  R === lambda/dlambda_fwhm
    # p, number of pixels per resolution element.  Default is 2.
    # If asSeries, return Pandas Series. Else, return np array
    nmax = int(math.ceil( math.log10(float(wavehi)/float(wavelo)) / math.log10(1. + 1./float(R*p))))  + 1
    index = np.arange(nmax)
    wave = wavelo*(1. + 1./float(R*p))**index
    if asSeries: return pandas.Series(wave)
    else :       return(wave)

def freq2wave(freq, units='micron/s') : # convert frequency in Hz to wavelength, by default in micron
     A_c = constants.c.to(units).value
     return ( A_c / freq)
 
def wave2freq(wave, units='micron/s') : # convert wavelength (default in micron) to frequency in Hz
     A_c = constants.c.to(units).value
     return ( A_c / wave)

def fnu2flam(wave, fnu) :
    '''Convert fnu (in erg/s/cm^2/Hz) to flambda (in erg/s/cm^2/A). Assumes wave in Angstroms.'''
    A_c = constants.c.to('cm/s').value
    flam = fnu *  A_c/(wave * 1E-8 ) / wave
    return(flam)

def fnu2flam_df(df, colwave='wave', colf='fnu', colf_u='fnu_u') : # same, for a data frame
    df['flam']   = pandas.Series(fnu2flam(df[colwave], df[colf]))
    df['flam_u'] = pandas.Series(fnu2flam(df[colwave], df[colf_u]))
    return(0)  # acts on df

def flam2fnu(wave, flam) :
    '''Convert from flambda (in erg/s/cm^2/A) to fnu (in erg/s/cm^2/Hz).  Assumes wave in Angstroms.'''
    A_c = constants.c.to('cm/s').value
    fnu = flam * (wave * 1E-8 ) * wave / A_c;
    return(fnu)

def flam2fnu_df(df, colwave='wave', colf='flam', colf_u='flam_u') : # same, for a data frame
    df['fnu']   = pandas.Series(flam2fnu(df[colwave], df[colf]))
    df['fnu_u'] = pandas.Series(flam2fnu(df[colwave], df[colf_u]))
    return(0)  # acts on df

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
    (wave, f, f_u) =  convert2obsframe(df[colrw], df[colrf], df[colrf_u], zz, units)
    df[sub('rest_', '', colrw)]   = pandas.Series(wave)
    df[sub('rest_', '', colrf)]   = pandas.Series(f)
    df[sub('rest_', '', colrf_u)] = pandas.Series(f_u)
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

def fit_quick_gaussian(sp, guess_pars, colwave='wave', colf='flam', zz=0., method='least_squares', sigma=None) : # Gaussian fit to emission line.  Uses Pandas
    # guess_pars are the initial guess at the Gaussian.
    if colwave == "index" : xvar = sp.index
    else                  : xvar = sp[colwave]
    popt, pcov = curve_fit(onegaus, xvar, sp[colf], p0=guess_pars, method=method, sigma=sigma)
    fit = onegaus(xvar, *popt)
    return(popt, fit)

def fit_gaussian_fixedcont(sp, guess_pars, contlevel=0.0, colwave='wave', colf='flam', zz=0., method='lm', sigma=None) : # Gaussian fit to emission line, continuum fixed.
    popt, pcov = curve_fit((lambda x, aa, bb, cc: onegaus(x, aa, bb, cc, contlevel)), sp[colwave], sp[colf], p0=guess_pars, method=method, sigma=sigma)
    popt = np.append(popt, contlevel)
    fit = onegaus(sp[colwave], *popt)
    return(popt, fit)

def sum_of_gaussian(gauss_pars) :  # Quick sum of a guassian.  Does not include continuum flux
    (aa, bb, cc, cont) = gauss_pars
    return( aa * cc * np.sqrt(2.0 * np.pi))

def rebin_spec_new(wave, specin, new_wave, fill=np.nan, return_masked=False):
    # Rebin spectra (wave, specin) to new wavelength aray new_wave.  Fill values are nan.  If return_masked, then mask the nans
    f = interp1d(wave, specin, bounds_error=False, fill_value=fill)  # With these settings, writes NaN to extrapolated regions
    new_spec = f(new_wave)
    if return_masked :  return(util.mask_nans(new_spec))
    else :              return(new_spec)
    
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

def convert_velocity_to_restwave(velocity, line_center) :  # vel in km/s
    A_c =  constants.c.to('km/s').value
    restwave =  (1.0 + velocity / A_c) * line_center 
    return(restwave)
    
def get_waverange_spectrum(sp, colwave='wave') :
    # Get the first and last wavelengths of input spectrum sp (assumed to be a pandas data frame).
    # Assumes spectrum is already ordered by wavelength.
    return (np.float(sp[0:1][colwave].values), np.float(sp[-1:][colwave].values))

def test_wave_in_spectrum(sp, linecen, colwave='wave') : # Is given wavelength linecen within the wavelength range of spectrum df?
    (lo, hi) = get_waverange_spectrum(sp, colwave=colwave)
    is_within  = linecen > lo and linecen < hi
    return(is_within)

# Below are functions to automatically fit a smooth continuum to a spectrum.  Generalized from jrr.mage

def get_boxcar4autocont(sp, smooth_length=100.) :
    # Helper function for fit_autocont().  For the given input spectrum, finds the number of pixels that
    # corresponds to smooth_length in rest-frame Angstroms.  This will be the boxcar smoothing length.
    # target is in rest-frame Angstroms.  Default is 100 A, which works well for MagE spectra.
    return(np.int(util.round_up_to_odd(smooth_length / sp.rest_disp.median())))  # in pixels

def flag_near_lines(sp, LL, colv2mask='vmask', colwave='wave', colmask='linemask', linetype='all') :
    # Flag regions within +- vmask km/s around lines in linelist LL
    # Inputs:   sp        spectrum as Pandas data frame
    #           LL        linelist as pandas data frame
    #           zz        redshift
    #           colv2mask  column containing velocity around which to mask each line +-, in km/s
    #           colwave   column to find the wavelength in sp
    #           linetype  list of types of lines to mask.  by default, mask all types of lines.  Example: linetype=('INTERVE')
    # Outputs:  None.  It acts directly on sp.linemask
    #print "Flagging regions near lines."
    if linetype == 'all' :  subset = LL
    else :                  subset = LL[LL['type'].isin(linetype)]
    line_lo = np.array(subset['restwav'] * (1. - subset[colv2mask]/2.997E5) * (1. + subset['zz']))
    line_hi = np.array(subset['restwav'] * (1. + subset[colv2mask]/2.997E5) * (1. + subset['zz']))
    temp_wave = np.array(sp[colwave])
    temp_mask = np.zeros_like(temp_wave).astype(np.bool)
    for ii in range(0, len(line_lo)) :    # doing this in observed wavelength
        temp_mask[np.where( (temp_wave > line_lo[ii]) & (temp_wave < line_hi[ii]))] = True
    sp[colmask] = temp_mask  # Using temp numpy arrays is much faster than writing repeatedly to the pandas data frame
    return(0)

# When most of the spectrum is contaminated, and only a few regions have good continuum, define good cont regions directly, rather than by flagging lines
# Nothing uses this yet.  Wrote it for grism, but then decided to fit continuum manually using XIDL's x_continuum 
def flag_good_contregions(sp, good_lo, good_hi, colwave='wave', colmask='contmask') :
    temp_wave = np.array(sp[colwave])
    temp_mask = np.ones_like(temp_wave).astype(np.bool)     # By default, spectrum is flagged as not suitable for fitting continuum
    for ii in range(0, len(good_lo)) :    
        temp_mask[np.where( (temp_wave > good_lo[ii]) & (temp_wave < good_hi[ii]))] = False   # Remove flag. These parts of spectra can be fit w cont
    sp[colmask] = temp_mask  # Using temp numpy arrays is much faster than writing repeatedly to the pandas data frame
    return(0)

def fit_autocont(sp, LL, zz, colv2mask='vmask', boxcar=1001, flag_lines=True, colwave='wave', colf='fnu', colmask='contmask', colcont='fnu_autocont') : 
    ''' Automatically fits a smooth continuum to a spectrum.  Generalized from mage version
     Inputs:  sp,  a Pandas data frame containing the spectra, 
              LL,  a Pandas data frame containing the linelist to mask, opened by mage.get_linelist(linelist) or similar
              zz,  the systemic redshift. Used to set window threshold around lines to mask
              colv2mask,  column containing velocity +- to mask each line,, km/s. 
              boxcar,   size of the boxcar smoothing window, in pixels
              flag_lines (Optional) flag, should the fit mask out known spectral features? Can turn off for debugging
              colwave, colf:  which columns to find wave, flux/flam/fnu
              colmask,  the column that has the mask, of what should be ignored in continuum fitting.  True is masked
              colcont, column to write the continuum.  The output!
    '''
    if flag_lines :
        if colmask not in sp : sp[colmask] = False
        flag_near_lines(sp, LL, colv2mask=colv2mask, colwave=colwave)  # lines are masked in sp.linemask
        sp.loc[sp['linemask'], colmask] = True           # add masked lines to the continuum-fitting mask
    # astropy.convolve barfs on pandas.Series as input.  Use .to_numpy() to send as np array
    smooth1 = astropy.convolution.convolve(sp[colf].to_numpy(), np.ones((boxcar,))/boxcar, boundary='extend', fill_value=np.nan, mask=sp[colmask].to_numpy()) # boxcar smooth
    small_kern = int(util.round_up_to_odd(boxcar/10.))
    smooth2 = astropy.convolution.convolve(smooth1, np.ones((small_kern,))/small_kern, boundary='extend', fill_value=np.nan) # Smooth again, to remove nans
    sp[colcont] = pandas.Series(smooth2)  # Write the smooth continuum back to data frame
    sp[colcont].interpolate(method='linear',axis=0, limit_direction='both', inplace=True)  #replace nans
    #print "DEBUGGING", np.isnan(smooth1).sum(),  np.isnan(smooth2).sum(), sp[colcont].isnull().sum()
    return(smooth1, smooth2) 


## Normalization methods.

def byspline_norm_func(wave, rest_fnu, rest_fnu_u, rest_cont, rest_cont_u, norm_region) :
    # Normalization method by the spline fit continuum
    temp_norm_fnu = rest_fnu / rest_cont
    temp_norm_sig = util.sigma_adivb(rest_fnu, rest_fnu_u,   rest_cont, rest_cont_u) # propogate uncertainty in continuum fit.
    return(temp_norm_fnu, temp_norm_sig)

def norm_by_median(wave, rest_fnu, rest_fnu_u, rest_cont, rest_cont_u, norm_region) :
    '''Normalize by the median within a spectral range norm_region.  Assumes Pandas.'''
    normalization = np.median(rest_fnu[wave.between(*norm_region)])
    #print "normalization was", normalization, type(normalization)
    return(rest_fnu / normalization,  rest_fnu_u / normalization)
            

### Generalized spectral stacking...

def stack_spectra(df, colwave='wave', colf='fnu', colfu='fnu_u', colmask=[], output_wave_array=[], pre='f', sigmaclip=3) :
    ''' General-purpose function to stack spectra.  Rebins wavelength.
    Does not de-redshift spectra.  If you want to stack in rest frame, run jrr.spec.convert2restframe_df(df) beforehand.
    Any normalization by continuum should be done beforehand.
    Input df{} is a dictionary of pandas data frames that contains the spectra.
    colwave, colf, colfu, colmask, tell where to find the columns for wavelength, flux/flam/fnu, uncertainty, & input mask.
    colmask is a column in dataframe of values to mask (True=masked)
    Output wavelength array will be output_wave_array if it is supplied; else will use 1st spectrum in df{}.
    '''
    if len(output_wave_array) :
        print("Caution: overriding the default wavelength range and dispersion!")
        stacked = pandas.DataFrame(data=output_wave_array, columns=(colwave,))
    else :
        stacked = pandas.DataFrame(data=df[list(df.keys())[0]][colwave])  # Get output wavelength array from first spectrum
    nbins = stacked.shape[0]  #N of pixels
    nspectra = len(df)
    nf    =   np.ma.zeros(shape=(nspectra, nbins))   # temp array that will hold the input spectra
    nf_u  =   np.ma.zeros(shape=(nspectra, nbins))   # using numpy masked arrays so can ignore nans from rebin_spec_new
    for ii, spec in enumerate(iter(df.values())):   # Rebin each spectrum (spec), and load all spectra into big fat arrays.
        ma_spec = spec     # masked version of spectrum
        if colmask :
            ma_spec.loc[ma_spec[colmask], colf]  = np.nan    # try setting to nan to solve the gap problem
            #ma_spec.loc[ma_spec[colmask], colfu] = ma_spec[colf].median() * 1E6  # set this huge
        nf[ii]   = np.ma.masked_invalid(rebin_spec_new(ma_spec[colwave], ma_spec[colf],  stacked[colwave], return_masked=True)) # fnu/flam rebinned
        nf_u[ii] = np.ma.masked_invalid(rebin_spec_new(ma_spec[colwave], ma_spec[colfu], stacked[colwave], return_masked=True))  # uncertainty on above
    nf[ :,  0:2].mask = True  # mask the first 2 and last 2 pixels  of each spectrum
    nf[ :, -2:].mask = True
    ## NEED TO HAVE IT USE THE COLMASK, IF DEFINED.
        # Is rebinning handling the uncertainties correctly?
    stacked[pre+'sum']    = np.ma.sum(nf, axis=0)
    stacked[pre+'sum_u']  = util.add_in_quad(nf_u, axis=0)
    stacked[pre+'avg']    = np.ma.average(nf, axis=0)
    stacked[pre+'avg_u']  = stacked[pre+'sum_u'] /  np.ma.MaskedArray.count(nf, axis=0)
    weights = nf_u ** -2              # compute the weighted avg
    (stacked[pre+'weightavg'], sumweight) = np.ma.average(nf, axis=0, weights=weights, returned=True) # weighted avg
    stacked[pre+'weightavg_u'] =  sumweight**-0.5
#    nf_clip  = sigma_clip(nf, sigma=sigmaclip, iters=None, axis=0)
    nf_clip  = sigma_clip(nf, sigma=sigmaclip, axis=0)
    stacked[pre+'clipavg'], sumweight2   = np.ma.average(nf_clip, axis=0, weights=weights, returned=True)
    stacked[pre+'clipavg_u'] = sumweight2**-0.5   
    stacked[pre+'median']   = np.ma.median(nf, axis=0)
    stacked[pre+'medianxN'] = np.ma.median(nf, axis=0) * np.ma.MaskedArray.count(nf, axis=0) 
    stacked['Ngal'] = np.ma.MaskedArray.count(nf, axis=0)  # How many spectra contribute to each wavelength
    
    # compute the jackknife variance.  Adapt from mage_stack_redo.py.
    jackknife= np.ma.zeros(shape=(nspectra, nbins)) # The goal.
    jack_var = np.ma.zeros(shape=nbins)
    for ii in range(0, nspectra) :
        jnf = nf.copy()
        #print "DEBUGGING Jackknife, dropping ", ii,  "from the stack"
        jnf[ii, :].mask = True  # Mask one spectrum
        jackknife[ii], weight = np.ma.average(jnf, axis=0, weights=weights, returned=True)  # all the work is done here.
        jack_var = jack_var +  (jackknife[ii] - stacked[pre+'weightavg'])**2
    jack_var *= ((nspectra -1.0)/float(nspectra))   
    jack_std = np.sqrt(jack_var)
    stacked[pre+'jack_std'] = jack_std
    return(stacked, nf, nf_u)

# NB: Can get MW extinction for a given RA, DEC from jrr.MW_EBV.get_MW_EBV

### Extinction routines below
def deredden_MW_extinction(sp, EBV_MW, colwave='wave', colf='fnu', colfu='fnu_u', colcont='fnu_cont', colcontu='fnu_cont_u', colmed='median') :
    #print "Dereddening Milky Way extinction"
    Rv = 3.1
    Av = -1 * Rv *  EBV_MW  # Want to deredden, so negative sign
    print("jrr.spec.deredden_MW_extinction, applying Av  EBV_MW: ", Av, EBV_MW)
    #sp['oldfnu'] = sp[colf]  # Debugging
    MW_extinction = extinction.ccm89(sp[colwave].astype('float64').to_numpy(), Av, Rv)
    sp['MWredcor'] = 10**(-0.4 * MW_extinction)
    sp[colf]     = pandas.Series(extinction.apply(MW_extinction, sp[colf].astype('float64').to_numpy()))
    sp[colfu]    = pandas.Series(extinction.apply(MW_extinction, sp[colfu].astype('float64').to_numpy()))
    if colcont  in list(sp.keys()) :  sp[colcont] = pandas.Series(extinction.apply(MW_extinction,  sp[colcont].astype('float64').to_numpy()))
    if colcontu in list(sp.keys()) : sp[colcontu] = pandas.Series(extinction.apply(MW_extinction, sp[colcontu].astype('float64').to_numpy()))
    if colmed   in list(sp.keys()) :   sp[colmed] = pandas.Series(extinction.apply(MW_extinction,   sp[colmed].astype('float64').to_numpy()))
    return(0)

def deredden_internal_extinction(sp, this_ebv, colf='rest_fnu', colu="rest_fnu_u", deredden_uncert=True, colwave='rest_wave') :
    # Remove internal extinction as fit by Chisholm's S99 fits.  Assumes Calzetti
    sp_filt = sp.loc[sp[colwave].between(1200,20000)]
    Rv = 4.05  # IS THIS RIGHT FOR STELLAR CONTINUUM?*****
    Av = -1 * Rv * this_ebv  # stupidity w .Series and .to_numpy() is bc extinction package barfs on pandas. pandas->np->pandas
    colfout = colf + "_dered"
    coluout = colu + "_dered"
    sp[colfout]  = pandas.Series(extinction.apply(extinction.calzetti00(sp[colwave].astype('float64').to_numpy(), Av, Rv, unit='aa'), sp[colf].astype('float64').to_numpy()))
    if deredden_uncert :
        sp[coluout]  = pandas.Series(extinction.apply(extinction.calzetti00(sp[colwave].astype('float64').to_numpy(), Av, Rv, unit='aa'), sp[colu].astype('float64').to_numpy()))
    return(0) 

def find_lines_simple(sp, abs=True, wavcol='wave', fcol='fnu', delta=0.15) :
    sp['temp'] = False  # 1st pass, found a peak
    sp['peak'] = False  # 2nd pass, peak is significant
    maxtab, mintab = peakdet.peakdet(sp[fcol], delta)  # Find peaks.
    if abs:
        peak_ind =  [np.int(p[0]) for p in mintab] # The minima, if absorption lines
        peakf    =  [y   for x,y in mintab]
    else:
        peak_ind =  [np.int(p[0]) for p in maxtab] # The maxima, if emission lines
        peakf    =  [y   for x,y in maxtab]
    sp.loc[peak_ind, 'peak'] = True
    peak_waves = sp[wavcol].iloc[peak_ind]
    print("Found this many peaks: ", sp['peak'].sum())
    return(peak_ind, peak_waves, peakf)


def find_lines_Schneider(sp, resoln, siglim=3., abs=True, delta=0.15) :
    # Blind search for absorption lines, following Schneider et al. 1993
    # Delta seems pretty damned arbitary, may bite me later.
    # Significant peaks identified as sp['peak']=True
    calc_schneider_EW(sp, resoln)  # Calculate EW and EW limits  # Pulled this out of ayan's package, so I control it. It's below
    sp['temp'] = False  # 1st pass, found a peak
    sp['peak'] = False  # 2nd pass, peak is significant
    maxtab, mintab = peakdet.peakdet(sp.W_interp,delta)  # Find peaks.
    if abs:   peak_ind =  [np.int(p[0]) for p in mintab] # The minima
    else:     peak_ind =  [np.int(p[0]) for p in maxtab] # The maxima
    sp['temp'].iloc[peak_ind] = True  # Convert back into pandas style
    # Choose only peaks that are greater than siglim significant
    if abs :  # If looking for absorption lines 
        subset = sp['temp']  &  (sp['W_interp'] < sp['W_u_interp'] * siglim * -1)
    else :    # If looking for emission lines
        subset = sp['temp']  &  (sp['W_interp'] > sp['W_u_interp'] * siglim)
    sp['peak'].iloc[subset] = True  # Peaks now marked
    sp.loc[sp['wave'].between(4737.6, 4738), 'peak'] = True  # ADD PEAK FOR S1527 at z=2.055, to work around bad skyline
    print("FINDING PEAKS (this is slow), N peaks: ", sp['temp'].sum(),  "  significant peaks: ", sp['peak'].sum())
    return(0)


def calc_schneider_EW(sp, resoln):
    ''' Function to calculate the limiting equivalent width for detection for every
    pixel in the spectrum, following Schneider et al. 1993, ApJS, 87, 45. Coded by Ayan in ~2017;
    copied over 2020 by jrigby to jrr so no longer dependent on ayan package.
    No output; instead, adds 2 columns to input dataframe sp.  XX describe them.  
    Ayan, could you please document whether EW is in rest-frame or observed frame?  Reading
    the code, I think it is observed frame if observed wavelengths are passed. -Jane. '''
    EW=[] ; sig=[] ; sig_int=[] ; signorm_int=[]
    w = sp.wave.values
    f = (sp.flam/sp.flam_autocont).values
    #--normalised error spectrum for EW limit----
    unorm = (sp.flam_u/sp.flam_autocont).values #normalised flux error
    func_u = interp1d(w, unorm, kind='linear')
    uinorm = func_u(w) #interpolated normalised flux error
    #---------------------------
    disp = np.concatenate(([np.diff(w)[0]],np.diff(w))) #disperion array
    n = len(w)
    lim = 3.
    N = 2.
    for ii in range(len(w)):
        b = w[ii]
        c = w[ii]* gaussian_fwhm_to_sigma  /resoln       
        j0 = int(np.round(N*c/disp[ii]))
        a = 1./np.sum([exp(-((disp[ii]*(j0-j))**2)/(2*c**2)) for j in range(2*j0+1)])
        P = [a*exp(-((disp[ii]*(j0-j))**2)/(2*c**2)) for j in range(2*j0+1)]
        j1 = max(1,j0-ii)
        j2 = min(2*j0, j0+(n-1)-ii)
        #For reference of following equations, please see 1st and 3rd equation of section 6.2 of Schneider et al. 1993.
        #The 2 quantities on the left side of those equations correspond to 'EW' and 'signorm_int' respectively which subsequently become 'W_interp' and 'W_u_interp'
        EW.append(disp[ii]*np.sum(P[j]*(f[ii+j-j0]-1.)for j in range(j1, j2+1))/np.sum(P[j]**2 for j in range(j1, j2+1)))
        signorm_int.append(disp[ii]*np.sqrt(np.sum(P[j]**2*uinorm[ii+j-j0]**2for j in range(j1, j2+1)))/np.sum(P[j]**2 for j in range(j1, j2+1)))
        #sig.append(disp[ii]*np.sqrt(np.sum(P[j]**2*unorm[ii+j-j0]**2for j in range(j1, j2+1)))/np.sum(P[j]**2 for j in range(j1, j2+1)))
    func_ew = interp1d(w, EW, kind='linear')
    W = func_ew(w) #interpolating EW
    #sp['W_interp'] = pd.Series(W) #'W_interp' is the result of interpolation of the weighted rolling average of the EW (based on the SSF chosen)
    #sp['W_u_interp'] = pd.Series(signorm_int) #'W_u_interp' is 1 sigma error in EW derived by weighted rolling average of interpolated flux error.
    sp['W_interp'] = W
    sp['W_u_interp'] = signorm_int
    return(0)
