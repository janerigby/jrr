''' General-purpose utilities.  jrigby May 2016'''
from __future__ import print_function

import pandas
import numpy as np
import subprocess
from os.path import exists, basename
from shutil import copy
import fileinput
import operator
from   re import split, sub, search
from astropy.io.fits import getdata
from astropy.io import fits
from astropy.stats import sigma_clip, median_absolute_deviation
from astropy.coordinates import SkyCoord
from astropy import units as u
#from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import Planck18 as cosmo
from astropy.wcs import WCS
import scipy  
import scikits.bootstrap as bootstrap  
import csv

#####  Math  #####

def bootstrap_val_confint(df, statfunction, alpha=0.05) :
    '''Calculates the value of statfunction for a dataset df (numpy array or pandas df), and
    also the confidence interval (by default, 5%,95%)
    Returns the value, and the -,+ uncertainties'''
    result = statfunction(df)
    CI = bootstrap.ci(data=df, statfunction=statfunction, alpha=alpha)
    return( result,  np.abs(CI[0] - result),  CI[1] - result) 

def sigma_atimesb(a, siga, b, sigb) : # find uncertainty in f, where f=a*b, and siga, sigb are uncerts in a,b
    return(np.sqrt( (siga * b)**2 + (sigb * a)**2 ))

def sigma_atimesb_df(df, cola, colsiga, colb, colsigb) : # convenience function for a dataframe
    return(sigma_atimesb(df[cola], df[colsiga], df[colb], df[colsigb]))
    
def sigma_adivb(a, siga, b, sigb) :  # find undertainty in f, where f=a/b , and siga, sigb are uncerts in a,b
    return(  np.sqrt( (siga/b)**2 + (sigb * a/b**2)**2)  )

def sigma_adivb_df(df, cola, colsiga, colb, colsigb) :  # convenience function for a dataframe
    return(sigma_adivb(df[cola], df[colsiga], df[colb], df[colsigb]))

def add_in_quad(array, axis=0) :
    # result = sqrt(a^2 + b^2 + c^2...). Input is a numpy array
    return( np.sqrt( np.sum((array**2), axis=axis)) )

def errorbar_to_log10(x, dx) :
    dp = np.log10(x + dx) - np.log10(x)
    dm = np.log10(x) - np.log10(x - dx)
    return((dm + dp)/2.0)

def convert_linear_to_log(x, dx, y, dy) :   # dx is uncertainty on x
    logx  = np.log10(x)
    logy  = np.log10(y)
    logdx = errorbar_to_log10(x, dx)
    logdy = errorbar_to_log10(y, dy)
    return(logx, logdx, logy, logdy)

def domath_2cols_df(df, new_col, new_ucol, col1, ucol1, col2, ucol2, theop=operator.add, roundflag=False, rounddec=3):
    # Do simple math (add subtract multiply divide) on 2 columns of a dataframe, and propogate errors
    if theop not in [operator.add, operator.sub, operator.mul, operator.truediv] : raise Exception("ERROR: operator not supported", theop)
    df[new_col]   = theop( df[col1],  df[col2])  # Applying the the operator is trivial. Below, handle uncertainties
    if theop in (operator.add, operator.sub) : # if addition or subtraction
        df[new_ucol] = np.round(np.sqrt( df[ucol1]**2 + df[ucol2]**2), 2)
    elif theop == operator.truediv :    df[new_ucol] = sigma_adivb_df(  df, col1, ucol1, col2, ucol2)
    elif theop == operator.mul :        df[new_ucol] = sigma_atimesb_df(df, col1, ucol1, col2, ucol2)
    if roundflag :
        df[new_col]  = (df[new_col]).round(decimals=rounddec)
        df[new_ucol] = (df[new_ucol]).round(decimals=rounddec)
    return(0) # acts on df

def convenience1(df) : # uncertainties get smaller when binning
    return( df.median() / np.sqrt(np.count_nonzero(df)) )  # Still working on this. **

def mad(data):
    return median_absolute_deviation(data)

def mad_nan(x) :  # Median absolute deviation, robust to NaNs
    return(np.nanmedian(np.absolute(x - np.nanmedian(x))))

def IQR(Series) :
    ''' Compute interquartile range.  Input is pandas data series.  Output is IQR as np.float64'''
    notnull = Series[Series.notnull()]
    return ( (notnull.quantile(0.25) + notnull.quantile(0.75))/2.)  # IS THIS RIGHT?  DiSAGREES W pandas.describe()

def robust_max(data, sigma=3) :  # maximum value after sigma clipping
    return (  (sigma_clip(data, sigma=sigma)).max())

def robust_mean(data, sigma=2.5) :
    return (  (sigma_clip(data, sigma=sigma)).mean())

def robust_sum(data, sigma=2.5) :
    return (  (sigma_clip(data, sigma=sigma)).sum())

def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

def mask_nans(data) :  # return a numpy masked array, with nans masked
    return(np.ma.masked_array(data,np.isnan(data)))

# My new NIRCam pipeline reduction is rotated with a lot of zeros on the edge.  Ignore these in computing median
def median_ignore_zero(somearray):
    masked_array = np.ma.masked_equal(somearray, 0)
    return(np.ma.median(masked_array))

def straightline(slope, intercept, x):  # obvious but decent coding
    return(slope * x + intercept )  # y = b + mx

def get_uncert_lines_given_dslope(slope, dslope, intercept, xmidpt, ymidpt):
    # For a linear fit with an uncertainty in the slope dslope, and the x,y coords of
    # at the midpoint of the original line, calculate the parameters of the two lines
    # that cross this line at xmidpt, ymidpt, with slopes +dslope and -dslope
    newintercept1 = ymidpt - (slope + dslope) * xmidpt     #b = y - mx
    newintercept2 = ymidpt - (slope - dslope) * xmidpt     #b = y - mx
    return((slope + dslope, newintercept1), (slope - dslope, newintercept2))



#####  Handle files  #####

def appendfile_if_exists_elseopenit(filename):  # If a file exists, open it as append. If it does not exist, open it for writing
    if exists(filename) :  handle = open(filename, 'a')
    else                :  handle = open(filename, 'w')
    return(handle)

def split_grab(line, place) :
    splitted = line.split()
    return(splitted[place])

def replace_text_in_file(oldtext, newtext, infile, outfile=None):
    ''' Globally replaces oldtext with newtext.  Like sed. If outfile=None, edits file inplace.'''
    data = open(infile, 'r').read()
    new = sub(oldtext, newtext, data)
    if not outfile : outfile = infile
    f2 = open(outfile, 'w')
    f2.write(new)

def strip_pound_before_colnames(infile) :
    ''' Astronomers often preface column names with "#".  I can't figure out
    how to teach pandas to ignore that, so I wrote this pre-processor.
    input is a file.  Returns filename of temp file that has first # stripped'''
    tmp = "/tmp/workaround"
    subprocess.check_output("sed s/\#// < " + infile + "> "+tmp, shell=True)
    return(tmp)

def put_header_on_file(infile, header_text, outfile) :
    ''' Pandas doesn't allow user to add explanatory header
    to output files.  Headers are useful.  So, writing wrapper to add one.'''
    tmp =  '/tmp/header'
    tmp2 = '/tmp/tmpfile'
    with open(tmp, "w") as myfile:  myfile.write(header_text)
    subprocess.check_output("cat " + tmp + " " + infile + " > " + tmp2, shell=True)
    copy(tmp2, outfile)  # This extra step prevents cat from making an infinite loop when infile=outfile
    return(0)

def read_header_from_file(infile, comment="#") :
    ''' Pandas DataFrame.read_table(), read_csv() discards the explanatory header.  Save it.'''
    head = ''
    with open(infile) as origin_file:
        for line in origin_file:
            if bool(search(comment, line)) :
                head += line
    return(head)

def check_df_for_Object(df) :
    for thiscol in list(df.keys()):
        if df[thiscol].dtype == 'O' : print("\n ***WARNING found dtype Object", thiscol, "\n")
    return(0)

#####  Basic Astronomy  #####

def Jy2AB(Jy):   # convert flux density fnu in Janskies to AB magnitude
    AB = -2.5 * np.log10(Jy * 1E-23) - 48.57
    return(AB)

def AB2Jy(AB) :   # convert AB magnitude to flux density in Janskies    
    Jy = 10**(-0.4*(AB + 48.57))/1E-23  
    return(Jy)

def Jy2cgs(Jy) : # convert flux density fnu in Janskies to fnu in erg/s/cm^2/Hz
    return(Jy * 1E-23)

def cgs2Jy(fnu_cgs) : # convert flux density fnu in Janskies to fnu in erg/s/cm^2/Hz
    return(fnu_cgs  /  1E-23)

#### Conversions that are useful to estimate SFRs  From Kennicutt 1998 and supporting
def mAB_to_fnu(m_AB, zz=0, convert_to_rest=False) :
    # Convert AB magnitude to fnu in erg/s/cm^2/Hz.  Does not remove bandwidth compression by default
     fnu = 10**((m_AB + 48.60)/-2.5)
     if convert_to_rest :  fnu /= (1 + zz)    # Remove the bandwidth compression
     return(fnu)

def fnu_to_Lnu(fnu, zz) :  # From observed fnu to rest Fnu.  Remove bandwidth compression
    d_L =  luminosity_distance(zz)
    #Lnu = fnu_obs / (1+z)  * 4 *pi * D_L^2      , where the (1+z) corrects for bandwith compression
    Lnu = fnu  / (1 + zz) * 4. * np.pi * d_L**2 
    return(Lnu)

def mAB_to_Lnu(m_AB, zz) :  # roll up the above 2 for convenience
    fnu = mAB_to_fnu(m_AB, zz)
    return(fnu_to_Lnu(fnu, zz))

def Kennicutt_LUV_to_SFR(LUV) : 
    return (LUV * 1.4E-28 )  # SFR, from Eqn 1 of Kennicutt 1998, in Msol/yr

def Kennicutt_SFR_to_LHa(SFR) :
    return (SFR / 7.9E-42)  # LHa in erg/s, from SFR in Msol/yr.   Eqn 2 of Kennicutt 1998

def Kennicutt_LHa_to_SFR(LHa) :
    return (LHa * 7.9E-42)  # from LHa in erg/s to SFR in Msol/yr.  Eqn 2 of Kennicutt 1998

def Kennicutt_SFR_to_fHa(SFR, zz) :
    LHa = Kennicutt_SFR_to_LHa(SFR)   # this is a flux, redshift invariant
    fHa = LHa / (4. * np.pi * util.luminosity_distance(zz)**2)
    return(fHa)
    

def luminosity_distance(zz) :
    return ( cosmo.luminosity_distance(zz).to(u.cm).value ) # in cm

def  lookback_time(zz) :
    return( cosmo.lookback_time(zz))



#####  Astronomy coordinate systems  #####

def convert_RADEC_segidecimal(RA_segi, DEC_segi) :  # convert RA, DEC in segidecimal to RA, DEC in degrees
    # RA_segi, DEC_segi are *lists*.  ([0., 22, 180], [-10., 10., 10])
    thisradec = SkyCoord(RA_segi, DEC_segi, unit=(u.hourangle, u.deg), frame='icrs')
    return(thisradec.ra.value, thisradec.dec.value)

def convert_RADEC_segidecimal_df(df, colra='RA', coldec='DEC', newRA='RA_deg', newDEC='DEC_deg') :
    #same as above, but act on a pandas dataframe
    (temp1, temp2) = convert_RADEC_segidecimal(df[colra], df[coldec])
    df[newRA]  = temp1
    df[newDEC] = temp2
    return(0)

def offset_coords(sk, delta_RA=0, delta_DEC=0) :
    ''' From Astropy SkyCoords position sk, compute new coordinates after offsets (in arcsec).  Positive deltas move N and E 
    This should be an method astropy SkyCoord, but I can't find one.'''
    # Coords should be in picky Astropy format, for example sk = SkyCoord(RA, DEC, unit='deg') 
    newsk = sk.directional_offset_by(0. *u.deg, delta_DEC * u.arcsec).directional_offset_by(90. *u.deg, delta_RA * u.arcsec)
    return(newsk)

def gethead(imagefile, header_keyword, extension=1) :
    # return a header keyword from a file, similar to wcstools gethead
    hdu = fits.open(imagefile)
    return( hdu[extension].header[header_keyword] )

def imval_at_coords(imagefile, sk) :
    ''' For a given image and coordinates, return value of that image at coordinates, and nearest xy pixel
    Coordinates should be astropy SkyCoord'''
    (image, hdr) = getdata(imagefile, header=True)
    wcs  = WCS(hdr)
    # should generalize this so it can take segidecimal, too
    xy= wcs.world_to_array_index(sk)  # Careful.  This rounds.  Array indexes to feed to the fits image. Numpy indices, NOT FITS!
    imval = image[xy]
    return(imval, xy)

def print_skycoords_decimal(sk) :
    #Astropy Skycoords prints in weird sexadecimal format unless told not to
    return (sk.ra.to_string(unit=u.deg, decimal=True, precision=6) +  ' ' + sk.dec.to_string(unit=u.deg, decimal=True, precision=6, alwayssign=True))

def distancefrom_coords(RA, DEC, RA_0, DEC_0, uRA=u.deg, uDEC=u.deg, uRA0=u.deg, uDEC0=u.deg) : #compute distance from a given position RA_0, DEC_0
    sk0 = SkyCoord(RA_0, DEC_0, unit=(uRA0, uDEC0)) # the coordinates we're comparing to
    sk1 = SkyCoord(RA, DEC, unit=(uRA, uDEC))
    return(sk0.separation(sk1).to(u.arcsec))

def distancefrom_coords_df(df, RA_0, DEC_0, colRA='RA', colDEC='DEC', uRA0=u.deg, uDEC0=u.deg, newcol='offset_arcsec') :  # same but for a dataframe
    df[newcol] = distancefrom_coords(df[colRA], df[colDEC], RA_0, DEC_0, uRA0=uRA0, uDEC0=uDEC0)
    return(0) # acts on df

def distancefrom_coords2_df(df, colRA1='RA1', colDEC1='DEC1', colRA2='RA2', colDEC2='DEC2', uRA=u.deg, uDEC=u.deg, newcol='offset_arcsec') :  # all cords from dataframe
    # Try an apply here

    df[newcol] = df.apply(distance_from_coords, args=())
    
    df[newcol] = distancefrom_coords(df[colRA1], df[colDEC1], df[colRA2], df[colDEC2], uRA0=uRA, uDEC0=uDEC)
    return(0) # acts on df
  
def convert_RADEC_Galactic(RA_deg, DEC_deg) :    # Convert (decimal) RA, DEC to Galactic.  Can be LISTS []
    thisradec = SkyCoord(RA_deg, DEC_deg, unit=(u.deg, u.deg), frame='icrs')
    return(thisradec.galactic.l.value, thisradec.galactic.b.value)

def convert_RADEC_Ecliptic(RA_deg, DEC_deg) :    # Convert (decimal) RA, DEC to Ecliptic. Can be LISTS []
    thisradec = SkyCoord(RA_deg, DEC_deg, unit=(u.deg, u.deg), frame='icrs')
    return(thisradec.barycentrictrueecliptic.lon.value, thisradec.barycentrictrueecliptic.lat.value)

def convert_RADEC_GalEclip_df(df, colra='RA', coldec='DEC') :
    # For a dataframe, compute Galactic & Ecliptic coords from RADEC
    (tempL, tempB) = convert_RADEC_Galactic(df[colra], df[coldec])
    (templon, templat) = convert_RADEC_Ecliptic(df[colra], df[coldec])
    df['Gal_lon'] = tempL
    df['Gal_lat'] = tempB
    df['Ecl_lon'] = templon
    df['Ecl_lat'] = templat
    return(0)  # acts on the dataframe

### Basic pandas list handling

def getval_or_nan(expression, place=0) :   # convenience function, better error handling 
    # Returns value if series is nonzzero, present, else return NaN
    if len(expression) : return expression.values[place]
    else               : return np.nan

def linerat_from_df(df, linename1, linename2, colname='linename', colf='flux', colfu='flux_u') :
    # Given a data frame of fluxes calculate flux ratio.  linename1 may be 'Hbeta'
    flux1  = getval_or_nan(df.loc[df[colname].eq(linename1)][colf])
    flux2  = getval_or_nan(df.loc[df[colname].eq(linename2)][colf])
    dflux1 = getval_or_nan(df.loc[df[colname].eq(linename1)][colfu])
    dflux2 = getval_or_nan(df.loc[df[colname].eq(linename2)][colfu])
    fluxrat  = flux1/flux2
    dfluxrat = sigma_adivb(flux1, dflux1, flux2, dflux2)
    return(flux1, dflux1, flux2, dflux2, fluxrat, dfluxrat)

def make_ds9_regions_file(outfile, df, racol='RA', deccol='DEC', textcol='text', colorcol='color', radiuscol='radius', font=None, roundregions=True, precision=8) :
    # Make a ds9 regions file from a pandas data frame.
    #Input is a dataframe containing RA, DEC, radius, text, and color for each circle to be made.  Precision is number of decimals for RA, DEC
    if font==None:  font="font=\"helvetica 14 normal roman\""
    if roundregions :
        df['ds9'] = 'circle(' + df[racol].round(precision).astype('str') + ',' + df[deccol].round(precision).astype('str') + ',' + df[radiuscol] + '\") # ' + df[colorcol] + ' ' + df[textcol]
    else :
        df['ds9'] = 'circle(' + df[racol].astype('str') + ',' + df[deccol].astype('str') + ',' + df[radiuscol] + '\") # ' + df[colorcol] + ' ' + df[textcol]
        
    header = '# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n'
    df['ds9'].to_csv(outfile, index=False, header=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\", sep='\t')  # Last 4 arguments prevent pandas from escaping commas & quotes
    put_header_on_file(outfile, header, outfile)
    return(0)

def make_ds9_regions_file_xy(outfile, df, xcol='x', ycol='y', rcol='r') :
    header = '# Region file format: DS9\nimage\n'
    df['ds9'] = 'circle(' + (df[xcol]).astype('str') + ',' + df[ycol].astype('str') + ',' + df[rcol].astype('str') + '\") # '
    df['ds9'].to_csv(outfile, index=False, header=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\", sep='\t')  # Last 4 arguments prevent pandas from escaping commas & quotes
    put_header_on_file(outfile, header, outfile)
    return(0)

def df_xy_from_ds9regions(regionsfile, skiprows=3, x='x', y='y', r='r') :
    # Parse a simple ds9 file of x,y,r in IMAGE coords, and convert to dataframe
    df2 = pandas.read_csv(regionsfile, skiprows=skiprows, names=(x, y, r))
    df2[x] = df2[x].str.replace('circle\(','')
    df2[r] = df2[r].str.replace('\)','')
    return(df2)

