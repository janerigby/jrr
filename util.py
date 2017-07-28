''' General-purpose utilities.  jrigby May 2016'''

import numpy as np
from   re import split, sub, search
import fileinput
from astropy.stats import sigma_clip, median_absolute_deviation
from astropy.coordinates import SkyCoord
from astropy import units
import scipy  
import scikits.bootstrap as bootstrap  
import subprocess

#####  Math  #####

def bootstrap_val_confint(df, statfunction, alpha=0.05) :
    '''Calculates the value of statfunction for a dataset df (numpy array or pandas df), and
    also the confidence interval (by default, 5%,95%)
    Returns the value, and the -,+ uncertainties'''
    result = statfunction(df)
    CI = bootstrap.ci(data=df, statfunction=statfunction, alpha=alpha)
    return( result,  CI[0] - result,  CI[1] - result) 

def sigma_adivb(a, siga, b, sigb) :  # find undertainty in f, where f=a/b , and siga, sigb are uncerts in a,b
    return(  np.sqrt( (siga/b)**2 + (sigb * a/b**2)**2)  )

def add_in_quad(array, axis=0) :
    # result = sqrt(a^2 + b^2 + c^2...). Input is a numpy array
    quad_sum = np.sqrt( np.sum((array**2), axis=axis))
    return (quad_sum)
    
def mad(data):
    return median_absolute_deviation(data)

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

#####  Handle files  #####

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
    tmp = "/tmp/header"
    with open(tmp, "w") as myfile:  myfile.write(header_text)
    subprocess.check_output("cat " + tmp + " " + infile + " > " + outfile, shell=True)
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
    for thiscol in df.keys():
        if df[thiscol].dtype == 'O' : print "\n ***WARNING found dtype Object", thiscol, "\n"
    return(0)

#####  Basic Astronomy  #####

def Jy2AB(Jy):   # convert flux density fnu in Janskies to AB magnitude
    AB = -2.5 * np.log10(Jy * 1E-23) - 48.57
    return(AB)

def AB2Jy(AB) :   # convert AB magnitude to flux density in Janskies    
    Jy = 10**(-0.4*(AB + 48.57))/1E-23  
    return(Jy)

#####  Astronomy coordinate systems  #####

def convert_RADEC_segidecimal(RA_segi, DEC_segi) :  # convert RA, DEC in segidecimal to RA, DEC in degrees
    # RA_segi, DEC_segi are *lists*.  ([0., 22, 180], [-10., 10., 10])
    thisradec = SkyCoord(RA_segi, DEC_segi, unit=(units.hourangle, units.deg), frame='icrs')
    return(thisradec.ra.value, thisradec.dec.value)

def convert_RADEC_segidecimal_df(df, colra='RA', coldec='DEC', newRA='RA_deg', newDEC='DEC_deg') :
    #same as above, but act on a pandas dataframe
    (temp1, temp2) = convert_RADEC_segidecimal(df[colra], df[coldec])
    df[newRA]  = temp1
    df[newDEC] = temp2
    return(0)
    
def convert_RADEC_Galactic(RA_deg, DEC_deg) :    # Convert (decimal) RA, DEC to Galactic.  Can be LISTS []
    thisradec = SkyCoord(RA_deg, DEC_deg, unit=(units.deg, units.deg), frame='icrs')
    return(thisradec.galactic.l.value, thisradec.galactic.b.value)

def convert_RADEC_Ecliptic(RA_deg, DEC_deg) :    # Convert (decimal) RA, DEC to Ecliptic. Can be LISTS []
    thisradec = SkyCoord(RA_deg, DEC_deg, unit=(units.deg, units.deg), frame='icrs')
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
