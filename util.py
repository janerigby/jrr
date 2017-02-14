''' General-purpose utilities.  jrigby May 2016'''

import numpy as np
from   re import split, sub
import fileinput
from astropy.stats import sigma_clip, median_absolute_deviation
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

def add_in_quad(array) :
    # result = sqrt(a^2 + b^2 + c^2...). Input is a numpy array
    quad_sum = np.sqrt((array**2).sum())
    return (quad_sum)
    
def mad(data):
    return median_absolute_deviation(data)

def IQR(Series) :
    ''' Compute interquartile range.  Input is pandas data series.  Output is IQR as np.float64'''
    notnull = Series[Series.notnull()]
    return ( (notnull.quantile(0.25) + notnull.quantile(0.75))/2.)

def robust_max(data, sigma=3) :  # maximum value after sigma clipping
    return (  (sigma_clip(data, sigma=sigma)).max())

def robust_mean(data, sigma=2.5) :
    return (  (sigma_clip(data, sigma=sigma)).mean())

def robust_sum(data, sigma=2.5) :
    return (  (sigma_clip(data, sigma=sigma)).sum())

def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

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
    # astronomers often preface column names with "#", which I can't figure out how to ignore in pandas
    # input is a file.  Returns filename of temp file that has first # stripped
    tmp = "/tmp/workaround"
    subprocess.check_output("sed s/\#// < " + infile + "> "+tmp, shell=True)
    return(tmp)
        
## Basic astronomy

def Jy2AB(Jy):   # convert flux density fnu in Janskies to AB magnitude
    AB = -2.5 * np.log10(Jy * 1E-23) - 48.57
    return(AB)

def AB2Jy(AB) :   # convert AB magnitude to flux density in Janskies    
    Jy = 10**(-0.4*(AB + 48.57))/1E-23  
    return(Jy)


