''' General-purpose utilities.  jrigby May 2016'''

import numpy as np
from   re import split
import fileinput
from astropy.stats import sigma_clip

#####  Math  #####

def sigma_adivb(a, siga, b, sigb) :  # find undertainty in f, where f=a/b , and siga, sigb are uncerts in a,b
    return(  np.sqrt( (siga/b)**2 + (sigb * a/b**2)**2)  )

def mad(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)

def IQR(Series) :
    ''' Compute interquartile range.  Input is pandas data series.  Output is IQR as np.float64'''
    notnull = Series[Series.notnull()]
    return ( (notnull.quantile(0.25) + notnull.quantile(0.75))/2.)

def robust_max(data, sigma=3) :  # maximum value after sigma clipping
    return max(sigma_clip(data, sigma=sigma))

def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

#####  Handle files  #####

def split_grab(line, place) :
    splitted = line.split()
    return(splitted[place])

def replace_text_in_file_like_sed(oldtext, newtext, infile):
    ''' This edits a file in-place, globally replacing oldtext with newtext.
    Should act like this sed equivalent:  sed -i .bak 's/oldtext/newtext/g' infile '''
    for line in fileinput.FileInput(infile,inplace=1):
        line = line.replace(oldtext, newtext)
        print line,
    return(0)

## Basic astronomy

def Jy2AB(Jy):   # convert flux density fnu in Janskies to AB magnitude
    AB = -2.5 * np.log10(Jy * 1E-23) - 48.57
    return(AB)

def AB2Jy(AB) :   # convert AB magnitude to flux density in Janskies    
    Jy = 10**(-0.4*(AB + 48.57))/1E-23  
    return(Jy)


