from __future__ import print_function
## Reads the Galactic dust maps of Green, Schlafly, Finkbeiner et al. 2015, ApJ, 810, 25G,
##     http://adsabs.harvard.edu/abs/2015ApJ...810...25G
## Code downloaded from https://gist.github.com/gregreen/20291d238bc398d9df84 on 14 July 2017

from os.path import expanduser
import json, requests
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas

def get_MW_extinctions() :  # helper function to read a list of target, RA, DEC, and calculate MW EBV
    coordfile = expanduser("~") + "/Dropbox/MagE_atlas/Contrib/ESI_spectra/coords_for_EBV.txt"  #ESI and Grism targets
    coords_df = pandas.read_table(coordfile, delim_whitespace=True, comment="#")
    coords_df['EBV'] =  coords_df.apply(calc_EBV_onerow, axis=1)
    coords_df.set_index('target', inplace=True)   
    return(coords_df) # returns a dataframe of MW E(B-V)

def calc_EBV_onerow(row):  # helper function to appl get_MW_EBV to a dataframe
    return get_MW_EBV(row['RA'], row['DEC'])

def get_MW_EBV(RA, DEC) :
    # Janes wrapper.  Get MW reddening E(B-V) from Green et al. 2015, using
    # their API query_argonaut.  Requires RA, DEC to be in DECIMAL.
    # returns E(B-V) from Milky Way
    coords = SkyCoord(ra=RA, dec=DEC, unit=(u.deg, u.deg))
    EBV_Green2015 = query(coords.ra.value, coords.dec.value, coordsys='equ', mode='sfd')  #
    return(list(EBV_Green2015.values())[0]) #  EBV value from MW dust


def query(lon, lat, coordsys='gal', mode='full'):
    '''
    Send a line-of-sight reddening query to the Argonaut web server.
    
    Inputs:
      lon, lat: longitude and latitude, in degrees.
      coordsys: 'gal' for Galactic, 'equ' for Equatorial (J2000).
      mode: 'full', 'lite' or 'sfd'
    
    In 'full' mode, outputs a dictionary containing, among other things:
      'distmod':    The distance moduli that define the distance bins.
      'best':       The best-fit (maximum proability density)
                    line-of-sight reddening, in units of SFD-equivalent
                    E(B-V), to each distance modulus in 'distmod.' See
                    Schlafly & Finkbeiner (2011) for a definition of the
                    reddening vector (use R_V = 3.1).
      'samples':    Samples of the line-of-sight reddening, drawn from
                    the probability density on reddening profiles.
      'success':    1 if the query succeeded, and 0 otherwise.
      'converged':  1 if the line-of-sight reddening fit converged, and
                    0 otherwise.
      'n_stars':    # of stars used to fit the line-of-sight reddening.
      'DM_reliable_min':  Minimum reliable distance modulus in pixel.
      'DM_reliable_max':  Maximum reliable distance modulus in pixel.
    
    Less information is returned in 'lite' mode, while in 'sfd' mode,
    the Schlegel, Finkbeiner & Davis (1998) E(B-V) is returned.
    '''
    
    url = 'http://argonaut.skymaps.info/gal-lb-query-light'
    
    payload = {'mode': mode}
    
    if coordsys.lower() in ['gal', 'g']:
        payload['l'] = lon
        payload['b'] = lat
    elif coordsys.lower() in ['equ', 'e']:
        payload['ra'] = lon
        payload['dec'] = lat
    else:
        raise ValueError("coordsys '{0}' not understood.".format(coordsys))
    
    headers = {'content-type': 'application/json'}
    
    r = requests.post(url, data=json.dumps(payload), headers=headers)
    
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print('Response received from Argonaut:')
        print(r.text)
        raise e
    
    return json.loads(r.text)


