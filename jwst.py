import os, requests, sys, getopt
from tqdm import tqdm
from astropy.io import fits
from astropy.time import Time
from jwst_backgrounds import jbt # Import the background module
from re import split
import pandas


def getwave_for_filter(*args):
    # Retrieve either a specific central wavelength for a filter, or a dict of them
    # just NIRCam so far.  Pivot wavelengths from JDox on 3/2022
    # This should be in the header; asked helpdesk why it isn't
    filter_wave = {'F070W': 0.704, 'F090W': 0.902, 'F115W': 1.154, 'F150W': 1.501, 'F150W2': 1.659, \
               'F200W': 1.989, 'F212N': 2.121, 'F250M': 2.503, 'F277W': 2.762, 'F300M': 2.989, \
               'F322W2': 3.232, 'F356W': 3.568, 'F410M': 4.082, 'F430M': 4.281,  'F444W': 4.408, 'F480M': 4.874}
    if len(args)==0 : return(filter_wave)
    elif len(args)==1 and args[0] in filter_wave.keys() :
        return(filter_wave[args[0]])

def pixscale(*args):
    # Retrieve the pixel scale in arcseconds, for a given detector. Have not added MIRI, NIRSpec yet
    pixscale = {'nrc_sw': 0.031, 'nrc_lw': 0.063, 'niriss': 0.0656} #from JDox
    if len(args)==0 : return(pixscale)
    elif len(args)==1 and args[0] in pixscale.keys() :
        return(pixscale[args[0]])
    else : raise Exception("ERROR: number of arguments must be 0 or 1.")

def cal_to_Jy(fnu_in, detector='nrc_sw'):
    # helper function to convert summed flux density in MJy/SR *Npixels to Jy.
    # This allows user to sum SB in the _CAL images and easily get a flux density in Jy
    megajy_sr_to_Jy_sqarcsec = 2.35044E-5   #1 MJy/sr = 2.35044E-5 Jy/arcsec^2
    area_1pix = pixscale(detector)**2
    fnu_Jy = fnu_in *  megajy_sr_to_Jy_sqarcsec * area_1pix
    return(fnu_Jy)


# KLUDGY INTERFACE TO CALL JWST_BACKGROUNDS.  Should add this functionality to that module, instead
def get_background_for_jwstfile(jwstfile, bkg_file='/tmp/background.txt') :
# wrapper around the wrapper, that opens the JWST file and imports the resulting background
    hdu = fits.open(jwstfile)
    get_background_for_hdu(hdu, bkg_file=bkg_file)
    # open and process the bkg file into dataframe
    bkg_names = ('wave', 'total', 'zody', 'Gal', 'straylight', 'thermal')
    bkg_df = pandas.read_csv(bkg_file, names=bkg_names, delim_whitespace=True, comment='#')
    return(bkg_df)

def get_background_for_hdu(hdu, bkg_file='/tmp/background.txt', debug=False) : # assume jwst datafile is open
# Wrapper that prints jwst_background file for the RA, DEC, date of a jwst hdu 
    RA       = hdu[0].header['TARG_RA']
    DEC      = hdu[0].header['TARG_DEC']
    datetime = hdu[0].header['DATE-BEG']
    dayofyear = int((split(':', Time(datetime).yday)[1]).lstrip('0'))  # cumbersome format for jwst_backgrounds
    # Gotta format day of year so it's int and doesn't have a leading zero
    thisfilt = hdu[0].header['FILTER']
    wave = getwave_for_filter(thisfilt)
    if debug: print("DEBUGGING:", RA, DEC, datetime, dayofyear, thisfilt, wave)
    jbt.get_background(RA, DEC, wave, thisday=dayofyear, thresh=1.1, \
        plot_background=False, plot_bathtub=False, background_file=bkg_file, write_bathtub=True)
    return(0)


# MAST ARCHIVE 

def get_mast_filename(filename, outputdir=None, overwrite=False, progress=False,
        mast_api_token=None, mast_url="https://mast.stsci.edu/api/v0.1/Download/file"):
    """Download the filename, writing to outputdir

    Default outputdir comes from filename; specify '.' to write to current directory.
    Set overwrite=True to overwrite existing output file.  Default is to raise ValueError.
    Set progress=True to show a progress bar.

    Other parameters are less likely to be useful:
    Default mast_api_token comes from MAST_API_TOKEN environment variable.
    mast_url should be correct unless the API changes.
    # Sample usage:
    # get_mast_filename.py jw01410021001_02101_00001_guider1_uncal.fits
    # R. White, 2022 February 15
    """
    if not mast_api_token:
        mast_api_token = os.environ.get('MAST_API_TOKEN')
        if mast_api_token is None:
            raise ValueError("Must define MAST_API_TOKEN env variable or specify mast_api_token parameter")
    assert '/' not in filename, "Filename cannot include directories"
    if outputdir is None:
        outputdir = '_'.join(filename.split('_')[:-1])
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    elif not os.path.isdir(outputdir):
        raise ValueError(f"Output location {outputdir} is not a directory")
    elif not os.access(outputdir, os.W_OK):
        raise ValueError(f"Output directory {outputdir} is not writable")
    outfile = os.path.join(outputdir, filename)

    if (not overwrite) and os.path.exists(outfile):
        raise ValueError(f"{outfile} exists, not overwritten")

    r = requests.get(mast_url, params=dict(uri=f"mast:JWST/product/{filename}"),
                headers=dict(Authorization=f"token {mast_api_token}"), stream=True)
    r.raise_for_status()

    total_size_in_bytes = int(r.headers.get('content-length', 0))
    block_size = 1024000
    if progress:
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        csize = 0
    with open(outfile, 'wb') as fd:
        for data in r.iter_content(chunk_size=block_size):
            fd.write(data)
            if progress:
                # use the size before uncompression
                dsize = r.raw.tell()-csize
                progress_bar.update(dsize)
                csize += dsize
    if progress:
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
