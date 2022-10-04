import os, requests, sys, getopt
from tqdm import tqdm
from astropy.io import fits
from astropy.time import Time
from jwst_backgrounds import jbt # Import the background module
from re import split
import matplotlib.pyplot as plt
import pandas
import numpy as np
from scipy.interpolate import interp1d
from astropy.stats import mad_std
import astropy.io.ascii as ascii


def get_mrieke_fluxcal_aug2022(infile='/Users/jrrigby1/Python/jrr/mrieke_fluxcalib_08252022.txt'):
    # Create a dict of Marcia Rieke's PHOTMJSR flux calibration zeropoints for NIRCam
    df = pandas.read_csv(infile, delim_whitespace=True, comment='#')
    df['detlong'] = 'NRC' + df['detector']
    df['filter_detector'] = df['filter'] + '_' + df['detlong']
    df.drop(['filter', 'detector', 'detlong'], inplace=True, axis=1)
    df.set_index('filter_detector', inplace=True)
    marcia_dict = df.to_dict()['PHOTMJSR']
    return(marcia_dict)

def get_mrieke_fluxcal_sept2022(infile='/Users/jrrigby1/Python/jrr/mrieke_fluxcalib_09202022.txt'):
    # Create a dict of Marcia Rieke's *updated* PHOTMJSR flux calibration zeropoints for NIRCam
    df = pandas.read_csv(infile, delim_whitespace=True, comment='#')
    df['detlong'] = 'NRC' + df['detector']
    df['filter_detector'] = df['filter'] + '_' + df['detlong']
    df.drop(['filter', 'detector', 'detlong'], inplace=True, axis=1)
    df.set_index('filter_detector', inplace=True)
    marcia_dict = df.to_dict()['PHOTMJSR']
    return(marcia_dict)

def get_gbrammer_fluxcal_aug2022(infile='/Users/jrrigby1/Python/jrr/gbrammer_fluxcalib_aug2022.txt'):
    # make a dict of Gabe Brammer's PHOTMJSR flux calibration zeropoints for NIRCam
    df = pandas.read_csv(infile, delim_whitespace=True, comment='#')
    df['filter_detector'] = df['filt'] + '_' + df['det']
    df.drop(['det', 'filt', 'pupil', 'mjsr_0942', 'gbr', 'ila'], inplace=True, axis=1)
    df.set_index('filter_detector', inplace=True)
    gabe_dict = df.to_dict()['mjsr_gbr'] # This is PHOTMJSR with Gabe's modification
    return(gabe_dict)

def get_mboyer_fluxcal_sep202022_fulldf(infile='/Users/jrrigby1/Python/jrr/mboyer_nircam_fluxcal_CRDS_Delivery_Sept29.txt'):
    df = pandas.read_csv(infile, comment='#', delim_whitespace=True)
    df['filter_detector'] = df['Filter'].str.replace('CLEAR+', '', regex=False) + '_' +  df['detector']
    return(df)

def get_mboyer_fluxcal_sep202022_justdict(infile='/Users/jrrigby1/Python/jrr/mboyer_nircam_fluxcal_CRDS_Delivery_Sept29.txt'):
    df = get_mboyer_fluxcal_sep202022_fulldf(infile=infile)
    df.drop(['Filter', 'detector'], inplace=True, axis=1)
    df.set_index('filter_detector', inplace=True)
    martha_dict = df.to_dict()['PHOTMJSR']
    return(martha_dict)


#def get_mboyer_fluxcal_sep202022_fulldf(infile='/Users/jrrigby1/Python/jrr/mbyoyer_P330E_zeropoints_Sept20_2022.txt'):
#    df = pandas.read_csv(infile, comment='#', delim_whitespace=True)
#    df['filter_detector'] = df['filter'].str.replace('CLEAR+', '', regex=False) + '_' +  df['detector']
#    return(df)
#
#def get_mboyer_fluxcal_sep202022_justdict(infile='/Users/jrrigby1/Python/jrr/mbyoyer_P330E_zeropoints_Sept20_2022.txt'):
#    df = get_mboyer_fluxcal_sep202022_fulldf(infile=infile)
#    df.drop(['filter', 'detector', 'std'], inplace=True, axis=1)
#    df.set_index('filter_detector', inplace=True)
#    martha_dict = df.to_dict()['PHOTMJSR']
#    return(martha_dict)
#
#def get_mboyer_fluxcal_aug2022_fulldataframe(infile='/Users/jrrigby1/Python/jrr/boyer2022_nircamoffsets.txt'):
#    temp_table = ascii.read(infile) # Read a machine-readable table into Pandas, via astropy.Table. Automatically gets f#ormat right
#    df = temp_table.to_pandas()
#    df['wave'] = df['Filter'].apply(getwave_for_filter)
#    df['filter_detector'] = df['Filter'] + '_' + df['Detector']
#    return(df)
#
#def get_mboyer_fluxcal_aug2022_justdict(infile='/Users/jrrigby1/Python/jrr/boyer2022_nircamoffsets.txt'):
#    df = get_mboyer_fluxcal_aug2022_fulldataframe(infile=infile)
#    df.drop(['Filter', 'Detector', 'Pupil', 'MJSR-0959'], inplace=True, axis=1)
#    df.drop(['MJSR-mr', 'MJSR-gbr', 'MJSR-1D', 'MJSR-LMC', 'wave'], inplace=True, axis=1)
#    df.rename(columns={'MJSR-2D' : 'PHOTMJSR'}, inplace=True)
#    df.set_index('filter_detector', inplace=True)
#    martha_dict = df.to_dict()['PHOTMJSR']
#    return(martha_dict)
    
def estimate_1fnoise_nircam(filename_wpath):
    # This is an experiment.  std1 should be the standard deviation in the medians of each columns, an estimate of the 1/f noise
    hdu = fits.open(filename_wpath)
    f0 = np.median(hdu[1].data, axis=0)
    f1 = np.median(hdu[1].data, axis=1)
    plt.plot(f1, color='red')    # median of each row
    plt.plot(f0, color='green')  # median of each column
    std0 = mad_std(f0)
    std1 = mad_std(f1)
    plt.grid()
    return(f0,f1, std0, std1, hdu)


def filter_properties():  # Functions using this should replace the 2 functions below, eventually.
    filterfile = '/Users/jrrigby1/Python/jrr/jwst_filters.txt'
    df_filters = pandas.read_csv(filterfile, delim_whitespace=True, comment='#')
    return(df_filters)

def getwave_for_filter(*args):
    # Retrieve either a specific central wavelength for a filter, or a dict of them
    # just NIRCam and MIRI so far.  Pivot wavelengths from JDox on 3/2022
    # This should be in the header; asked helpdesk why it isn't
    filter_wave = {'F070W': 0.704, 'F090W': 0.902, 'F115W': 1.154, 'F150W': 1.501, 'F150W2': 1.659, \
               'F200W': 1.989, 'F212N': 2.121, 'F250M': 2.503, 'F277W': 2.762, 'F300M': 2.989, \
               'F322W2': 3.232, 'F356W': 3.568, 'F410M': 4.082, 'F430M': 4.281,  'F444W': 4.408, 'F480M': 4.874, \
               'F560W': 5.6, 'F770W': 7.7, 'F1000W': 10.0, 'F1130W': 11.3, 'F1280W': 12.8, 'F1500W': 15.0, \
               'F1800W': 18.0, 'F2100W': 21.0, 'F2550W': 25.5 }
    if len(args)==0 : return(filter_wave)
    elif len(args)==1 and args[0] in filter_wave.keys() :
        return(filter_wave[args[0]])

def getwidth_for_filter(*args):
    # Retrieve either a specific central wavelength for a filter, or a dict of them
    # just MIRI so far.  Pivot wavelengths from JDox on 3/2022
    # This should be in the header; asked helpdesk why it isn't
    df_filters = filter_properties()
    filter_width = df_filters.set_index('filtname')['width'].to_dict() 
    if len(args)==0 : return(filter_width)
    elif len(args)==1 and args[0] in filter_width.keys() :
        return(filter_width[args[0]]) 
    
def pixscale(*args):
    # Retrieve the pixel scale in arcseconds, for a given detector. Have not added NIRSpec yet
    pixscale = {'nrc_sw': 0.031, 'nrc_lw': 0.063, 'niriss': 0.0656, 'fgs': 0.0656, 'miri_imager':0.11} #from JDox
    if len(args)==0 : return(pixscale)
    elif len(args)==1 and args[0] in pixscale.keys() :
        return(pixscale[args[0]])
    else : raise Exception("ERROR: number of arguments must be 0 or 1.")

def cal_to_Jy(fnu_in, detector):  # no longer assuming a default, that's not safe
    # helper function to convert summed flux density in MJy/SR *Npixels to Jy.
    # This allows user to sum SB in the _CAL images and easily get a flux density in Jy
    megajy_sr_to_Jy_sqarcsec = 2.35044E-5   #1 MJy/sr = 2.35044E-5 Jy/arcsec^2
    area_1pix = pixscale(detector)**2
    fnu_Jy = fnu_in *  megajy_sr_to_Jy_sqarcsec * area_1pix
    return(fnu_Jy)

def get_coords_dayofyear_from_jwstfile(jwstfile, verbose=False):
     hdu = fits.open(jwstfile)
     return(get_coords_dayofyear_from_jwst_hdu(hdu))
    
def get_coords_dayofyear_from_jwst_hdu(hdu, verbose=False):
    # Grab RA, DEC, Day of Year from header.  Converts UTC date (in header) to DOY (what JWST Background Tool expects).
    datetime = hdu[0].header['DATE-BEG']
    dayofyear = int((split(':', Time(datetime).yday)[1]).lstrip('0'))  # cumbersome format for jwst_backgrounds
    # Gotta format day of year so it's int and doesn't have a leading zero
    RA       = hdu[0].header['TARG_RA']
    DEC      = hdu[0].header['TARG_DEC']
    if verbose: print("DEBUGGING:", dayofyear)
    return(RA, DEC, dayofyear)

def open_background_file(bkg_file):  # open an output file generated by jwst_background tool
    bkg_names = ('wave', 'total', 'zody', 'Gal', 'straylight', 'thermal')
    bkg_df = pandas.read_csv(bkg_file, names=bkg_names, delim_whitespace=True, comment='#')
    return(bkg_df)

# KLUDGY INTERFACE TO CALL JWST_BACKGROUNDS.  Should add this functionality to that module, instead
def get_background_for_jwstfile(jwstfile, bkg_file='/tmp/background.txt', plot_background=False, plot_bathtub=False, wave=2.0, showsubbkgs=False, debug=False) :
# wrapper around the wrapper, that opens the JWST file and imports the resulting background
    hdu = fits.open(jwstfile)
    get_background_for_hdu(hdu, bkg_file=bkg_file, wave=wave, plot_background=plot_background, plot_bathtub=plot_bathtub, debug=debug)
    # open and process the bkg file into dataframe
    bkg_df = open_background_file(bkg_file)
    return(bkg_df)

def get_background_for_hdu(hdu, bkg_file='/tmp/background.txt', wave=2.0, debug=False, plot_background=False, plot_bathtub=False, showsubbkgs=False) : # assume jwst datafile is open
# Wrapper that prints jwst_background file for the RA, DEC, date of a jwst hdu 
    #RA       = hdu[0].header['TARG_RA']
    #DEC      = hdu[0].header['TARG_DEC']
    (RA, DEC, dayofyear) =  get_coords_dayofyear_from_jwst_hdu(hdu)
    datetime = hdu[0].header['DATE-BEG']
#    dayofyear = int((split(':', Time(datetime).yday)[1]).lstrip('0'))  # cumbersome format for jwst_backgrounds
#    # Gotta format day of year so it's int and doesn't have a leading zero
#    #thisfilt = hdu[0].header['FILTER']
#    #wave = getwave_for_filter(thisfilt)  # wave is only used for the bathtub plot; I can't figure out how to get it from header
    if debug: print("DEBUGGING:", RA, DEC, datetime, dayofyear)
    jbt.get_background(RA, DEC, wave, thisday=dayofyear, thresh=1.1, \
        plot_background=plot_background, plot_bathtub=plot_bathtub, background_file=bkg_file, write_bathtub=True, showsubbkgs=showsubbkgs)
    return(0)

# Interpolate the background.  Given a JWST background file, linearly interpolate the values at requested wavelength
def interpolate_bkg_try2(bkg_df, wave_to_find):  #wavelength in micron
    result_dict = {}
    fill=np.nan
    for key in bkg_df.keys() :
        f = interp1d(bkg_df['wave'], bkg_df[key], bounds_error=False, fill_value=fill)  # With these settings, writes NaN to extrapolated regions
        result_dict[key] = float(f(wave_to_find))
    return(result_dict)


# Convenience functions for plotting stray light and other backgrounds
def plot_expected_bkgs(plotdf, scalestray=1.0, plotlegend=True, plotthermal=False, plot_not_thermal=False, plotall=True, plotsuffix=""):
    wave = np.arange(1,30.,0.1)  
    plotdf['scaledstraylight'] = plotdf.straylight * scalestray
    plt.plot(plotdf.wave, plotdf.total - plotdf.straylight + plotdf.scaledstraylight, lw=2, color='k', ls='dashed')
    if plotall:
        plt.plot(plotdf.wave, plotdf.total - plotdf.straylight, color='y', lw=1, label='Predicted bkg if no stray light', linestyle='dashed')
        plt.plot(plotdf.wave, plotdf.scaledstraylight, color='b', lw=1, label='Predicted stray light only', linestyle='dashed')
    if plotthermal :
        plt.plot(plotdf.wave, plotdf.thermal, color='r', lw=1, label='thermal only', linestyle='dashed')
    if plot_not_thermal :
        plt.plot(plotdf.wave, plotdf.total - plotdf.thermal, color='g', lw=1, label='all but thermal', linestyle='dashed')
    plt.grid()
    if plotlegend: plt.legend()
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
