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
from astropy import constants
from scipy.optimize import brentq    # Equation solving
from jrr.spec import rebin_spec_new
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base # Definition of a Lvl3 association file
from jwst.associations import asn_from_list as afl
import json


h = constants.h.cgs.value
c = constants.c.cgs.value
k = constants.k_B.cgs.value

def get_mrieke_fluxcal_aug2022(infile='/Users/jrrigby1/Python/jrr/mrieke_fluxcalib_08252022.txt'):
    # Create a dict of Marcia Rieke's PHOTMJSR flux calibration zeropoints for NIRCam
    df = pandas.read_csv(infile, sep='\s+', comment='#')
    df['detlong'] = 'NRC' + df['detector']
    df['filter_detector'] = df['filter'] + '_' + df['detlong']
    df.drop(['filter', 'detector', 'detlong'], inplace=True, axis=1)
    df.set_index('filter_detector', inplace=True)
    marcia_dict = df.to_dict()['PHOTMJSR']
    return(marcia_dict)

def get_mrieke_fluxcal_sept2022(infile='/Users/jrrigby1/Python/jrr/mrieke_fluxcalib_09202022.txt'):
    # Create a dict of Marcia Rieke's *updated* PHOTMJSR flux calibration zeropoints for NIRCam
    df = pandas.read_csv(infile, sep='\s+', comment='#')
    df['detlong'] = 'NRC' + df['detector']
    df['filter_detector'] = df['filter'] + '_' + df['detlong']
    df.drop(['filter', 'detector', 'detlong'], inplace=True, axis=1)
    df.set_index('filter_detector', inplace=True)
    marcia_dict = df.to_dict()['PHOTMJSR']
    return(marcia_dict)

def get_gbrammer_fluxcal_aug2022(infile='/Users/jrrigby1/Python/jrr/gbrammer_fluxcalib_aug2022.txt'):
    # make a dict of Gabe Brammer's PHOTMJSR flux calibration zeropoints for NIRCam
    df = pandas.read_csv(infile, sep='\s+', comment='#')
    df['filter_detector'] = df['filt'] + '_' + df['det']
    df.drop(['det', 'filt', 'pupil', 'mjsr_0942', 'gbr', 'ila'], inplace=True, axis=1)
    df.set_index('filter_detector', inplace=True)
    gabe_dict = df.to_dict()['mjsr_gbr'] # This is PHOTMJSR with Gabe's modification
    return(gabe_dict)

def get_mboyer_fluxcal_sep202022_fulldf(infile='/Users/jrrigby1/Python/jrr/mboyer_nircam_fluxcal_CRDS_Delivery_Sept29.txt'):
    df = pandas.read_csv(infile, comment='#', sep='\s+')
    df['filter_detector'] = df['Filter'].str.replace('CLEAR+', '', regex=False) + '_' +  df['detector']
    return(df)

def get_mboyer_fluxcal_sep202022_justdict(infile='/Users/jrrigby1/Python/jrr/mboyer_nircam_fluxcal_CRDS_Delivery_Sept29.txt'):
    df = get_mboyer_fluxcal_sep202022_fulldf(infile=infile)
    df.drop(['Filter', 'detector'], inplace=True, axis=1)
    df.set_index('filter_detector', inplace=True)
    martha_dict = df.to_dict()['PHOTMJSR']
    return(martha_dict)


#def get_mboyer_fluxcal_sep202022_fulldf(infile='/Users/jrrigby1/Python/jrr/mbyoyer_P330E_zeropoints_Sept20_2022.txt'):
#    df = pandas.read_csv(infile, comment='#', sep='\s+')
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
    df_filters = pandas.read_csv(filterfile, sep='\s+', comment='#')
    return(df_filters)

def getwave_for_filter(*args):
    # Retrieve either a specific central wavelength for a filter, or a dict of them
    # just NIRCam and MIRI so far.  Pivot wavelengths from JDox on 3/2022
    # This should be in the header; asked helpdesk why it isn't
    filter_wave = {'F070W': 0.704, 'F090W': 0.902, 'F115W': 1.154, 'F140M': 1.405, 'F150W': 1.501, 'F150W2': 1.659, \
               'F162M': 1.627, 'F182M': 1.845, 'F200W': 1.989, 'F210M': 2.096, 'F212N': 2.121, 'F250M': 2.503, \
               'F277W': 2.762, 'F300M': 2.989, 'F322W2': 3.232, 'F335M': 3.362, 'F356W': 3.568, 'F360M': 3.623, \
               'F410M': 4.082, 'F430M': 4.281, 'F444W': 4.408, 'F460M': 4.630, 'F480M': 4.874, \
               'F560W': 5.6, 'F770W': 7.7, 'F1000W': 10.0, 'F1130W': 11.3, 'F1280W': 12.8, 'F1500W': 15.0, \
               'F1800W': 18.0, 'F2100W': 21.0, 'F2550W': 25.5, 'NIS_F200W': 2.0999, \
               'F1065C' : 10.65, 'F1140C' : 11.40, 'F1550C' : 15.50, 'F2300C' : 23.00 
                       }  # last is the MIRI coronagraph
               # CLEAR is a kludge for PEARLS, for NIS
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

def cal_to_Jy(fnusb_in, detector): 
    # helper function to convert summed flux density in MJy/SR *Npixels to Jy.
    # This allows user to sum SB in the _CAL images and easily get a flux density in Jy
    megajy_sr_to_Jy_sqarcsec = 2.35044E-5   #1 MJy/sr = 2.35044E-5 Jy/arcsec^2
    area_1pix = pixscale(detector)**2
    fnu_Jy = fnusb_in *  megajy_sr_to_Jy_sqarcsec * area_1pix
    return(fnu_Jy)

def get_coords_dayofyear_from_jwstfile(jwstfile, verbose=False):
     hdu = fits.open(jwstfile)
     return(get_coords_dayofyear_from_jwst_hdu(hdu))

def date_to_DOY(date) :
    # date in format: '2023-02-21' or '2022-07-01T00:00:00.0' 
    dayofyear = int((split(':', Time(date).yday)[1]).lstrip('0'))
    return(dayofyear)
    
def get_coords_dayofyear_from_jwst_hdu(hdu, verbose=False):
    # Grab RA, DEC, Day of Year from header.  Converts UTC date (in header) to DOY (what JWST Background Tool expects).
    datetime = hdu[0].header['DATE-BEG']
    dayofyear = date_to_DOY(datetime)
    #dayofyear = int((split(':', Time(datetime).yday)[1]).lstrip('0'))  # cumbersome format for jwst_backgrounds
    # Gotta format day of year so it's int and doesn't have a leading zero
    RA       = hdu[0].header['TARG_RA']
    DEC      = hdu[0].header['TARG_DEC']
    if verbose: print("DEBUGGING:", dayofyear)
    return(RA, DEC, dayofyear)

def open_background_file(bkg_file):  # open an output file generated by jwst_background tool
    bkg_names = ('wave', 'total', 'zody', 'Gal', 'straylight', 'thermal')
    bkg_df = pandas.read_csv(bkg_file, names=bkg_names, sep='\s+', comment='#')
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
def plot_expected_bkgs(plotdf, scalestray=1.0, plotlegend=True, plotthermal=False, plot_not_thermal=False, plotall=True, plotsuffix="", \
                           override_color='k', override_width=2.0, override_ls='dashed'):
    wave = np.arange(1,30.,0.1)  
    plotdf['scaledstraylight'] = plotdf.straylight * scalestray
    plt.plot(plotdf.wave, plotdf.total - plotdf.straylight + plotdf.scaledstraylight, lw=override_width, color=override_color, ls=override_ls)
    if plotall:
        plt.plot(plotdf.wave, plotdf.total - plotdf.straylight, color='orange', lw=2, label='Predicted bkg if no stray light', linestyle='dashed')
        plt.plot(plotdf.wave, plotdf.scaledstraylight, color='b', lw=1, label='Predicted stray light only', linestyle='dashed')
    if plotthermal :
        plt.plot(plotdf.wave, plotdf.thermal, color='r', lw=1, label='thermal only', linestyle='dashed')
    if plot_not_thermal :
        plt.plot(plotdf.wave, plotdf.total - plotdf.thermal, color='g', lw=1, label='all but thermal', linestyle='dashed')
    plt.grid()
    if plotlegend: plt.legend()
    return(0)

# pulling these 2 functions out of MIRI_bkg_levels_vX.ipynb

def intermediate_func(x, fnu1, fnu2, nu1, nu2) :
    '''Intermediate step for inferring T,A of BB from 2 flux densities'''
    a_const =  (nu1/nu2)**3 * (fnu2/fnu1)
    return ( np.exp(h*nu1 / k / x) -1.0) / ( np.exp(h*nu2 / k / x) -1.0) - a_const

def BB_given_2fnu_computeT(fnu1, fnu2, wave1=10., wave2=20.) :
    '''Given two fnu values (in MJy/SR) and two wavelengths (in micron), 
    compute the temperature (in K) and area*emissivity (in funky units 
    of MJy/SR/(erg/s/cm^2/Hz/SR)) of a blackbody with that fnu ratio.'''
    nu1 = c / (wave1 * 1E-4)  # units: cm/s /cm = Hz
    nu2 = c / (wave2 * 1E-4)
    T = brentq(intermediate_func, 10., 300., args=(fnu1,fnu2,nu1,nu2), maxiter=10000)
    area = ( np.exp(h*nu2/k/T) - 1.0)* fnu2/ (2.0*h * nu2**3 / c**2)
    return(np.float64(T), area) 

# Load the measurements of the MIRI imager backgrounds and their rate of increase
def open_MIRI_JRsummaryfile(infile):
    df = pandas.read_csv(infile,  comment='#', sep='\s+')
    df['wave'] = df['Filter'].map(getwave_for_filter())    
    black_df       = df.loc[df.symbol == 'MIRI_imager_alltexp']   # Filtered by eclipt lat, Gal lat
    red_df         = df.loc[df.symbol == 'MIRI_imager_longtexp']  # addnl filtering by exposure time
    df_glowsticks  = df.loc[df.symbol == 'MIRI_corona_glowstick'] # MIRI coronagraph in glowstick region
    df_corona      = df.loc[df.symbol == 'MIRI_corona_notglow']   # MIRI coronagraph avoiding the glowsticks
    raw_df         = df.loc[df.symbol == 'MIRI_imager_raw_longtexp']   # same as red_df, but raw flux density rather than subtracted for astrophysical backgrounds
    return(df, black_df, red_df, df_glowsticks, df_corona, raw_df)


#### Pitch and Roll (attitude) of JWST

# Load JWST pitch and roll angles, from telemetry provided by FOT
def jwst_load_pitchroll_database(telemfile=None):
    #Load the prettefied concatenated telemetry file, made by JWST_getroll_pitch.ipynb
    if telemfile == None:
        telemfile = '/Users/jrrigby1/Python/JWST_scienceops/RollPitch/' + 'jwst_sunpitch_sunroll_JR_032025.csv'
    df_pitchroll = pandas.read_csv(telemfile, comment='#', index_col=0)
    format = '%Y:%j:%H:%M:%S.%f'
    dti = pandas.to_datetime(df_pitchroll.index, format=format)  # date/time in a format pandas understands
    df_pitchroll.set_index(dti)
    return(df_pitchroll)

## Interpolate the roll and pitch for one given time
def jwst_lookup_1_pitchandroll(timetofind, df_pitchroll, timecol='decimalyr') :   # time is in decimal years
    right_after = df_pitchroll[timecol].searchsorted(timetofind)
    subset = df_pitchroll.iloc[right_after - 1 : right_after + 1]
    #print(subset.head())
    new_pitch = rebin_spec_new(subset[timecol],  subset.sun_pitch, timetofind)
    new_roll  = rebin_spec_new(subset[timecol],  subset.sun_roll,  timetofind)
    return(float(new_pitch), float(new_roll))

def jwst_lookup_pitchandroll_df(df, df_pitchroll, timecol='time') : # timecol should have time in decimal years
    # look up the pitch and roll for every entry in input dataframe, from database df_pitchroll=jwst_load_pitchroll_database()
    df['temp'] = df[timecol].apply(lambda x: jwst_lookup_1_pitchandroll(x, df_pitchroll))
    df[['sun_pitch', 'sun_roll']] = df['temp'].apply(pandas.Series)
    df.drop(['temp',], axis=1, inplace=True)
    return(0) # acts on df

#  Read a jsonfile of solar activity
def read_1solar_activity_file(filename=None):
    # How to get the solar activity data:
    # Go find this infile:  ISWALayout_July22topresent_from_BobMeloy.json
    # Load it on https://iswa.ccmc.gsfc.nasa.gov/IswaSystemWebApp/
    # Then download the particular metric you want.  Bob recommended 50 MeV and higher
    # For an example, see Solar_storm_example.ipynb
    # pandas.read_json won't read the json file, bc it's too irregular.  Pull out what we need
    ajson = json.load(open(filename))
    #print(ajson.keys())   # Grab metadata
    columns = [ajson['parameters'][0]['name'], ajson['parameters'][1]['name']]
    df = pandas.DataFrame(ajson['data'], columns=columns)
    # Prepare for pandas timeseries analysis later
    df['datetime'] = pandas.to_datetime(df['Time'])
    df.set_index('datetime', inplace=True)
    df.drop('Time', axis=1, inplace=True)
    return(df)

def read_several_solar_activity_files(filenames=None, thedir=None) :
    df_temp = {}
    if not filenames :  filenames = ['P30MeV.json', 'P50MeV.json', 'P100MeV.json']
    if not thedir:        thedir = '/Users/jrrigby1/Python/JWST_scienceops/Radiation_env/'
    for filename in filenames:
        df_temp[filename] = read_1solar_activity_file(thedir + filename)
    df = pandas.concat(df_temp, axis=1)
    df.columns = df.columns.get_level_values(1)
    return(df)



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


            
# BELOW COPIED FROM DAVID LAW'S MIRI MRS NOTEBOOK: 
# https://github.com/STScI-MIRI/MRS-ExampleNB/blob/main/Flight_Notebook1/MRS_FlightNB1.ipynb
# 
# Define a useful function to write out a Lvl3 association file from an input list
# Note that any background exposures have to be of type x1d.
def writel3asn(scifiles, bgfiles, asnfile, prodname):
    # Define the basic association of science files
    asn = afl.asn_from_list(scifiles, rule=DMS_Level3_Base, product_name=prodname)
        
    # Add background files to the association
    if bgfiles:
        nbg=len(bgfiles)
        for ii in range(0,nbg):
            asn['products'][0]['members'].append({'expname': bgfiles[ii], 'exptype': 'background'})
        
    # Write the association to a json file
    _, serialized = asn.dump()
    with open(asnfile, 'w') as outfile:
        outfile.write(serialized)


def extract1D_SB(x1dfile, s2dfile, sourcepix_range=None):
    '''
    Extract spectrum from s2d file, using wavelength solution from x1dfile
    perform local background subtraction if bgpix_range set to something
    extracts spectrum from sourcepix_range. Just sums surface brightness in those pixels
    By Brian Welch 3/2025
    '''
    with fits.open(x1dfile) as xfile:
        #print("DEBUG", x1dfile)
        wl = xfile[1].data["WAVELENGTH"]
    with fits.open(s2dfile) as sfile:
        s2dim = sfile[1].data 
        s2derr = sfile[2].data
        pixar_sr = sfile[1].header["PIXAR_SR"] # assuming this is in MJy/s
        sb_im = s2dim #/ pixar_sr
        sberr_im = s2dim #/ pixar_sr            
        
    if sourcepix_range == None:
        ysize = s2dim.shape[0] # vertical size of slit
        sourcepix_range = (2,ysize-1) # account for nan row at bottom, split rows at top&bottom
    elif sourcepix_range == 'middle':
        ysize = s2dim.shape[0] # vertical size of slit
        midpt = int(np.round(ysize/2))
        sourcepix_range = (midpt, midpt+1) # account for nan row at bottom, split rows at top&bottom
    
    sourcemin, sourcemax = sourcepix_range[0], sourcepix_range[1]
    #s2dim *= pixar_sr
    source_sb = np.nanmedian(sb_im[sourcemin:sourcemax,:],axis=0) # surface brightness in MJy/sr
    #source_flux = sum(s2dim[sourcemin:sourcemax,:]*pixar_sr*(sourcemax-sourcemin)) * 1e6 # convert MJy/sr -> MJy -> Jy
    source_err = np.sqrt(np.nansum(sberr_im[sourcemin:sourcemax,:]**2, axis=0)) 
        
    return source_sb, source_err, wl
