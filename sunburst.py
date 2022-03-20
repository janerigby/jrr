''' Useful functions from the Sunburst Arc notebooks.  jrigby Dec 2021'''
from re import sub
from jrr import phot
from pandas import DataFrame, concat, Series

def contsub_filename(whichline, convolved=False) :  # return continuum subtracted filename and wht map
    if 'Lya' in whichline: 
        firstpart    = sub('_', '_contsub', whichline)
    else : firstpart = whichline + '_contsub'
    if convolved : firstpart += '_convolved'
    return(firstpart + '/' + firstpart + '.fits',   firstpart + '/' + sub('contsub', 'wht', firstpart) + '.fits')


def setup_contsub_files(convolved=True) :
    if convolved == True :
        fitsdir = "/Users/jrrigby1/SCIENCE/Lensed-LBGs/Planck_Arc/Sunburst_HST_Version4.0/Convolved_to_F160W_v2/"
        prefix= "V4.0_PSZ1G311.65-18.48_"  ; 
        suffix = "_sci_convolved_to_F160Wv4.fits.gz"
        #          F390W, F555W  F128N, F153, F153)  # W in the filter part of filename is Mike's typo
        offband = ("F390W_0.03g0.8_cr1.2_0.7_drc", "F555W_0.03g0.8_cr1.2_0.7_drc", "F128W_0.03g0.8_cr4.0_0.7_drz", \
                   "F153W_0.03g0.8_cr4.0_0.7_drz", "F153W_0.03g0.8_cr4.0_0.7_drz")
        #       F410M   F410M  F126N  F164N   F167N
        onband  = ("F410M_0.03g0.8_cr1.2_0.7_drc", "F410M_0.03g0.8_cr1.2_0.7_drc", "F126W_0.03g0.8_cr4.0_0.7_drz", \
                   "F164W_0.03g0.8_cr4.0_0.7_drz", "F167W_0.03g0.8_cr4.0_0.7_drz")
        outfilename = ("Lya_contsubF390W_convolved", "Lya_contsubF555W_convolved", "OII_contsub_convolved", \
                       "Hbeta_contsub_convolved", "OIII_contsub_convolved")
    else :
        fitsdir = "/Users/jrrigby1/SCIENCE/Lensed-LBGs/Planck_Arc/Sunburst_HST_Version4.0/"
        prefix= "V4.0_PSZ1G311.65-18.48_"  ; suffix = "_sci.fits.gz"
        #          F390W, F555W  F128N, F153, F153)  # W in the filter part of filename is Mike's typo
        offband = ("F390W_0.03g0.8_cr1.2_0.7_drc", "F555W_0.03g0.8_cr1.2_0.7_drc", "F128W_0.03g0.8_cr4.0_0.7_drz", \
                   "F153W_0.03g0.8_cr4.0_0.7_drz", "F153W_0.03g0.8_cr4.0_0.7_drz")
        #       F410M   F410M  F126N  F164N   F167N
        onband  = ("F410M_0.03g0.8_cr1.2_0.7_drc", "F410M_0.03g0.8_cr1.2_0.7_drc", "F126W_0.03g0.8_cr4.0_0.7_drz", \
                   "F164W_0.03g0.8_cr4.0_0.7_drz", "F167W_0.03g0.8_cr4.0_0.7_drz")
        outfilename = ("Lya_contsubF390W", "Lya_contsubF555W", "OII_contsub", "Hbeta_contsub", "OIII_contsub")
    return(fitsdir, prefix, suffix, offband, onband, outfilename)

def compute_linerats(df) :
    df['O32']  = df['OIII_both'] / df['OII']
    df['O3HB'] = df['OIII_5007'] / df['HBeta']
    df['R23']  = (df['OIII_both'] + df['OII'])/ df['HBeta'] # R23=(4959+5007+3727+3729) / Hbeta

def measure_NB_fluxes_in_FIRE_slits():
    ffdir = '/Users/jrrigby1/SCIENCE/Lensed-LBGs/Planck_Arc/JR_narrowband/'
    dirs2phot = ('Flux_maps_v2', 'Flux_maps_v2_convolved', 'Seeing_blurred', 'Seeing_blurred_0p75/')
    f_images = ['Lya_F390W.fits', 'OII.fits', 'Hbeta.fits', 'OIII_5007.fits', 'OIII_both.fits']
    f_names  = ['Lya', 'OII', 'HBeta', 'OIII_5007', 'OIII_both']
    regdir = '/Users/jrrigby1/Dropbox/MagE_atlas/Finders/pszarc1550m78/'
    F814_regfile = 'simple_FIRE_slits_F814W_forphotometry.reg'
    short_regfile = 'simple_FIRE_slits_F814W_forphot_short.reg'
    df_NB_fireslits = {}
    for mapdir in dirs2phot :      # Do photometry on NB images, using FIRE slits.  
        df_tmp2 = {} # dict of dfs
        for ii, image in enumerate(f_images) :
            image_w_path = ffdir + mapdir + '/' + image
            #print("Doing photometry on", f_names[ii], ", dir", mapdir)
            tmp_results = phot.photometry_loop_regions(image_w_path,  regdir + short_regfile)
            df_tmp = DataFrame.from_dict(tmp_results).T
            df_tmp.drop(['npix', 'median', 'mean', 'stddev'], inplace=True, axis=1)
            df_tmp.rename(columns={'thesum': f_names[ii]}, inplace=True)
            df_tmp2[f_names[ii]] = df_tmp
        df_tmp3 = concat(df_tmp2, axis=1)
        df_tmp3.columns = df_tmp3.columns.get_level_values(0)
        compute_linerats(df_tmp3)
        df_NB_fireslits[mapdir] = df_tmp3
        df_NB_fireslits[mapdir]['isleak'] = Series(fireslit_is_leaker())
    return(df_NB_fireslits)

def fireslit_is_leaker() :
    # returns a dictionary of which FIRE slits contain the leaker
    slits = {'F-0': True, 'F-1': True, 'F-2': True, 'F-7': True, 'F-8': True, 'F-9': True, \
            'F-3': False, 'F-4': False, 'F-5': False, 'F-6': False}
    return(slits)


