''' Useful functions from the Sunburst Arc notebooks.  jrigby Dec 2021'''
from re import sub 

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
