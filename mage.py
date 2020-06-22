''' Scripts to read and analyze MagE/Magellan spectra.  
    jrigby, begun Oct 2015.  Updates March--Dec 2016, then a later trickle.
'''
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
from jrr import spec
from jrr import util
import numpy as np
import pandas 
from   matplotlib import pyplot as plt
from   re import split, sub, search
from   os.path import expanduser, isfile, basename
import re
import glob
from subprocess import check_output   # Used for grepping from files
from   astropy.io import fits
import astropy.convolution
from astropy.wcs import WCS

color1 = 'k'     # color for spectra
color2 = '0.65'  # color for uncertainty spectra
color3 = '0.5'   # color for continuum
color4 = 'b'     # color for 2nd spectrum, for comparison

def Chisholm_norm_regionA() :   # Region where John Chisholm says to normalize
    return(1267.0, 1276.0)  # These are rest wavelengths, in Angstroms

def sunburst_translate_names() :
    # Translate the old MagE names for the sunburst arc pointings into the same format as the FIRE pointings.
    # See ~/Dropbox/SGAS-shared/NIR_spectra/FIRE/sunburst_reduced/what_these_data_are_JRmods_v2.txt
    sunburst_translate = {'planckarc_pos1'    : 'M-0',   'planckarc_h6'  : 'M-2',  'planckarc_h4' : 'M-3',            'planckarc_h1' : 'M-4'}
    sunburst_translate.update({'planckarc_h3' : 'M-5',   'planckarc_h1a' : 'M-6',  'planckarc_h1andh1a' : 'M-4+M-6'})
    sunburst_translate.update({'planckarc_h9' : 'M-7',   'planckarc_f'   : 'M-8',  'planckarc_h2' : 'M-9'})
    #there is no M-1; it was observed with FIRE but not with MagE.
    long_dict = {}
    for key, value in sunburst_translate.items() :
        long_dict.update({key : 'sunburst_'+value})
    return(long_dict)
# Now, how to integrate this new function into organize_labels, and speclist??

def organize_labels(group) :
    # Batch1 is what is published in Rigby et al. 2018.  Batch2 is what was processed by Feb 2018.  Batch 3 was processed Dec 2018
    batch1  = ('rcs0327-B', 'rcs0327-E', 'rcs0327-G', 'rcs0327-U', 'rcs0327-counterarc', 'S0004-0103', 'S0033+0242', 'S0108+0624', 'S0900+2234')
    batch1 += ('S0957+0509', 'S1050+0017', 'Horseshoe',  'S1226+2152', 'S1429+1202', 'S1458-0023', 'S1527+0652', 'S1527+0652-fnt', 'S2111-0114', 'Cosmic~Eye', 'S2243-0935')      
    batch2 = ('planckarc_pos1', 'planckarc', 'PSZ0441_slitA', 'PSZ0441_slitB', 'PSZ0441', 'SPT0310_slitA', 'SPT0310_slitB', 'SPT0310', 'SPT2325')  # Friends of Megasuara, batch2
    batch3 = ('planckarc_h1',  'planckarc_h1a', 'planckarc_h1andh1a', 'planckarc_h2', 'planckarc_h3', 'planckarc_h4', 'planckarc_h6', 'planckarc_f',  'planckarc_h9',  'SPT0356',  'SPT0142')
    batch4 = ('S1226+2152image3') # 2020 observations
    metabatch = ('planckarc_nonleak', 'planckarc_leak', 'planckarc_fire_nonleak', 'planckarc_fire_leak', 'rcs0327-all')
    if group   == 'batch1' : return(batch1)
    elif group == 'batch2' : return(batch2)
    elif group == 'batch3' : return(batch3)
    elif group == 'batch4' : return(batch4)
    elif group == 'batch23' : return(batch2 + batch3)
    elif group == 'batch123' : return(batch1 + batch2 + batch3)
    elif group == 'batch1234' : return(batch1 + batch2 + batch3 + batch4)
    elif group == 'metabatch' : return(metabatch)
    else : raise Exception("Error: label group unrecognized, not one of these: batch1, batch2, batch3, batch4, batch123, batch1234, metabatch.")
        
def longnames(mage_mode) :
    (spec_path, line_path) = getpath(mage_mode)
    thefile = spec_path + "dict_longnames.txt"
    longnames = pandas.read_table(thefile, delim_whitespace=True, comment="#")
    longnames['longname'] = longnames['longname'].str.replace("_", " ")
    longnames.set_index("short_label", inplace=True, drop=True)  # New! Index by short_label.  May screw stuff up downstream, but worth doing
    return(longnames)

def prettylabel_from_shortlabel(short_label) :  # For making plots
    temp = re.sub('Horseshoe', 'Cosmic Horseshoe SE', re.sub('-fnt', ' faint tail', re.sub('~',' ', short_label)))
    temp2 = re.sub("rcs0327-", "RCS0327 Knot ", ( re.sub("rcs0327-counterarc", "RCS0327 counterarc", temp)))
    if re.match("SPT", temp2) :  temp3 = temp2
    else :    temp3 = re.sub("^S", "SGAS J",    re.sub("-bright", "bright", temp2))
    return ( re.sub("-", r'$-$', temp3))

def getpath(mage_mode) : 
    ''' Haqndle paths for python MagE scripts.  Two use cases:
    A) I am on satchmo, & want to use spectra in  /Users/jrrigby1/SCIENCE/Lensed-LBGs/Mage/Combined-spectra/
       mage_mode = "reduction"
    B) I am on Milk, or a collaborator, using the "released" version of the MagE data, ~/Dropbox/MagE_atlas/
       mage_mode = "released"
       This is how to analyze the mage Data, same as the other collaborators.'''
    if mage_mode == "reduction" :
        spec_path = "/Users/jrrigby1/SCIENCE/Lensed-LBGs/Mage/Combined-spectra/"
        line_path = "/Users/jrrigby1/SCIENCE/Lensed-LBGs/Mage/Analysis/Plot-all/Lines/"
        return(spec_path, line_path)
    elif mage_mode == "released" :
        homedir = expanduser('~')
        spec_path = homedir + "/Dropbox/MagE_atlas/Spectra/"
        line_path = homedir + "/Dropbox/MagE_atlas/Linelists/"
        return(spec_path, line_path)
    else :
        print("Unrecognized mage_mode " + mage_mode)
        return(1)
        
def getlist(mage_mode, optional_file=False, zchoice='stars', MWdr=True) : 
    ''' Load list of MagE spectra and redshifts.  Loads into a Pandas data frame'''
    if optional_file :  # User wants to supply an optional file.  Now deprecated; better to filter in Pandas
        thelist = optional_file
        print("WARNING! Using alternate file ", optional_file, "ONLY DO SO FOR TESTING!")
    else :
        (spec_path, line_path) = getpath(mage_mode)
        thelist = spec_path + "spectra-filenames-redshifts.txt"
    pspecs = pandas.read_table(thelist, delim_whitespace=True, comment="#")
    if MWdr : pspecs['filename'] = pspecs['filename'].str.replace('.txt', '_MWdr.txt')
    # specs holds the filenames and redshifts, for example   specs['filename'], specs['z_stars'] 
   
    # z_syst is best estimate of systemic redshift.  Default is stars if measured, else nebular, else ISM.  Can also choose nebular if measured, else ISM
    pspecs['fl_st']   = pspecs['fl_st'].astype(np.bool, copy=False)
    pspecs['fl_neb']  = pspecs['fl_neb'].astype(np.bool, copy=False)
    pspecs['fl_ISM']  = pspecs['fl_ISM'].astype(np.bool, copy=False)
    pspecs['z_stars'] = pspecs['z_stars'].astype(np.float64)
    pspecs['sig_st']  = pspecs['sig_st'].astype(np.float64)
    pspecs['z_neb']   = pspecs['z_neb'].astype(np.float64)
    pspecs['sig_neb'] = pspecs['sig_neb'].astype(np.float64)
    pspecs['z_ISM']   = pspecs['z_ISM'].astype(np.float64)
    pspecs['sig_ISM'] = pspecs['sig_ISM'].astype(np.float64)
    pspecs['z_syst'] = np.float64() # initialize
    pspecs['dz_syst'] = np.float64()
    # dz_syst is uncertainty in systemic redshift.
    extra_dv = 500. # if using a proxy for systematic redshift, increase its uncertainty
    extra_dz = extra_dv / 2.997E5 * (1.+ pspecs['z_syst'])
    if zchoice  == 'stars' :
        pspecs.loc[~pspecs['fl_st'], 'z_syst'] = pspecs['z_stars']   # stellar redshift if have it
        pspecs.loc[(pspecs['fl_st']) & (~pspecs['fl_neb']), 'z_syst']                      = pspecs['z_neb']  # else, use nebular z
        pspecs.loc[(pspecs['fl_st']) & (pspecs['fl_neb']) & (~pspecs['fl_ISM']), 'z_syst'] = pspecs['z_ISM'] #else ISM
        pspecs.loc[(pspecs['fl_st']) & (pspecs['fl_neb']) & (pspecs['fl_ISM']), 'z_syst']  = -999  # Should never happen
        pspecs.loc[~pspecs['fl_st'], 'dz_syst']                        = pspecs['sig_st']
        pspecs.loc[(pspecs['fl_st']) & (~pspecs['fl_neb']),  'dz_syst'] = np.sqrt(( pspecs['sig_neb']**2 + extra_dz**2).astype(np.float64))
        pspecs.loc[(pspecs['fl_st']) & (pspecs['fl_neb']) & (~pspecs['fl_ISM']), 'dz_syst'] = np.sqrt(( pspecs['sig_ISM']**2 + extra_dz**2).astype(np.float64))
        pspecs.loc[(pspecs['fl_st']) & (pspecs['fl_neb']) & (pspecs['fl_ISM']), 'dz_syst'] = -999  # Should never happen
    elif zchoice == 'neb' :
        pspecs.loc[~pspecs['fl_neb'], 'z_syst']                        = pspecs['z_neb']  # use nebular z if have
        pspecs.loc[(pspecs['fl_neb']) & (pspecs['fl_ISM']), 'z_syst'] = -999  # Should never happen
        pspecs.loc[~pspecs['fl_neb'], 'dz_syst'] = pspecs['sig_neb']
        pspecs.loc[(pspecs['fl_neb']) & (~pspecs['fl_ISM']), 'dz_syst'] = np.sqrt(( pspecs['sig_ISM']**2 + extra_dz**2).astype(np.float64))
        pspecs.loc[(pspecs['fl_neb']) & (pspecs['fl_ISM']), 'dz_syst'] = -999  # Should never happen
    else : raise Exception("Error: zchoice unrecognized, is not neb or stars")

    if mage_mode == "reduction" :  # Add the path to the filename, if on satchmo in /SCIENCE/
        pspecs['filename'] = pspecs['origdir'] + pspecs['filename']
    pspecs.set_index("short_label", inplace=True, drop=False)  # Index by short_label.
    return(pspecs)  # I got rid of Nspectra.  If need it, use len(pspecs)

def wrap_getlist(mage_mode, which_list="wcont", drop_s2243=True, optional_file=False, labels=(), zchoice='stars', MWdr=True) :
    ''' Wrapper for getlist above.  Get list of MagE spectra and redshifts, for subsets.'''
    if which_list == "wcont" :      # Get those w jane's bespoke continuum fit.
        pspecs = getlist(mage_mode, zchoice=zchoice, MWdr=MWdr) 
        speclist =   pspecs[pspecs['filename'].str.contains('combwC1')]  # only w continuum
        if drop_s2243 :          # Drop S2243 bc it's an AGN
            #wcont = wcont[~wcont.index.isin("S2243-0935")].reset_index(drop=True)
            speclist = speclist[~speclist.index.isin(("S2243-0935",))]
    elif which_list == "all" :
        speclist = getlist(mage_mode, zchoice=zchoice, MWdr=MWdr)
    elif (which_list == "labels" and labels) or 'batch' in which_list :   # Get speclist for a list of short_labels
        if 'batch' in which_list:  labels =  organize_labels(which_list)  # should enable batch1, batch2, batch3
        pspecs = getlist(mage_mode, optional_file, zchoice=zchoice, MWdr=MWdr) 
        speclist = pspecs[pspecs.index.isin(labels)]
        speclist['sort_cat'] = pandas.Categorical(speclist['short_label'], categories=labels, ordered=True)  #same order as labels
        speclist.sort_values('sort_cat', inplace=True)
    else : raise Exception("Error: variable which_list not understood")
    return(speclist)   # "short_label" is now .index

def convert_spectrum_to_restframe(sp, zz) :
    (rest_wave, rest_fnu, rest_fnu_u)         = spec.convert2restframe(sp.wave, sp.fnu,  sp.fnu_u,  zz, 'fnu')
    sp['rest_wave']   = pandas.Series(rest_wave)
    sp['rest_fnu']    = pandas.Series(rest_fnu)
    sp['rest_fnu_u']  = pandas.Series(rest_fnu_u)
    (rest_wave, rest_flam, rest_flam_u) = spec.convert2restframe(sp.wave, sp.flam,  sp.flam_u,  zz, 'flam')
    sp['rest_flam']    = pandas.Series(rest_flam)
    sp['rest_flam_u']  = pandas.Series(rest_flam_u)
    spec.calc_dispersion(sp, 'rest_wave', 'rest_disp')   # rest-frame dispersion, in Angstroms    
    if 'fnu_cont' in sp :
        (junk   , rest_fnu_cont, rest_fnu_cont_u) = spec.convert2restframe(sp.wave, sp.fnu_cont, sp.fnu_cont_u, zz, 'fnu')
        sp['rest_fnu_cont']   = pandas.Series(rest_fnu_cont)
        sp['rest_fnu_cont_u'] = pandas.Series(rest_fnu_cont_u)
        (junk   , rest_flam_cont, rest_flam_cont_u) = spec.convert2restframe(sp.wave, sp.flam_cont, sp.flam_cont_u, zz, 'flam')
        sp['rest_flam_cont']   = pandas.Series(rest_flam_cont)
        sp['rest_flam_cont_u'] = pandas.Series(rest_flam_cont_u)
    if 'fnu_autocont' in sp :
        (junk   , rest_fnu_cont, dummy) = spec.convert2restframe(sp.wave, sp.fnu_autocont, sp.fnu_autocont, zz, 'fnu')
        sp['rest_fnu_autocont']   = pandas.Series(rest_fnu_cont)
        (junk   , rest_flam_cont, dummy) = spec.convert2restframe(sp.wave, sp.flam_autocont, sp.flam_autocont, zz, 'flam')
        sp['rest_flam_autocont']   = pandas.Series(rest_flam_cont)

    return(0)  # acts directly on sp.  
    
def open_spectrum(infile, zz, mage_mode) :
    '''Reads a reduced MagE spectrum, probably ending in *combwC1.txt
      Inputs:   filename to read in, systemic redshift (to convert to rest-frame), mage_mode
      Outputs:  the object spectrum, in both flam and fnu (why not?)  plus continuum both ways, all in Pandas data frame
      Here are the most important columns in the output spectrum:
      wave               observed-frame vacuum barycentric wavelength in angstroms
      fnu			     observed-frame f_nu, in erg/s/cm^2/s
      wave_sky 	         ignore
      fnu_sky		     ignore
      badmask		     True: this pixel is flagged as bad
      fnu_cont		     JRR's hand-fit continuum, to fnu
      fnu_cont_u         uncertainty in above
      disp			     dispersion in wave, in Angstroms
      rest_fnu		     rest-frame f_nu, in erg/s/cm^2/s
      rest_fnu_cont	     JRR's hand-fit continuum, in rest-frame f_nu
      rest_fnu_autocont	 An automatic continuum fit.  Almost as good as hand-fit
      fnu_s99model		 John Chisholm's best S99 fit, same units as fnu
      rest_fnu_s99model	 Same as above but for rest-frame fnu
      fnu_s99data	     ignore (sanity-checking, should be identical to fnu, checks that s99 model imported correctly)
      Many of the key columns have an equivalent f_lambda, abbreviated flam, for example rest_flam_cont
      call:  (Pandas_spectrum_dataframe, spectral_resolution, dresoln) = jrr.mage.open_spectrum(infile, zz, mage_mode)
    '''
    (spec_path, line_path) = getpath(mage_mode)
    specdir = spec_path

    if 'RESOLUTIONGOESHERE' in open(specdir+infile).read() :  # spectral_resoln hasn't set resln yet
        resoln = -99; dresoln=-99
    else :
        command = "grep resolution " + specdir+infile   
        resoln = float(check_output(command, shell=True).split()[1])
        dresoln = float(check_output(command, shell=True).split()[3])

    if search("wC1", infile) :  hascont = True # The continuum exists        
    else :  hascont=False   # file lacks continuum, e.g. *comb1.txt

    sp =  pandas.read_table(specdir+infile, delim_whitespace=True, comment="#", header=0, dtype=np.float64)#, names=names)
    sp.rename(columns= {'noise'  : 'fnu_u'}, inplace=True)
    sp.rename(columns= {'avgsky' : 'fnu_sky'}, inplace=True)
    sp.rename(columns= {'obswave' : 'wave_sky'}, inplace=True)
    spec.calc_dispersion(sp, 'wave', 'disp')
    
    if hascont :
        sp.rename(columns= {'cont_fnu'    : 'fnu_cont'}, inplace=True)  # Rename for consistency
        sp.rename(columns= {'cont_uncert' : 'fnu_cont_u'}, inplace=True)
    sp['fnu'] = sp['fnu'].astype(np.float64)   #force to be a float, not a str
    sp['fnu_sky'] = sp['fnu_sky'].astype(np.float64)   #force to be a float, not a str
    sp['flam']     = spec.fnu2flam(sp.wave, sp.fnu)          # convert fnu to flambda
    sp['flam_u']   = spec.fnu2flam(sp.wave, sp.fnu_u)
    sp['flam_sky'] = spec.fnu2flam(sp.wave_sky, sp.fnu_sky)
    if(hascont) : 
        sp['flam_cont']   = spec.fnu2flam(sp.wave, sp.fnu_cont)
        sp['flam_cont_u'] = spec.fnu2flam(sp.wave, sp.fnu_cont_u)
    #if 'does_this_column_appear' in sp.columns  : print "Yes it does"  # way to check if continuum appears
    
    # Masks to be used later, by flag_skylines and flag_near_lines
    sp['badmask']  = False  # Mask: True are BAD pixels.  Sign convention of np.ma
    sp['linemask'] = False  # Mask: True are pixels that may be near a spectral feature
    flag_skylines(sp)    # Flag the skylines.  This modifies sp.badmask
    sp.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace any inf values with nan
    flag_huge(sp, colfnu='fnu',   thresh_hi=3E4, thresh_lo=-3E4, norm_by_med=True)
    flag_huge(sp, colfnu='fnu_u', thresh_hi=0.5, thresh_lo=-0.5, norm_by_med=False)    
    flag_oldflags(sp)
    convert_spectrum_to_restframe(sp, zz)
    sp['fnu_autocont'] = pandas.Series(np.ones_like(sp.wave)*np.nan)  # Will fill this with automatic continuum fit
    return(sp, resoln, dresoln)   # Returns the spectrum as a Pandas data frame, the spectral resoln as a float, and its uncertainty

def wrap_open_spectrum(label, mage_mode, addS99=False, zchoice='stars', MWdr=True) :
    '''  Convenience wrapper, open one MagE spectrum in one line.  label is one short_name.
    if addS99, adds Chisholm's S99 continuum fit to the sp dataframe  '''
    (spec_path, line_path) = getpath(mage_mode)
    specs = wrap_getlist(mage_mode, which_list="labels", labels=[label], zchoice=zchoice, MWdr=MWdr)
    if specs.size == 0 : raise Exception("DF of spectral info has zero size.  Invalid label?")
    zz_syst = specs.z_syst.values[0]
    infile = specs.filename[0]
    (sp, resoln, dresoln) = open_spectrum(infile, zz_syst, mage_mode)
    linelist = get_linelist_name(infile, line_path)
    (LL, zz_notused) =  get_linelist(linelist)
    boxcar = spec.get_boxcar4autocont(sp)
    fit_autocont(sp, LL, zz_syst, boxcar=boxcar)
    if addS99 and  getfullname_S99_spectrum(label) :  # If a Chisholm S99 fit file exists, and user wants to read it
        (S99, ignore_linelist) = open_S99_spectrum(label, denorm=True, MWdr=MWdr)
        sp['fnu_s99model']      = spec.rebin_spec_new(S99['wave'], S99['fnu_s99'], sp['wave'])
        sp['fnu_s99data']       = spec.rebin_spec_new(S99['wave'], S99['fnu_data'], sp['wave'])  # used for debugging
        sp['rest_fnu_s99model'] = spec.rebin_spec_new(S99['rest_wave'], S99['rest_fnu_s99'], sp['rest_wave'])
        sp['rest_fnu_s99data']  = spec.rebin_spec_new(S99['rest_wave'], S99['rest_fnu_data'], sp['rest_wave']) # used for debugging
    return(sp, resoln, dresoln, LL, zz_syst)

def open_many_spectra(mage_mode, which_list="wcont", labels=(), verbose=True, zchoice='stars', addS99=True, MWdr=True, silent=False) :
    ''' This opens all the Megasaura MagE spectra (w hand-fit-continuua) into honking dictionaries of pandas dataframes. Returns:
    sp:        dictionary of pandas dataframes containing the spectra
    resoln:    dictionary of resolutions (float)
    dresoln:   dictionary of uncertainty in resolutions (float)
    LL:        dictionary of pandas dataframes of linelists
    zz_syst:   dictionary of systemic redshifts (float)
    speclist:  pandas dataframe describing the spectra (from getlist or variants)
    MWdr:      use the Milky Way dereddened spectra?  YES YOU WANT THIS.  Default=False for backward compatability'''
    if not silent: print("Loading MagE spectra in advance; this may be slow, but worthwhile if doing a lot of back and forth.")
    sp = {}; resoln = {}; dresoln = {}
    LL = {}; zz_sys = {}; boxcar  = {}
    speclist = wrap_getlist(mage_mode, which_list=which_list, labels=labels, zchoice=zchoice, MWdr=MWdr)
    for label in speclist.index :
        if verbose: print("Loading  ", label)
        (sp[label], resoln[label], dresoln[label], LL[label], zz_sys[label]) = wrap_open_spectrum(label, mage_mode, addS99=addS99, zchoice=zchoice, MWdr=MWdr)
    return(sp, resoln, dresoln, LL, zz_sys, speclist)  # returns dictionaries for first 5 things, and the speclist
    
def get_S99_path(MWdr=True) :  # Get path for S99
    (spec_path, line_path) = getpath("released")
#    if MWdr :      S99path = spec_path + "../Contrib/S99/MW_Corrected/"
#    else    :      S99path = spec_path + "../Contrib/S99/New_right_flam/"
    if MWdr :      S99path = spec_path + "../Contrib/S99/Chisholm19/ApJ/Megasaura/"  #3/2020, updating to published S99 fits
    else    :      S99path = spec_path + "../Contrib/S99/Obsolete/New_right_flam/"
    return(S99path)
    
def getfullname_S99_spectrum(rootname, MWdr=True) : # Check whether a S99 spectrum exists
    S99file = get_S99_path(MWdr=MWdr) + rootname + '-sb99-fit.txt'
    if isfile(S99file) :  return(S99file)  # returns full path and filename 
    else : return(0)                       # returns 0 if cannot find the file

def list_S99_rootnames():
    names = [ basename(x) for x in glob.glob(get_S99_path() + "*-sb99-fit.txt") ]
    rootnames = [ re.sub("-sb99-fit.txt", "", x) for x in names]
    return(rootnames)
                
def open_S99_spectrum(rootname, denorm=True, altfile=None, MWdr=True, debug=False) : 
    ''' Reads a best-fit Starburst99 spectrum generated by John Chisholm.  It's in the rest-frame already
    Outputs: a pandas data frame that contains some of the columns that open_spectrum makes.
    Updated Dec 2016 to take JC's file (in flam), un-normalize it, and convert back to fnu.
    inputs:
    rootname:  label that Chisholm used.  Usually same as JRR's sort-label
    denorm:    Calculate the denormalization factor?  Will break for non-megasaura spectra
    altfile:   (Optional) Direct path and link to file    '''
    (spec_path, line_path) = getpath("released")
    if altfile :  S99file = altfile # manual override for path & filename
    else :        S99file =  getfullname_S99_spectrum(rootname, MWdr=MWdr)
    if debug : print("DEBUGGING open_S99_spectrum, file was", S99file)
    sp =  pandas.read_table(S99file, delim_whitespace=True, comment="#", names=('rest_wave', 'rest_flam_data_norm', 'rest_flam_data_u_norm', 'rest_flam_s99_norm'))
    sp['rest_flam_s99_u'] = np.nan
    sp['badmask']  = False
    sp['linemask'] = False
    if denorm : 
        # JC converted spectra to flam, then normalized over norm_regionA.  I need to take this back out.
        if rootname == "chuck" :  orig_sp =  read_chuck_UVspec()
        elif rootname == "Stack-A" :
            (orig_sp, dummyLL) = open_stacked_spectrum("released", which_stack="Stack-A")
        else:
            (orig_sp, orig_resoln, orig_dresoln, orig_LL, orig_zz_syst)  =  wrap_open_spectrum(rootname, "released", zchoice='stars')
        norm_region = Chisholm_norm_regionA() 
        norm1 = orig_sp['rest_flam'][orig_sp['rest_wave'].between(*norm_region)].median() #norm for orig spectrum
        norm2 = sp['rest_flam_data_norm'][sp['rest_wave'].between(*norm_region)].median() #norm for JC's fit.  should be ~1
    else :
        norm1 = 1.0  ; norm2 = 1.0  # Don't renormalize.  Good choice for Chuck's spectrum, possibly the stack
    #print "DEBUGGING, norm1, norm2 were", norm1, norm2
    sp['rest_flam_data']   = sp['rest_flam_data_norm']   * norm1 / norm2 # Un-do the normalization that JC applied.
    sp['rest_flam_data_u'] = sp['rest_flam_data_u_norm'] * norm1 / norm2
    sp['rest_flam_s99']    = sp['rest_flam_s99_norm']    * norm1 / norm2
    if not (rootname == "Stack-A" or rootname == "chuck") :  # if an indy spectrum, convert S99 to rest frame
        (sp['wave'], sp['flam_S99'], sp['flam_S99_u'])   = spec.convert2obsframe(sp.rest_wave, sp.rest_flam_s99,  sp.rest_flam_s99_u,  orig_zz_syst, "flam")
        (sp['wave'], sp['flam_data'], sp['flam_data_u']) = spec.convert2obsframe(sp.rest_wave, sp.rest_flam_data, sp.rest_flam_data_u, orig_zz_syst, "flam")
        sp['fnu_s99']  = spec.flam2fnu(sp.wave, sp.flam_S99)
        sp['fnu_data']   = spec.flam2fnu(sp.wave, sp.flam_data)
        sp['fnu_data_u'] = spec.flam2fnu(sp.wave, sp.flam_data_u)
    sp['rest_fnu_data']    = spec.flam2fnu(sp['rest_wave'], sp['rest_flam_data']) # convert back to fnu
    sp['rest_fnu_data_u']  = spec.flam2fnu(sp['rest_wave'], sp['rest_flam_data_u'])
    sp['rest_fnu_s99']    = spec.flam2fnu(sp['rest_wave'], sp['rest_flam_s99'])
    sp['rest_fnu_s99model'] =  sp['rest_fnu_s99'] 
    sp['rest_fnu_s99_u']  = spec.flam2fnu(sp['rest_wave'], sp['rest_flam_s99_u'])
    flag_huge(sp, colfnu='rest_fnu_data', thresh_hi=50., thresh_lo=-50., norm_by_med=False)
    flag_huge(sp, colfnu='rest_fnu_s99', thresh_hi=50., thresh_lo=-50., norm_by_med=False)
    (LL, z_sys) = get_linelist(line_path + "stacked.linelist")  #z_syst should be zero here.
    spec.calc_dispersion(sp, 'rest_wave', 'rest_disp')
    boxcar = spec.get_boxcar4autocont(sp)
    fit_autocont(sp, LL, zz=0, make_derived=False, colwave='rest_wave', colfnu='rest_fnu_s99', colfnuu='rest_fnu_s99_u', colcont='rest_fnu_s99_autocont', boxcar=boxcar)
    fit_autocont(sp, LL, zz=0, make_derived=False, colwave='rest_wave', colfnu='rest_fnu_data', colfnuu='rest_fnu_data_u', colcont='rest_fnu_data_autocont', boxcar=boxcar)
    return(sp, LL)
    
def open_all_S99_spectra(MWdr=True) :
    ''' This opens all of JC's Starburst99 spectra, into two honking dictionaries of pandas dataframes
    one for spectra, pone for linelists'''
    S99 = {} ;  LL = {}
    for rootname in list_S99_rootnames() :
        print("Loading S99 ", rootname) 
        (S99[rootname], LL[rootname]) = open_S99_spectrum(rootname, MWdr=MWdr)
    return(S99, LL)
           
def dict_of_stacked_spectra(mage_mode) :
    ''' Returns:  a) main directory of stacked spectra,  b) a list of all the stacked spectra.'''
    (spec_path, line_path)  = getpath(mage_mode)
    if mage_mode == "reduction" :
        indir = spec_path + "../Analysis/Stacked_spectra_MWdr/"
    elif mage_mode == "released" :
        indir = spec_path + "../Stacked_spectra_MWdr/"
    files = glob.glob(indir+"*spectrum.txt")
    justfiles = [re.sub(indir, "", file) for file in files]         # list comprehension
    short_stackname = [re.sub("magestack_", "", file) for file in justfiles]         # list comprehension
    short_stackname = [re.sub("_spectrum.txt", "", file) for file in short_stackname]         # list comprehension
    stack_dict =  dict(list(zip(short_stackname, justfiles)))
    return(indir, stack_dict)

def open_stacked_spectrum(mage_mode, alt_infile=False, which_stack="standard", zchoice='byneb', colfnu='fweightavg', colfnuu='fjack_std', addS99=False) :
    ''' Open a stacked MagE spectrum. 
    Rewritten 8/2016 to make stacked spectrum look like a normal spectrum, so can re-use tools.
    Optional arguments are:
     - alt_infile: alternative stacked spectrum to grab (just the file, path from indir).  Over-rides which_stack
     - which_stack: easy selection of default spectra.  Current choices are "standard", "Stack-A"
     - colfnu:     which column to use as fnu (choose from favg (straight avg) fmedian (median), fweightavg (weighted average)
     - colfnuu:    which column to use for fnu_u '''
    (spec_path, line_path) = getpath(mage_mode)
    (indir, stack_dict) = dict_of_stacked_spectra(mage_mode)
    if alt_infile :
        print("  Caution, I am using ", alt_infile, "instead of the default stacked spectrum.")
        infile = indir + alt_infile
    else :
        if   which_stack == "standard" :  stackfile = "magestack_" + zchoice + "_standard_spectrum.txt"
        elif which_stack == "Stack-A"  :  stackfile = "magestack_" + zchoice + "_ChisholmstackA_spectrum.txt"
        elif which_stack == "divbys99" :  stackfile = "magestack_" + zchoice + "_divbyS99_spectrum.txt"
        else : raise Exception("I do not recognize which_stack as one of the choices: standard, Stack-A, or divbys99")
        infile = indir + stackfile
    sp =  pandas.read_table(infile, delim_whitespace=True, comment="#", header=0, dtype=np.float64)
    if 'restwave' in list(sp.keys()) :   sp.rename(columns = {'restwave' : 'rest_wave'}, inplace=True)
    if which_stack == "Stack-A" :
        sp['rest_wave'] = sp['rest_wave'] / (1 + 0.0000400)  # correct slight redshift offset, from 12 km/s measured by Chisholm
    sp['wave'] = sp['rest_wave']
    sp['fnu'] = sp[colfnu]
    sp['fnu_u'] = sp[colfnuu]    
    sp['flam']      = spec.fnu2flam(sp['wave'], sp['fnu'])
    sp['flam_u']    = spec.fnu2flam(sp['wave'], sp['fnu_u'])
    if addS99 and which_stack == "Stack-A" and getfullname_S99_spectrum(which_stack) :
        (S99, ignore_linelist) = open_S99_spectrum(which_stack, denorm=True, MWdr=True, debug=True)
        #print "DEBUGGING", sp.keys(), "\n\n", S99.keys()
        sp['fnu_s99model']      = spec.rebin_spec_new(S99['rest_wave'], S99['rest_fnu_s99'], sp['wave'])
        sp['fnu_s99data']       = spec.rebin_spec_new(S99['rest_wave'], S99['rest_fnu_data'], sp['wave'])  # used for debugging
        sp['rest_fnu_s99model'] = sp['fnu_s99model']
        sp['rest_fnu_s99data']  = sp['fnu_s99data']
    (LL, z_sys) = get_linelist(line_path + "stacked.linelist")  #z_syst should be zero here.
    spec.calc_dispersion(sp, 'wave', 'disp')
    sp['badmask'] = False
    sp['linemask'] = False
    convert_spectrum_to_restframe(sp, 0.0)  # z=0
    boxcar = spec.get_boxcar4autocont(sp)
    #print "DEBUGGING, boxcar is ", boxcar
    fit_autocont(sp, LL, zz=0.0, vmask=1000, boxcar=3001)
    sp['unity'] = 1.0
    return(sp, LL) # return the Pandas DataFrame containing the stacked spectrum
     
def open_Crowther2016_spectrum() :
    print("STATUS:  making velocity plots of the stacked Crowther et al. 2016 spectrum")
    infile = "/Users/jrrigby1/SCIENCE/Lensed-LBGs/Mage/Lit-spectra/Crowther2016/r136_stis_all.txt" # on satchmo
    sp =  pandas.read_table(infile, delim_whitespace=True, comment="#", header=0)  # cols are wave, flam
    sp['fnu'] = spec.fnu2flam(sp['wave'], sp['flam'])
    return(sp)
        
def plot_1line_manyspectra(line_cen, line_label, win, vel_plot=True, mage_mode="reduction", specs=[], size=(5,16), fontsize=16, suptitle=False) :
    ''' Plot one transition, versus (wavelength or velocity?), for many MagE spectra
    line_cen:     rest-wavelength of line to plot, in Angstroms.
    line_label:   label for that line
    win:          window around that line.  If vel_plot, then units are km/s.  If not vel_plot, units are rest-frame Angstroms
    vel_plot:     (optional, Boolean): If True, x-axis is velocity.  If False, x-axis is wavelength.
    specs:        (optional) Pandas dataframe of MagE filenames & redshifts.
    size:         (optional) size of plot, in inches
    fontsize:     (optional) fontsize
    suptitle:     (optional) plot line label as super title?
'''    
    if len(specs) == 0 :
        specs = wrap_getlist(mage_mode, which_list="wcont", MWdr=True)
    Nspectra = len(specs)
    Ncol = 1
    Nrow = np.ceil(Nspectra / Ncol)  # Calculate how many rows to generate
    fig = plt.figure(figsize=size)
    
    for ii in range(0, Nspectra) :              
        label     = specs['short_label'][ii]
        filename  = specs['filename'][ii]
        zz =  specs['z_neb'][ii]
        (sp, resoln, dresoln)  = open_spectrum(filename, zz, mage_mode)
        restwave   = sp.wave / (1.0+zz)
        fnu_norm   = sp.fnu / sp.fnu_cont
        fnu_norm_u = util.sigma_adivb(sp.fnu, sp.fnu_u, sp.fnu_cont, sp.fnu_cont_u)                
        ax = fig.add_subplot(Nrow, Ncol, ii+1)

        if(vel_plot) : # Do velocity plot first, since already coded up.  Later, can add versus wavelength
            vel = spec.convert_restwave_to_velocity(restwave, line_cen)   # velocity in km/s
            in_window = vel.between(-1*win, win)
            plt.step(vel[in_window], fnu_norm[in_window], color=color1)
            plt.step(vel[in_window], fnu_norm_u[in_window], color=color2)
            plt.plot( (0., 0.), (0.0,2), color=color3, linewidth=2)  # plot tics at zero velocity
            plt.xlim(-1*win, win)
        else :
            in_window = restwave.between(line_cen - win, line_cen + win)
            plt.step(restwave[in_window], fnu_norm[in_window], color=color1)
            plt.step(restwave[in_window], fnu_norm_u[in_window], color=color2)
            plt.plot( (line_cen, line_cen), (0.0,2), color=color3, linewidth=2)  # plot tics at zero velocity
            plt.plot( (1037.6167, 1037.6167), (0.0,2), color=color3, linewidth=2)  # TEMP KLUDGE PLOT OVI 1037 as well
            plt.xlim(line_cen - win, line_cen + win)
        plt.plot( (-1*win, win), (1.0,1.0), color=color3)  # Already normalized by continuum, so plot unity continuum. 
        plt.ylim(0.0, 1.5)  # May need to change these limits
        if ii == (Nspectra - 1) :  # if last subplot, make xlabel
            #plt.annotate( line_label, (0.2,0.1), xycoords="axes fraction", fontsize=fontsize)
            if suptitle : plt.suptitle(line_label, fontsize=fontsize)
            if vel_plot :
                plt.xlabel("rest-frame velocity (km/s)", fontsize=fontsize)  
            else :
                plt.xlabel(r'rest-frame wavelength ($\rm \AA$)', fontsize=fontsize)    
        else :
            ax.axes.xaxis.set_ticklabels([])  # if not last subplot, suppress  numbers on x axis
        plt.annotate( label, (0.2,0.8), xycoords="axes fraction", fontsize=fontsize)
        plt.annotate("z="+str(zz), (0.75,0.1), xycoords="axes fraction", fontsize=fontsize)
        #plt.ylabel("continuum-normalized fnu")
        plt.ylabel(r'norm. f$_{\nu}$', fontsize=14)  #fontsize=fontsize)
#        fig.(hspace=0)
    return(0)

def get_linelist_name(filename, line_path) :
    ''' Given a spectrum filename, guess what its associated linelist should be.  A convenience function.'''
    if '_MWdr.txt' in filename :
        linelist = line_path + sub("_MWdr.txt", ".linelist", (split("/", filename)[-1]))  #backward compatibility, and still handle MW-dereddened spectra
    else :
        linelist = line_path + sub(".txt", ".linelist", (split("/", filename)[-1]))
    return(linelist)
        
def get_linelist(linelist) :  
    ''' Grabs linelist (generated beforehand by concat-linelist.pl) and saves to a Pandas Data Frame.
    Update: Make it grab the systemic redshift, too.  
    '''
    L = pandas.read_table(linelist,  delim_whitespace=True, comment="%", names=('restwav', 'lab1', 'lab2', 'foo1', 'foo2', 'color', 'zz', 'type', 'src'))
    L.lab1 = L.lab1.str.replace("_", " ")  # clean up formatting.  Should go fix original Linelists/MINE to get all to have _
    L.insert(7, 'obswav', L.restwav * (1.0 + L.zz))  # observed wavelength for this line
    L.insert(8, 'fake_v',   0)  # need this for plot_linelist if velplot
    L.insert(8, 'fake_wav', 0)  # need this for Mage_plot if restframe
    L['vmask'] = 500.0 # default window to mask the line 
    # Grab the systemic redshift, too
    command = "grep SYSTEMIC " + linelist
    z_systemic = float(check_output(command, shell=True).split()[3])
    #print "Loaded linelist ", linelist, " with systemic redshift of ", z_systemic
    return(L, z_systemic)  # Returning a pandas data frame of the linelist.  Key cols are restwav, obswav, zz, lab1 (the label)

def plot_linelist(L_all, z_systemic=np.nan, restframe=False, velplot=False, line_center=np.nan, alt_ypos=False) :  
    '''Plot ticks and line labels for lines from a linelist, loaded by get_linelist.
    Can do either rest-frame or observed frame, x-axis wavelength or velocity
    '''
    #ypos = plt.ylim()[1] * 0.93  # previous
    ypos = (plt.ylim()[1] - plt.ylim()[0]) * 0.15    #trying moving ticks to bottom
    if (not restframe) and (not velplot) :
        L = L_all[L_all.obswav.between(plt.xlim()[0], plt.xlim()[1])].reset_index()
        wav2plot = L.obswav  # which col of wave to plot? 
    if restframe :
        if np.isnan(z_systemic) :
            raise Exception('ERROR: If restframe, then must specify z_systemic')    
        L_all.fake_wav =  L_all.obswav / (1.0+z_systemic) # to put labels at right place
        if not velplot :
            L = L_all[L_all.fake_wav.between(plt.xlim()[0], plt.xlim()[1])].reset_index()
            wav2plot = L.fake_wav
        if velplot :
            if np.isnan(line_center) :
                raise Exception('ERROR: If restframe and velplot, must specify rest wavelength line_center')
            L_all.fake_v =  spec.convert_restwave_to_velocity(L_all.fake_wav, line_center)
            L = L_all[L_all.fake_v.between(plt.xlim()[0], plt.xlim()[1])].reset_index()
            wav2plot = L.fake_v

    if (not restframe) and velplot :
        raise Exception('ERROR: If x-axis is velocity, then restframe must be True')        
        
    for i in range(0, L.index.size):  # Loop through the lines, and plot the ones on the current subplot
        plt.annotate(L.lab1[i], xy=(wav2plot[i], ypos*3), xytext=(wav2plot[i], ypos), color=L.color[i], rotation=90, ha='center', va='center', arrowprops=dict(color=L.color[i], shrink=0.2,width=0.5, headwidth=0))
    L = []
    L_all = []
    return(0)
    
def open_Leitherer_2011_stack() :
    # call:  (restwave, avg_flux, median_flux) = open_Leitherer_2011_stack() :
    infile = "/Users/jrrigby1/SCIENCE/Lensed-LBGs/Mage/Lit-spectra/Leitherer_2011/fos_ghrs_composite.txt"
    leith =  pandas.read_table(infile, delim_whitespace=True, comment="#", header=0)
    return(leith)

    
def mage2D_xy2wave(filename, xx) :  # indexed same as IRAF, NOT same as python. 1 is first pixel
    '''Get wavelength for Dan Kelson's 2D rectified MagE order images, e.g. s1527-2d-278-009sum.fits '''
    data, header = fits.getdata(filename, header=True)
    wave = header['crval1'] + header['cdelt1'] * (xx - header['crpix1'])
    return(wave)

def mage2D_iswave(filename, wave) :
    ''' Check whether a wave is contained in a Kelson 2D rectified MagE order image. If True, return x coordinate.'''
    data, header = fits.getdata(filename, header=True)
    firstwave =  header['crval1']
    lastwave  = mage2D_xy2wave(filename, header['naxis1'])
    if wave > firstwave and wave < lastwave :
        xx = header['crpix1'] + (wave - header['crval1'])/header['cdelt1']
        return(xx)
    else :
        return(False)
              
def flag_skylines(sp, skywidth=17.0, skyline=(5577., 6300.)) :
    # Mask skylines [O I] 5577\AA\ and [O I]~6300\AA,  flag spectrum +- skywidth of the skyline
    for thisline in skyline :
        sp.badmask.loc[sp['wave'].between(thisline-skywidth, thisline+skywidth)] = True 
    return(0)  # This function works directly on sp

def flag_huge(sp, colfnu='fnu', thresh_hi=3E4, thresh_lo=-3E4, norm_by_med=False) :
    # Set mask for pixels that have crazy high uncertainty.  These were flagged further upstream, where i had no mask.
    # Inputs:  sp is spectrum as Pandas data frame
    # if norm_by_med,  treat thresh_hi and thresh_lo as values
    # if not norm_by_med, treat thresh_hi and thresh_lo as relative to mean of colfnu
    # Output:  None.  It directly acts on sp.badmask
    if norm_by_med :
        thresh_hi *= sp[colfnu].median()
        thresh_lo *= sp[colfnu].median()
    sp.badmask.loc[sp[colfnu].gt(thresh_hi)] = True
    sp.badmask.loc[sp[colfnu].lt(thresh_lo)] = True
    return(0)    

def flag_oldflags(sp, colfnu='fnu') :
    sp.badmask.loc[sp[colfnu].eq(-0.999)]   = True  # a flag from upstream

def flag_where_nocont(sp) :
    sp.badmask.loc[sp['fnu_cont'].eq(9999)] = True  # where hand-drawn continuum is undefined
    return(0)
        
def fit_autocont(sp, LL, zz, vmask=500, boxcar=1001, flag_lines=True, make_derived=True, colwave='wave', colfnu='fnu', colfnuu='fnu_u', colcont='fnu_autocont') : 
    ''' Automatically fits a smooth continuum to a spectrum.
     Inputs:  sp,  a Pandas data frame containing the spectra, opened by mage.open_spectrum or similar
              LL,  a Pandas data frame containing the linelist, opened by mage.get_linelist(linelist) or similar
              zz,  the systemic redshift. Used to set window threshold around lines to mask
              vmask (Optional), velocity around which to mask lines, km/s. 
              boxcar (Optional), the size of the boxcar smoothing window, in pixels
              flag_lines (Optional) flag, should the fit mask out known spectral features? Can turn off for debugging
              colwave, colfnu, colfnuu:  which columns to find wave, fnu, fnu_u.
    '''
    # First, mask out big skylines. Done by mage.flag_skylines, which is called by mage.open_spectrum
    # Second, flag regions with crazy high uncertainty.  done in flag_huge, called by mage.open_spectrum
    # Third, mask out regions near lines.  Flagged in sp.linemask
    if flag_lines :
        LL['vmask'] = vmask
        spec.flag_near_lines(sp, LL, colv2mask='vmask', colwave=colwave)
    # Populate colcont with fnu, unless pixel is bad or has a spectral feature, in which case it stays nan.
    temp_fnu = sp[colfnu].copy(deep=True)
    temp_fnu.loc[(sp.badmask | sp.linemask).astype(np.bool)] = np.nan
    # Last, smooth with a boxcar
    smooth1 = astropy.convolution.convolve(np.array(temp_fnu), np.ones((boxcar,))/boxcar, boundary='fill', fill_value=np.nan)
    sp[colcont] = pandas.Series(smooth1)  # Write the smooth continuum back to data frame
    if make_derived : 
        # Make derived products from fnu_autocont:    flam_autocont,  rest_flam_autocont
        (dum, rest_fnu_autocont, dum) =  spec.convert2restframe(sp[colwave], sp.fnu_autocont, sp.fnu_autocont,  zz, 'fnu')
        sp['rest_fnu_autocont']       = pandas.Series(rest_fnu_autocont)
        sp['flam_autocont']     = spec.fnu2flam(sp[colwave], sp.fnu_autocont)
        (dum, rest_flam, dum)  = spec.convert2restframe(sp[colwave], sp.flam_autocont,  sp.flam_u,  zz, 'flam')    
        sp['rest_flam_autocont'] = pandas.Series(rest_flam)
    return(0)


# Routines to read in spectra from the literature

def add_columns_to_litspec(df) :
    ''' Add the right columns to a pandas df created fory a spectrum from the literature, so that my plotting
    tools won't barf.'''
    df['badmask']  = False  
    df['linemask'] = False  
    df['fnu_autocont'] = pandas.Series(np.ones_like(df.rest_wave)*np.nan)  # Will fill this with automatic continuum fit
    df['flam']     = spec.fnu2flam(df.wave, df.fnu)          # convert fnu to flambda
    df['flam_u']   = spec.fnu2flam(df.wave, df.fnu_u)
    # Give it a default line-list, so it can fit autocont w smoothing
    mage_mode = "reduction"
    (spec_path, line_path) = getpath(mage_mode)
    (LL, foobar) = get_linelist(line_path + "stacked.linelist")  #z_syst should be zero here.    
    fit_autocont(df, LL, 0.0, boxcar=501)
    return(0) # directly modified the df
    
def convert_chuck_UVspec(infile="KBSS-LM1.uv.fnu.fits", uncert_file="KBSS-LM1.uv.fnu.sig.fits", outfile=None, mage_mode="released") :
    ''' Takes a 1D spectrum with the wavelength stuck in the WCS header,          
    gets the wavelength array out, and packages the spectrum into a nice          
    pandas data frame.  Returns DF, and dumps a pickle file and csv file too.                  
    This was written to reach Chuck Steidel's stacked spectrum, and is not
    general enough to make into a function.  Example of munging: the wcs.dropaxis business.'''
    (spec_path, line_path) = getpath(mage_mode)
    litdir = spec_path + "Lit-spectra/Steidel2016/"
    sp = fits.open(litdir + infile)
    header = sp[0].header
    wcs = WCS(header)
    wcs2 = wcs.dropaxis(1)  # Kill the WCS's dummy 2nd dimension                  
    index = np.arange(header['NAXIS1'])
    temp =  (np.array(wcs2.wcs_pix2world(index, 0))).T
    wavelength = (10**temp)[:,0]
    fnu = sp[0].data
    if uncert_file :
        sp2 = fits.open(litdir + uncert_file)      # Get the uncertainty                   
        fnu_u = sp2[0].data
    else : fnu_u = np.zeros_like(fnu)
    # Make a pandas data frame                                                    
    foo = np.array((wavelength, fnu, fnu_u))
    print("DEBUG", foo.shape, wavelength.shape, fnu.shape, fnu_u.shape)
    df = pandas.DataFrame(foo.T, columns=("wave", "fnu", "fnu_u"))
    add_columns_to_litspec(df)
    if not outfile :
        outfile = sub(".fits", ".p", litdir + infile)
    df.to_pickle(outfile)
    txtfile = sub(".fits", ".csv", litdir + infile)
    df.to_csv(txtfile, sep='\t')    
    return(df)

def read_chuck_UVspec(mage_mode="released", addS99=False, autofitcont=False) :
    (spec_path, line_path) = getpath(mage_mode)
    litfile = spec_path + "../Lit-spectra/Steidel2016/KBSS-LM1.uv.fnu.csv"
    #names = ("index", "rest_wave", "rest_fnu", "rest_fnu_u")
    #sp = pandas.read_table(litfile, delim_whitespace=True, comment="#", names=names, skiprows=1, index_col=0)
    sp = pandas.read_table(litfile, delim_whitespace=True, comment="#")
    sp['rest_wave'] = sp['wave']
    sp['rest_fnu']  = sp['fnu']
    sp['rest_fnu_u']  = sp['fnu_u']
    sp['rest_flam']    = spec.fnu2flam(sp['rest_wave'], sp['rest_fnu'])
    sp['rest_flam_u']  = spec.fnu2flam(sp['rest_wave'], sp['rest_fnu_u'])
    if addS99 and getfullname_S99_spectrum("chuck") :
        sp['rest_wave'] = sp['rest_wave'] / (1 + 0.00013013)
        (S99, ignore_linelist) = open_S99_spectrum("chuck", denorm=True)
        #print "DEBUGGING", sp.keys(), "\n\n", S99.keys()
        sp['fnu_s99model']      = spec.rebin_spec_new(S99['rest_wave'], S99['rest_fnu_s99'], sp['rest_wave'])
        sp['fnu_s99data']       = spec.rebin_spec_new(S99['rest_wave'], S99['rest_fnu_data'], sp['rest_wave'])  # used for debugging
        sp['rest_fnu_s99model'] = sp['fnu_s99model']
        sp['rest_fnu_s99data']  = sp['fnu_s99data']
    if autofitcont :
        spec.calc_dispersion(sp, 'rest_wave', 'rest_disp')   # rest-frame dispersion, in Angstroms
        sp['badmask'] = False   
        boxcar = spec.get_boxcar4autocont(sp)
        (LL, z_sys) = get_linelist(line_path + "stacked.linelist")  #z_syst should be zero here.
        fit_autocont(sp, LL, 0.0, make_derived=False, boxcar=boxcar, colwave='rest_wave', colfnu='rest_fnu', colfnuu='rest_fnu_u', colcont='rest_fnu_autocont')
        sp['fnu_autocont'] = sp['rest_fnu_autocont'] # make multipanel-stacks.py happy
    return(sp)
    
def convert_chuck_mosfire(infile, outfile=None) :
    sp = fits.open(infile)
    header = sp[0].header
    wcs = WCS(header)
    wcs2 = wcs.dropaxis(1)  # Kill the WCS's dummy 2nd dimension
    index = np.arange(header['NAXIS1'])
    temp =  (np.array(wcs2.wcs_pix2world(index, 0))).T
    wavelength = (temp)[:,0]
    (flam, flam_u) = sp[0].data
    # Make a pandas data frame
    foo = np.array((wavelength, flam, flam_u))
    print("DEBUG", foo.shape, wavelength.shape, flam.shape, flam_u.shape)
    df = pandas.DataFrame(foo.T, columns=("wave", "flam", "flam_u"))
    if not outfile :
        outfile = sub(".fits", ".p", infile)
    df.to_pickle(outfile)
    return(df)

def read_shapley_composite() :
    infile = "/Users/jrrigby1/SCIENCE/Lensed-LBGs/Mage/Lit-spectra/LBGs/composite-LBG-shapley.dat"
    df = pandas.read_table(infile, delim_whitespace=True, comment="#")
    df['fnu_u'] = 0.0
    df['flam']     = spec.fnu2flam(df.wave, df.fnu)          # convert fnu to flambda
    df['flam_u']   = spec.fnu2flam(df.wave, df.fnu_u)
    convert_spectrum_to_restframe(df, 0.0)
    add_columns_to_litspec(df)
    return(df)

def read_our_COS_stack(resoln="full") :
    (specpath, linepath) = getpath("released")
    if resoln   == "matched_mage" : infile = specpath + "../Contrib/Chisholm16/raw/stacked_COS_spectrum_R3300.csv"
    elif resoln == "full"         : infile = specpath + "../Contrib/Chisholm16/raw/stacked_COS_spectrum_R1.4E4.csv"
    else : raise Exception("resoln must be full or matched_mage")
    df = pandas.read_csv(infile)
    df['rest_fnu'] = df['fweightavg']     # Kludge to make multipanel-stacks.py plot it
    df['rest_fnu_u'] = df['fweightavg_u'] # ditto
    df['unity'] = 1.0        # ditto.  Dummy continuum
#    df['rest_fnu_autocont'] = 1.0
    return(df)

def open_planckarc_sum(zz, vmask1, vmask2, smooth_length=50., option="full") :
    if   option == "full" : pfile = "psz1550_muse_full_F.fits.txt"
    elif option == "wtd"  : pfile = "psz1550_muse_full_whtd_F.fits.txt"
    else : raise Exception("Error: Option is one one of full or wtd")
    pf = pandas.read_table(pfile, delim_whitespace=True, comment="#", names=('wave', 'flam', 'flam_u'))
    pf['badmask']  = False
    pf['contmask'] = False
    pf['unity'] = 1.0
    spec.calc_dispersion(pf)
    pf['fnu']   =  spec.flam2fnu(pf['wave'], pf['flam'])
    pf['fnu_u'] =  spec.flam2fnu(pf['wave'], pf['flam_u'])
    (specpath, linepath) = getpath("reduction")
    linelist_filename = get_linelist_name("planck1.txt", linepath)
    (LL, dumzz) = get_linelist(linelist_filename)
    LL['vmask'] = vmask1
    LL.loc[LL['lab1'] == 'C IV', 'vmask'] =  vmask2
    LL.loc[LL['lab1'] == 'He II', 'vmask'] =  vmask2
    convert_spectrum_to_restframe(pf, zz)
    boxcar = spec.get_boxcar4autocont(pf, smooth_length)    
    spec.fit_autocont(pf, LL, zz, colv2mask='vmask', boxcar=boxcar, flag_lines=True, colwave='wave', colf='fnu', colmask='contmask', colcont='fnu_autocont')
    pf['flam_autocont']   = spec.fnu2flam(pf.wave, pf.fnu_autocont)
    convert_spectrum_to_restframe(pf, zz)  # again, to get contfit
    return(pf, LL)


#### Below are functions that are useful for intervening absorbers in megasaura

def recover_1shortname(gal, pointing) : # JRRedit tables don't have a column of shortname. So, reconstruct it
    shortname = ""
    if pointing == 'main' : shortname = gal
    else :
        shortname = string.replace(gal + '_' + pointing, "_knot", "-")
        shortname = string.replace(shortname, '_faint', '-fnt')
    print(shortname, gal, row)
    return(shortname)

def recover_all_shortnames(df, colgal='gal', colpointing='pointing') :  # same as above, but for the dataframe
    df['shortname'] = df[colgal] + "_" + df[colpointing]
    df['shortname'] = df['shortname'].str.replace('_faint', '-fnt')
    df['shortname'] = df['shortname'].str.replace('_knot', '-')
    df['shortname'] = df['shortname'].str.replace('_main', '')
    return(0) # acts on df

def load_doublet_df(batch) :  # Load a dataframe containing the intervening doublets.
    # batch should be one of 'batch1' (in Rigby et al 2018), 'batch23' (2017, 2018 data), or 'batch123' (everything)
    if batch not in ('batch1', 'batch23', 'batch123') : raise Exception("Error: batch must be batch1, batch23, or batch123")
    doubletfiles = {}
    doubletfiles['batch1']  = expanduser("~") + "/Dropbox/MagE_atlas/Contrib/Intervening/Doublet_search/Results_16Feb2018/found_doublets_SNR4_JRRedit.txt"
    doubletfiles['batch23'] = expanduser("~") + "/Dropbox/MagE_atlas/Contrib/Intervening/Doublet_search/Results_21Feb2019/found_doublets_batch3_SNR4_JRRedit.txt"
    if batch in ('batch1', 'batch23') :
        doublet_df = pandas.read_table(doubletfiles[batch], delim_whitespace=True, comment="#")
    elif batch == ('batch123') :
        doublet_df1 = pandas.read_table(doubletfiles['batch1'], delim_whitespace=True, comment="#")
        doublet_df2 = pandas.read_table(doubletfiles['batch23'], delim_whitespace=True, comment="#")
        doublet_df = pandas.concat([doublet_df1, doublet_df2])#, sort=False)
    recover_all_shortnames(doublet_df)
    return(doublet_df)
