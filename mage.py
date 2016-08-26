''' Scripts to read and analyze MagE spectra.  
    jrigby, begun Oct 2015.  Updates March--July 2016
'''
from jrr import spec
from jrr import util
import numpy as np
import pandas 
from   matplotlib import pyplot as plt
from cycler import cycler
from   astropy.io import fits
from   re import split, sub, search
from   os.path import expanduser
import re
import glob
from subprocess import check_output   # Used for grepping from files
import astropy.convolution

color1 = 'k'     # color for spectra
color2 = '0.65'  # color for uncertainty spectra
color3 = '0.5'   # color for continuum
color4 = 'b'     # color for 2nd spectrum, for comparison

def getpath(mage_mode) : 
    ''' Haqndle paths for python MagE scripts.  Two use cases:
    A) I am on satchmo, & want to use spectra in  /Volumes/Apps_and_Docs/WORK/Lensed-LBGs/Mage/Combined-spectra/
       mage_mode = "reduction"
    B) I am on Milk, or a collaborator, using the "released" version of the MagE data, ~/Dropbox/MagE_atlas/
       mage_mode = "released"
       This is how to analyze the mage Data, same as the other collaborators.'''
    if mage_mode == "reduction" :
        spec_path = "/Volumes/Apps_and_Docs/WORK/Lensed-LBGs/Mage/Combined-spectra/"
        line_path = "/Volumes/Apps_and_Docs/WORK/Lensed-LBGs/Mage/Analysis/Plot-all/Lines/"
        return(spec_path, line_path)
    elif mage_mode == "released" :
        homedir = expanduser('~')
        spec_path = homedir + "/Dropbox/MagE_atlas/Spectra/"
        line_path = homedir + "/Dropbox/MagE_atlas/Linelists/"
        return(spec_path, line_path)
    else :
        print "Unrecognized mage_mode " + mage_mode
        return(1)
        
def getlist(mage_mode, optional_file=False) : 
    ''' Load list of MagE spectra and redshifts.  Loads into a Pandas data frame'''
    if optional_file :  # User wants to supply an optional file.  Now deprecated; better to filter in Pandas
        thelist = optional_file
        print "WARNING! Using alternate file ", optional_file, "ONLY DO SO FOR TESTING!"
    else :
        (spec_path, line_path) = getpath(mage_mode)
        thelist = spec_path + "spectra-filenames-redshifts.txt"
    pspecs = pandas.read_table(thelist, delim_whitespace=True, comment="#")
    #pspecs.info()  # tell me about this pandas data frame
    #pspecs.keys()  # tell me column names
    #len(pspecs)    # length of pspecs
    # specs holds the filenames and redshifts, for example   specs['filename'], specs['z_stars']
    
    # z_syst is best estimate of systemic redshift.  From stars if measured, else nebular, else ISM
    pspecs['fl_st'] = pspecs['fl_st'].astype(np.bool, copy=False)
    pspecs['fl_neb'] = pspecs['fl_neb'].astype(np.bool, copy=False)
    pspecs['fl_ISM'] = pspecs['fl_ISM'].astype(np.bool, copy=False)
    pspecs['z_syst'] = 'foobar' # initialize
    pspecs['z_syst'][~pspecs['fl_st']] = pspecs['z_stars']  # stellar redshift if have it
    pspecs['z_syst'][(pspecs['fl_st']) & (~pspecs['fl_neb'])] = pspecs['z_neb']  # else, use nebular z
    pspecs['z_syst'][(pspecs['fl_st']) & (pspecs['fl_neb']) & (~pspecs['fl_ISM'])] = pspecs['z_ISM'] #else ISM
    pspecs['z_syst'][(pspecs['fl_st']) & (pspecs['fl_neb']) & (pspecs['fl_ISM'])] = -999  # Should never happen
    # dz_syst is uncertainty in systemic redshift.
    extra_dv = 500. # if using a proxy for systematic redshift, increase its uncertainty
    extra_dz = extra_dv / 2.997E5 * (1.+ pspecs['z_syst'])
    pspecs['dz_syst'] = 'foobar'
    pspecs['dz_syst'][~pspecs['fl_st']] = pspecs['sig_st']
    pspecs['dz_syst'][(pspecs['fl_st']) & (~pspecs['fl_neb'])] = np.sqrt(( pspecs['sig_neb']**2 + extra_dz**2).astype(np.float64))
    pspecs['dz_syst'][(pspecs['fl_st']) & (pspecs['fl_neb']) & (~pspecs['fl_ISM'])] = np.sqrt(( pspecs['sig_ISM']**2 + extra_dz**2).astype(np.float64))
    pspecs['dz_syst'][(pspecs['fl_st']) & (pspecs['fl_neb']) & (pspecs['fl_ISM'])] = -999  

    if mage_mode == "reduction" :  # Add the path to the filename, if on satchmo in /WORK/
        pspecs['filename'] = pspecs['origdir'] + pspecs['filename']
    return(pspecs)  # I got rid of Nspectra.  If need it, use len(pspecs)

def getlist_wcont(mage_mode, drop_s2243=True) :
    ''' Get the list of MagE spectra and redshifts, for those w continuum fit.'''
    pspecs = getlist(mage_mode) 
    wcont =   pspecs[pspecs['filename'].str.contains('combwC1')].reset_index()  # only w continuum
    # Drop S2243 bc it's an AGN
    if drop_s2243 :
        wcont = wcont[wcont.short_label.ne("S2243-0935")].reset_index()
    return(wcont)
    
def getlist_labels(mage_mode, labels, optional_file=False) :
    ''' Get the list of MagE spectra and redshifts, filtered by a list of short_labels (galaxy names)'''
    pspecs = getlist(mage_mode, optional_file) 
    filtlist = pspecs[pspecs['short_label'].isin(labels)].reset_index()
    return(filtlist)

def convert_spectrum_to_restframe(sp, zz) :
    (rest_wave, rest_fnu, rest_fnu_u)         = spec.convert2restframe(sp.wave, sp.fnu,  sp.fnu_u,  zz, 'fnu')
    sp['rest_wave']   = pandas.Series(rest_wave)
    sp['rest_fnu']    = pandas.Series(rest_fnu)
    sp['rest_fnu_u']  = pandas.Series(rest_fnu_u)
    (rest_wave, rest_flam, rest_flam_u) = spec.convert2restframe(sp.wave, sp.flam,  sp.flam_u,  zz, 'flam')
    sp['rest_flam']    = pandas.Series(rest_flam)
    sp['rest_flam_u']  = pandas.Series(rest_flam_u)
    if 'fnu_cont' in sp :
        (junk   , rest_fnu_cont, rest_fnu_cont_u) = spec.convert2restframe(sp.wave, sp.fnu_cont, sp.fnu_cont_u, zz, 'fnu')
        sp['rest_fnu_cont']   = pandas.Series(rest_fnu_cont)
        sp['rest_fnu_cont_u'] = pandas.Series(rest_fnu_cont_u)
        (junk   , rest_flam_cont, rest_flam_cont_u) = spec.convert2restframe(sp.wave, sp.flam_cont, sp.flam_cont_u, zz, 'flam')
        sp['rest_flam_cont']   = pandas.Series(rest_flam_cont)
        sp['rest_flam_cont_u'] = pandas.Series(rest_flam_cont_u)    
    return(0)  # acts directly on sp.  
    
def open_spectrum(infile, zz, mage_mode) :
    '''Reads a reduced MagE spectrum, probably ending in *combwC1.txt
      Inputs:   filename to read in, systemic redshift (to convert to rest-frame), and mage_mode
      Outputs:  the object spectrum, in both flam and fnu (why not?)  plus continuum both ways, all in Pandas data frame
      Pandas keys:  wave, fnu, fnu_u, wave_sky, fnu_sky, fnu_cont, fnu_cont_u, flam, flam_u, flam_sky, flam_cont, flam_cont_u
      call:  (Pandas_spectrum_dataframe, spectral_resolution) = jrr.mage.open_spectrum(infile, zz, mage_mode)
    '''
    (spec_path, line_path) = getpath(mage_mode)
    specdir = spec_path

    if 'RESOLUTIONGOESHERE' in open(specdir+infile).read() :  # spectral_resoln hasn't set resln yet
        resoln = -99
    else :
        command = "grep resolution " + specdir+infile   
        resoln = float(check_output(command, shell=True).split()[1])
        dresoln = float(check_output(command, shell=True).split()[3])

    if search("wC1", infile) :  hascont = True # The continuum exists        
    else :  hascont=False   # file lacks continuum, e.g. *comb1.txt

    sp =  pandas.read_table(specdir+infile, delim_whitespace=True, comment="#", header=0)#, names=names)
    sp.rename(columns= {'noise'  : 'fnu_u'}, inplace=True)
    sp.rename(columns= {'avgsky' : 'fnu_sky'}, inplace=True)
    sp.rename(columns= {'obswave' : 'wave_sky'}, inplace=True)
    sp['disp'] = sp.wave.diff()  # dispersion, in Angstroms
    sp['disp'].iloc[0] = sp['disp'][1] # first value will be nan
    
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
    sp['fnu_autocont'] = pandas.Series(np.ones_like(sp.wave)*np.nan)  # Will fill this with automatic continuum fit
    flag_skylines(sp)    # Flag the skylines.  This modifies sp.badmask
    sp.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace any inf values with nan
    flag_huge_uncert(sp, 0.5)  # flag uncertainties >0.5

    convert_spectrum_to_restframe(sp, zz)
    sp['rest_disp'] = sp['disp'] / (1.0+zz)
    return(sp, resoln, dresoln)   # Returns the spectrum as a Pandas data frame, the spectral resoln as a float, and its uncertainty

def open_S99_spectrum(rootname, zz) :
    ''' Reads a best-fit Starburst99 spectrum generated by John Chisholm.
    Outputs: a pandas data frame that contains some of the columns that open_spectrum makes.
    '''
    mage_mode = "released"
    (spec_path, line_path) = getpath(mage_mode)
    S99path = spec_path + "../Contrib/S99/"
    S99file = S99path + rootname + '-sb99-fit.txt'
    sp =  pandas.read_table(S99file, delim_whitespace=True, comment="#", names=('wave', 'data_fnu', 'data_fnu_u', 's99_fnu'))
    sp['fnu'] = sp['s99_fnu']      # Ayan's EW fitter is looking for fnu, so put the s99 spectrum into fnu
    sp['fnu_u'] = sp['fnu']  * 0.01 # hardcoded, feeding a dummy fnu_u that is 1% of fnu
    sp['badmask'] = False
    sp['flam']     = spec.fnu2flam(sp.wave, sp.fnu)          # convert fnu to flambda
    sp['flam_u']   = spec.fnu2flam(sp.wave, sp.fnu_u)
    linelist = line_path + "stacked.linelist"
    (LL, z_sys) = get_linelist(linelist)  #z_syst should be zero here.    
    auto_fit_cont(sp, LL, zz)
    return(sp)

def dict_of_stacked_spectra(mage_mode) :
    ''' Returns:  a) main directory of stacked spectra,  b) a list of all the stacked spectra.'''
    (spec_path, line_path)  = getpath(mage_mode)
    if mage_mode == "reduction" :
        indir = spec_path + "../Analysis/Stacked_spectra/"
    elif mage_mode == "released" :
        indir = spec_path + "../Stacked/"
    files = glob.glob(indir+"*spectrum.txt")
    justfiles = [re.sub(indir, "", file) for file in files]         # list comprehension
    short_stackname = [re.sub("magestack_", "", file) for file in justfiles]         # list comprehension
    short_stackname = [re.sub("_spectrum.txt", "", file) for file in short_stackname]         # list comprehension
    stack_dict =  dict(zip(short_stackname, justfiles))
    return(indir, stack_dict)
        
def open_stacked_spectrum(mage_mode, alt_infile=False) :
    #call: (restwave, X_avg, X_clipavg, X_median, X_sigma, X_jack_std, Ngal)= jrr.mage.open_stacked_spectrum(mage_mode)
    (indir, stack_dict) = dict_of_stacked_spectra(mage_mode)
    if alt_infile :
        print "Caution, I am using ", alt_infile, "instead of the default stacked spectrum."
        infile = indir + alt_infile
    else :
        infile = indir + "magestack_bystars_standard_spectrum.txt"
    sp =  pandas.read_table(infile, delim_whitespace=True, comment="#", header=0)
    sp['X_avg'] = sp['X_avg'].astype(np.float64)   #force to be a float, not a str
    sp['X_avg_flam']      = spec.fnu2flam(sp['restwave'], sp['X_avg'])
    sp['X_clipavg_flam']  = spec.fnu2flam(sp['restwave'], sp['X_clipavg'])
    sp['X_median_flam']   = spec.fnu2flam(sp['restwave'], sp['X_median'])
    sp['X_sigma_flam']    = spec.fnu2flam(sp['restwave'], sp['X_sigma'])
    sp['X_jack_std_flam'] = spec.fnu2flam(sp['restwave'], sp['X_jack_std'])
    return(sp) # return the Pandas DataFrame containing the stacked spectrum

def open_Crowther2016_spectrum() :
    print "STATUS:  making velocity plots of the stacked Crowther et al. 2016 spectrum"
    infile = "/Volumes/Apps_and_Docs/WORK/Lensed-LBGs/Mage/Lit-spectra/Crowther2016/r136_stis_all.txt" # on satchmo
    sp =  pandas.read_table(infile, delim_whitespace=True, comment="#", header=0)  # cols are wave, flam
    sp['fnu'] = spec.fnu2flam(sp['wave'], sp['flam'])
    return(sp)
        
def plot_1line_manyspectra(line_cen, line_label, win, vel_plot=True, mage_mode="reduction", specs=[]) :
    ''' Plot one transition, versus (wavelength or velocity?), for many MagE spectra
    line_cen:     rest-wavelength of line to plot, in Angstroms.
    line_label:   label for that line
    win:          window around that line.  If vel_plot, then units are km/s.  If not vel_plot, units are rest-frame Angstroms
    vel_plot:     (optional, Boolean): If True, x-axis is velocity.  If False, x-axis is wavelength.
    specs:        (optional) Pandas dataframe of MagE filenames & redshifts.'''
    if len(specs) == 0 :
        (specs) = getlist_wcont(mage_mode)  # Default grabs all MagE spectra w continuum fits
    Nspectra = len(specs)
    Ncol = 1
    Nrow = np.ceil(Nspectra / Ncol)  # Calculate how many rows to generate
    fig = plt.figure(figsize=(5,16))
    
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
            plt.xlim(line_cen - win, line_cen + win)
        plt.plot( (-1*win, win), (1.0,1.0), color=color3)  # Already normalized by continuum, so plot unity continuum. 
        plt.ylim(0.0, 1.5)  # May need to change these limits
        if ii == (Nspectra - 1) :  # if last subplot, make xlabel
            plt.annotate( line_label, (0.2,0.1), xycoords="axes fraction")
            if vel_plot :
                plt.xlabel("rest-frame velocity (km/s)")  
            else :
                plt.xlabel(r'rest-frame wavelength($\AA$)')                                
        else :
            ax.axes.xaxis.set_ticklabels([])  # if not last subplot, suppress  numbers on x axis
        plt.annotate( label, (0.2,0.8), xycoords="axes fraction")
        plt.annotate("z="+str(zz), (0.75,0.1), xycoords="axes fraction")
        #plt.ylabel("continuum-normalized fnu")
#        fig.(hspace=0)
    return(0)

def get_linelist_name(filename, line_path) :
    ''' Given a spectrum filename, guess what its associated linelist should be.  A convenience function.'''
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

    # Grab the systemic redshift, too
    command = "grep SYSTEMIC " + linelist
    z_systemic = float(check_output(command, shell=True).split()[3])
    #print "Loaded linelist ", linelist, " with systemic redshift of ", z_systemic
    return(L, z_systemic)  # Returning a pandas data frame of the linelist.  Key cols are restwav, obswav, zz, lab1 (the label)

def plot_linelist(L_all, z_systemic=np.nan, restframe=False, velplot=False, line_center=np.nan) :  
    '''Plot ticks and line labels for lines from a linelist, loaded by get_linelist.
    Can do either rest-frame or observed frame, x-axis wavelength or velocity
    '''
    #temp = plt.ylim()[1] * 0.93  # previous
    temp = (plt.ylim()[1] - plt.ylim()[0]) * 0.15    #trying moving ticks to bottom
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
        plt.annotate(L.lab1[i], xy=(wav2plot[i], temp*3), xytext=(wav2plot[i], temp), color=L.color[i], rotation=90, ha='center', va='center', arrowprops=dict(color=L.color[i], shrink=0.2,width=0.5, headwidth=0))
    L = []
    L_all = []
    return(0)
    
def open_Leitherer_2011_stack() :
    # call:  (restwave, avg_flux, median_flux) = open_Leitherer_2011_stack() :
    infile = "/Volumes/Apps_and_Docs/WORK/Lensed-LBGs/Mage/Lit-spectra/Leitherer_2011/fos_ghrs_composite.txt"
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

def mage_boxplot_Nspectra(thewaves, thefnus, thedfnus, thezs, line_label, line_center, win, Ncol, LL, extra_label="",figsize=(8,16), vel_plot=True, plot_xaxis=True, ylims=(0.0,1.5)) :
    '''Plot flux density versus rest-frame velocity or rest-frame wavelength for several spectral lines,
    in a [Nrow x Ncol] box.  CAN PLOT MULTIPLE SPECTRA IN EACH BOX.
    Inputs are:
    thewaves:        tuple of arrays of observed wavelength (Angstrom).  If only 1 spectrum, use thewaves=(wav_ar,) to keep as tuple.
    thefnus:         tuple of arrays of cont-normalized flux density (erg/s/cm^2/Hz)  ** should this be flambda instead?.
    thedfnus:        tuple of arrays of 1 sigma uncertainty on fnu
    thezs:           tuple of arrays of redshifts, to convert to rest-frame
    line_label:      tuple of line labels.  Makes one box per line
    line_center:     np array of rest-frame wavelengths (Angstroms).  Makes one box per line
    win:             window to use (km/s if vel_plot; if not, wavelength units.)
    Ncol:            number of columns to plot.  Number of rows set automatically.
    LL:              The linelist, a Pandas data frame, read in by jrr.mage.get_linelist(linelist).
    extra_label:     (Optional) Extra text to annotate to lower panel.
    figsize:         (Optional) Figure size in inches.
    vel_plot:        (Optional, Bool) If True, x-axis is velocity.  If False, x-axis is wavelength.
    plot_xaxis:      (Optional, Bool) Plot the xaxis?
    ylims:           (Optional, tuple) ylim, over-rides default
         thewaves is now a tuple? of wavelength arrays.  same for thefnus, the dfnus, thezs
    If only plotting one spectrum, still need input format to be tuples, as in thewaves=(wave_array,).'''
    
    mycol = ['black', 'blue', 'green', 'purple', 'red', 'orange', 'cyan']
    linestyles = ['solid'] #, 'dashed', 'dotted']#, 'dashdot']
    plt.rc('axes', prop_cycle=(cycler('color', mycol) * cycler('linestyle', linestyles)))  # fix stupid colors after lunch.**
    
    Nrow = int(np.ceil( float(line_center.size) / Ncol))  # Calculate how many rows to generate
    print "DEBUGGING ", Nrow, Ncol, len(line_center)
    fig = plt.figure(figsize=figsize)

    for ii, dum in enumerate(line_label) :
        print "    Plotting ", line_label[ii], " at ", line_center[ii]
        ax = fig.add_subplot(Nrow, Ncol, ii+1)
        plt.annotate( line_label[ii], (0.3,0.9), xycoords="axes fraction")
        if(vel_plot) :
            for ss in range(0, len(thewaves)) :  # For each spectrum to overplot
                restwave = thewaves[ss] / (1.0 + thezs[ss])
                vel = spec.convert_restwave_to_velocity(restwave, line_center[ii])   # velocity in km/s
                in_window = vel.between(-1*win, win)
                plt.step(vel[in_window], thefnus[ss][in_window], color=mycol[ss])   # assumes input is continuum-normalized
                plt.step(vel[in_window], thedfnus[ss][in_window], color=mycol[ss])  # plot uncertainty
            plt.plot( (0., 0.), (0.0,2), color=color2, linewidth=2)  # plot tics at zero velocity
            plt.xlim(-1*win, win)
        else :
            for ss in range(0, len(thewaves)) :  # For each spectrum to overplot
                restwave = thewaves[ss] / (1.0 + thezs[ss])
                in_window = restwave.between((line_center[ii] - win), (line_center[ii] + win))
                plt.step(restwave[in_window], thefnus[ss][in_window], color=mycol[ss])
                plt.step(restwave[in_window], thedfnus[ss][in_window], color=mycol[ss])
            plt.plot( (line_center[ii], line_center[ii]), (0.0,2), color=color3, linewidth=2)  # plot tics at zero velocity
            plt.xlim(line_center[ii] - win, line_center[ii] + win)
        plt.ylim(ylims[0], ylims[1])  # May need to change these limits
        plot_linelist(LL, thezs[0], True, vel_plot, line_center[ii])  # plot the line IDs for the first spectrum only
     # plot_linelist(L_all, z_systemic=0.0, restframe=False, velplot=False, line_center=0.0) :  
        if ii == len(line_label) -1 :
            plt.annotate(extra_label, (0.6,0.1), xycoords="axes fraction")
            if vel_plot : 
                plt.xlabel("rest-frame velocity (km/s)")  # if last subplot, make xlabel
            else :
                plt.xlabel(r'rest-frame wavelength($\AA$)')
        if (not plot_xaxis) and (ii < len(line_label)-1) :
            ax.axes.xaxis.set_ticklabels([])  # if not last subplot, suppress  numbers on x axis
            # But warning, this will disable x= on interactive matplotlib.  comment out above line to measure numbers interactively on graph
        if not plot_xaxis :
            fig.subplots_adjust(hspace=0)

                    
def flag_skylines(sp) :
    # Mask skylines [O I] 5577\AA\ and [O I]~6300\AA,
    skyline = (5577., 6300.)
    skywidth = 17.0  # flag spectrum +- skywidth of the skyline
    for thisline in skyline :
        sp.badmask.loc[sp['wave'].between(thisline-skywidth, thisline+skywidth)] = True 
    return(0)  # This function works directly on sp

def flag_huge_uncert(sp, thresh=0.5, crazy_high=30000) :
    # Set mask for pixels that have crazy high uncertainty.  These were flagged further upstream, where i had no mask.
    # Inputs:  sp is spectrum as Pandas data frame
    #          thresh is threshold of uncertainty values to flag.
    # Output:  None.  It directly acts on sp.badmask
    sp.badmask.loc[sp.fnu_u.ge(thresh)] = True    # these are fluxed spectra, so normal values are 1E-28 or so.
    sp.badmask.loc[sp.fnu.eq(-0.999)]   = True  # a flag from upstream
    sp.badmask.loc[sp.fnu.gt(sp.fnu.median() * crazy_high)] = True  # flag pixels with crazy high flux.
    sp.badmask.loc[sp.fnu.lt(sp.fnu.median() * crazy_high * -1)] = True  # flag pixels with crazy high flux.
    return(0)

def flag_where_nocont(sp) :
    sp.badmask.loc[sp['fnu_cont'].eq(9999)] = True  # where hand-drawn continuum is undefined
    return(0)

def flag_near_lines(sp, LL, zz, vmask) :
    # Flag regions within +- vmask km/s around lines in linelist LL
    # Inputs:   sp, the spectrum as Pandas data frame
    #           LL, the linelist as pandas data frame
    #           zz, the redshift.  
    #           vmask, the velocity around which to mask each line +-, in km/s
    # Outputs:  None.  It acts directly on sp.linemask
    #print "Flagging regions near lines."
    rest_cen = np.array( LL.restwav)
    line_lo   = rest_cen * (1.0 - vmask/2.997E5) * (1. + LL.zz)
    line_hi   = rest_cen * (1.0 + vmask/2.997E5) * (1. + LL.zz)
    temp_wave = np.array(sp.wave)
    temp_mask = np.zeros_like(temp_wave).astype(np.bool)
    for ii in range(0, len(rest_cen)) :    # doing this in observed wavelength
        temp_mask[np.where( (temp_wave > line_lo[ii]) & (temp_wave < line_hi[ii]))] = True
    sp['linemask'] = temp_mask  # Using temp numpy arrays is much faster than writing repeatedly to the pandas data frame
    return(0)
    # To make this work for S99 synthetic spectra, should load S99 into a similar sp data frame, 
    # with zz=0 and sp.wave = the rest wavelength.

        
def auto_fit_cont(sp, LL, zz, vmask=500, boxcar=1001, flag_lines=True) : # Automatically fit the continuum
    ''' Automatically fits a smooth continuum to a spectrum.
     Inputs:  sp,  a Pandas data frame containing the spectra, opened by mage.open_spectrum or similar
              LL,  a Pandas data frame containing the linelist, opened by mage.get_linelist(linelist) or similar
              zz,  the systemic redshift. Used to set window threshold around lines to mask
              vmask (Optional), velocity around which to mask lines, km/s. 
              boxcar (Optional), the size of the boxcar smoothing window, in pixels
              flag_lines (Optional) flag, should the fit mask out known spectral features? Can turn off for debugging
    '''
    # First, mask out big skylines. Done by mage.flag_skylines, which is called by mage.open_spectrum
    # Second, flag regions with crazy high uncertainty.  done in flag_huge_uncert, called by mage.open_spectrum
    # Third, mask out regions near lines.  Flagged in sp.linemask
    if flag_lines : flag_near_lines(sp, LL, zz, vmask)
    # Populate fnu_autocont with fnu, unless pixel is bad or has a spectral feature, in which case it stays nan.
    temp_fnu = sp.fnu.copy(deep=True)
    temp_fnu.loc[(sp.badmask | sp.linemask).astype(np.bool)] = np.nan
    # Last, smooth with a boxcar
    smooth1 = astropy.convolution.convolve(np.array(temp_fnu), np.ones((boxcar,))/boxcar, boundary='fill', fill_value=np.nan)
    sp['fnu_autocont'] = pandas.Series(smooth1)  # Write the smooth continuum back to data frame
    # Make derived products from fnu_autocont:    flam_autocont,  rest_flam_autocont
    (rest_wave, rest_fnu, rest_fnu_u)         = spec.convert2restframe(sp.wave, sp.fnu_autocont, sp.fnu_u,  zz, 'fnu')
    sp['rest_fnu_autocont']    = pandas.Series(rest_fnu)
    sp['flam_autocont']     = spec.fnu2flam(sp.wave, sp.fnu_autocont)
    (rest_wave, rest_flam, rest_flam_u) = spec.convert2restframe(sp.wave, sp.flam_autocont,  sp.flam_u,  zz, 'flam')    
    sp['rest_flam_autocont'] = pandas.Series(rest_flam)
    return(0)


