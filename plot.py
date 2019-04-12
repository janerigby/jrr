''' Functions to make plots.  At present, just plotting spectra.
jrigby, may 2016 '''
from __future__ import print_function

from builtins import range
from jrr import spec
from jrr import mage
from jrr import util
import pandas
from   math import ceil
from   matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import numpy as np
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages

color1 = 'k'     # color for spectra
color2 = '0.65'  # color for uncertainty spectra
color3 = '0.5'   # color for continuum

def annotate_from_dataframe(df, xcol='x', ycol='y', text='label', xycoords='data', xytext=(4,3), ha='right', fontsize=10, fontname='Times New Roman') :
    # Matplotlib annotate works one annotation at a time -- it can't handle arrays.  Workaround: feed it a dataframe.
    for index, row in df.iterrows():
         plt.annotate(row[text], xy=(row[xcol], row[ycol]), xycoords=xycoords, xytext=xytext, textcoords="offset points", ha=ha, fontsize=fontsize, fontname=fontname)
    return(0)

def standard_colors1():
    return(['black', 'blue', 'green', 'purple', 'red', 'orange', 'cyan'])
def standard_colors2():
#    return(['black', 'midnightblue', 'dimgray', 'lightgrey'])
    return(['black', 'blue', 'dimgray', 'lightgrey'])
def standard_colors3():
    return(['blue', 'black', 'red'])
def standard_colors4():
    return([color1, color2, color3])

        
def boxplot_spectra(wave, fnu, dfnu, line_label, line_center, redshift, win, Ncol, spec_label="",figsize=(8,8), vel_plot=True, verbose=True) :
    '''Make a plot of flux density versus rest-frame velocity or rest-frame wavelength for several spectral lines, in a
    [Nrow x Ncol] box.
    Inputs are:
    wave:        observed wavelength (currently Angstrom)
    fnu:         cont-normalized flux density (erg/s/cm^2/Hz) 
    dfnu:        1 sigma uncertainty on fnu
    line_label:  tuple of line labels
    line_center: np array of rest-frame wavelengths (Angstroms)
    redshift:    redshift to assume,
    win:         window to use (km/s if vel_plot; if not, wavelength units.)
    Ncol:        number of columns to plot.  Number of rows set automatically.
    spec_label:  Name of spectrum to plot in lower panel.  May be blank
    figsize:     (Optional) Figure size in inches.
    vel_plot:    (Optional, Boolean):  If True, x-axis is velocity.  If False, x-axis is wavelength.
    '''
    Nrow = ceil(line_center.size / Ncol)  # Calculate how many rows to generate
    fig = plt.figure(figsize=figsize)
    restwave = wave / (1.0 + redshift)

    for ii, dum in enumerate(line_label) :
        if verbose : print("    Plotting ", line_label[ii], " at ", line_center[ii])
        ax = fig.add_subplot(Nrow, Ncol, ii+1)
        plt.annotate( line_label[ii], (0.3,0.9), xycoords="axes fraction")
        if(vel_plot) :
            vel = spec.convert_restwave_to_velocity(restwave, line_center[ii])   # velocity in km/s
            in_window = vel.between(-1*win, win)
            plt.step(vel[in_window], fnu[in_window], color=color1)   # assumes input is continuum-normalized
            plt.step(vel[in_window], dfnu[in_window], color=color2)  # plot uncertainty
            plt.plot( (-1*win, win), (1.0,1.0), color=color3)        # plot unity continuum
            plt.plot( (0., 0.), (0.0,2), color=color2, linewidth=2)  # plot tics at zero velocity
            plt.ylim(0.0, 1.5)  # May need to change these limits
            plt.xlim(-1*win, win)
        else :
            in_window = restwave.between((line_center[ii] - win), (line_center[ii] + win))
            plt.step(restwave[in_window], fnu[in_window],  color=color1)
            plt.step(restwave[in_window], dfnu[in_window], color=color2)
            plt.plot( (line_center[ii] - win, line_center[ii] + win), (1., 1.), color=color3)  # plot unity continuum
            plt.plot( (line_center[ii], line_center[ii]), (0.0,2), color=color3, linewidth=2)  # plot tics at zero velocity
            plt.xlim(line_center[ii] - win, line_center[ii] + win)
        if ii == len(line_label) -1 :
            if vel_plot : 
                plt.xlabel("rest-frame velocity (km/s)")  # if last subplot, make xlabel
            else :
                plt.xlabel(r'rest-frame wavelength($\AA$)')
        else :
            ax.axes.xaxis.set_ticklabels([])  # if not last subplot, suppress  numbers on x axis
        fig.subplots_adjust(hspace=0)

def boxplot_Nspectra(thewaves, thefnus, thedfnus, thezs, line_label, line_center, spec_label=(), win=2000, Ncol=1, LL=(), extra_label="",figsize=(8,16), vel_plot=True, plot_xaxis=True, ymax=(), colortab=False, verbose=True, drawunity=False, label_loc=(0.55,0.85), lw=(1,)) :
    '''Plot flux density versus rest-frame velocity or rest-frame wavelength for several spectral lines,
    in a [Nrow x Ncol] box.  CAN PLOT MULTIPLE SPECTRA IN EACH BOX.
    Inputs are:
    thewaves:        tuple of arrays of observed wavelength (Angstrom).  If only 1 spectrum, use thewaves=(wav_ar,) to keep as tuple.
    thefnus:         tuple of arrays of cont-normalized flux density (erg/s/cm^2/Hz)  ** should this be flambda instead?.
    thedfnus:        tuple of arrays of 1 sigma uncertainty on fnu
    thezs:           tuple of arrays of redshifts, to convert to rest-frame
    line_label:      tuple of line labels.  Makes one box per line
    line_center:     np array of rest-frame wavelengths (Angstroms).  Makes one box per line
    spec_label:      (Optional), labels to name the spectra in a legend
    win:             window to use (km/s if vel_plot; if not, wavelength units.)  Format can be scalar, interpreted as +-win, or list: (-300, 300)  
    Ncol:            number of columns to plot.  Number of rows set automatically.
    LL:              (Optional) Linelist, a Pandas data frame, read in by jrr.mage.get_linelist(linelist), to mark other lines
    theconts:        (Optional, tuple of continuum fits)
    extra_label:     (Optional) Extra text to annotate to lower panel.
    figsize:         (Optional) Figure size in inches.
    vel_plot:        (Optional, Bool) If True, x-axis is velocity.  If False, x-axis is wavelength.
    plot_xaxis:      (Optional, Bool) Plot the xaxis?
    ymax:            (Optional), array of max y values to use in subplots.  Over-rides auto-scaling. Size must == line_center
    colortab:        (Optional) color table to use, to replace default colortab
    verbose:         (Optional) Verbose or terse account of status?
    drawunity:       (Optional), draw a line at unity?  Good for spectra that are already continuum-normalized
    label_loc:       (Optional), where to draw the line label for each boxplot.  By default, upper right
    lw:              (Optional), scalar or tuple of linewidths.  If scalar, same lw for all
    thewaves is now a tuple? of wavelength arrays.  same for thefnus, the dfnus, thezs
    If only plotting one spectrum, still need input format to be tuples, as in thewaves=(wave_array,).'''

    if colortab :  mycol=colortab
    else : mycol = standard_colors1()
    linestyles = ['solid'] #, 'dashed', 'dotted']#, 'dashdot']
    plt.rc('axes', prop_cycle=(cycler('color', mycol) * cycler('linestyle', linestyles))) 
    
    Nrow = int(np.ceil( float(len(line_center)) / Ncol))  # Calculate how many rows to generate
    #print "DEBUGGING ", Nrow, Ncol, len(line_center)
    fig = plt.figure(figsize=figsize)

    if len(spec_label) == len(thewaves) :  pass  #print "DEBUGGING, adding spectrum label to legend"
    else : spec_label = np.repeat('_nolegend_', len(thewaves))  # override the legend
    
    if hasattr(win, "__len__") and len(win) == 2:
        win0 = win[0]  ; win1 = win[1]
    elif not hasattr(win, "__len__")  :    win0 = -1*win ; win1=win
    else : raise Exception("ERROR: Window width win must either be a scalar, or a two- element list/tuple/array")
        
    if len(lw) == 1 : lw = lw * np.ones(shape=len(thewaves))  # if lw is a scalar, change to an array, same for all
    elif len(lw) != len(thewaves) : raise Exception('Error, linewidth lw must be scalar (len=1) or tuple with same sizes as thewaves')
    for ii, dum in enumerate(line_label) :
        if verbose : print("    Plotting ", line_label[ii], " at ", line_center[ii])
        ax = fig.add_subplot(Nrow, Ncol, ii+1)
        plt.annotate( line_label[ii], label_loc, xycoords="axes fraction")
        max_in_window = 0.
        if(vel_plot) :
            for ss in range(0, len(thewaves)) :  # For each spectrum to overplot
                restwave = thewaves[ss] / (1.0 + thezs[ss])
                vel = spec.convert_restwave_to_velocity(restwave, line_center[ii])   # velocity in km/s
                in_window = vel.between(win0, win1)
                plt.step(vel[in_window], thefnus[ss][in_window], color=mycol[ss], linewidth=lw[ss], label=spec_label[ss])
                if len(thedfnus[ss]) :
                    plt.step(vel[in_window], thedfnus[ss][in_window], color=mycol[ss], linewidth=1, label='_nolegend_')  # plot uncertainty
                thismax = thefnus[ss][in_window].max()
                max_in_window =  util.robust_max((thismax, max_in_window))
            plt.plot( (0., 0.), (0.0,2), color=color2, linewidth=2)  # plot tics at zero velocity
            if drawunity: plt.plot( (win0, win1), (1.0,1.0), color=color3)        # plot unity continuum
            plt.xlim(win0, win1)
        else :
            for ss in range(0, len(thewaves)) :  # For each spectrum to overplot
                restwave = thewaves[ss] / (1.0 + thezs[ss])
                in_window = restwave.between((line_center[ii] + win0), (line_center[ii] + win1))
                plt.step(restwave[in_window], thefnus[ss][in_window], color=mycol[ss], linewidth=lw[ss], label=spec_label[ss])
                if len(thedfnus[ss]) :
                    plt.step(restwave[in_window], thedfnus[ss][in_window], color=mycol[ss], linewidth=1, label='_nolegend_')
                thismax = thefnus[ss][in_window].max()
                max_in_window =  util.robust_max((thismax, max_in_window))
            plt.plot( (line_center[ii], line_center[ii]), (0.0,100), color=color3, linewidth=2)  # plot tics at zero velocity
            if drawunity: plt.plot( (line_center[ii] + win0, line_center[ii] + win1), (1.0,1.0), color=color3,  label='_nolegend_')  # plot unity continuum
            plt.xlim(line_center[ii] +win0, line_center[ii] + win1)
        ax.locator_params(axis='y', nbins=4)
        ax.locator_params(axis='x', nbins=7)
        if len(ymax)  == 1 :  plt.ylim(0, ymax[0])   # user can over-ride autoscaling
        elif len(ymax) > 1 :  plt.ylim(0, ymax[ii])  # user can over-ride autoscaling 
        else :                plt.ylim(0., max_in_window*1.1)  
        if len(LL) :
            mage.plot_linelist(LL, thezs[0], True, vel_plot, line_center[ii])  # plot the line IDs for the first spectrum only
        if ii == len(line_label) -1 :
            plt.annotate(extra_label, (0.6,0.1), xycoords="axes fraction")
            if vel_plot : 
                plt.xlabel("rest-frame velocity (km/s)")  # if last subplot, make xlabel
            else :
                plt.xlabel(r'rest-frame wavelength ($\rm \AA$)')
        if (not plot_xaxis) and (ii < len(line_label)-1) :
            ax.axes.xaxis.set_ticklabels([])  # if not last subplot, suppress  numbers on x axis
            # But warning, this will disable x= on interactive matplotlib.  comment out above line to measure numbers interactively on graph
        if not plot_xaxis :
            fig.subplots_adjust(hspace=0)


def velocity_overplot(wave, fnu, line_label, line_center, redshift, vwin1, vwin2, figsize=(8,8), colortab=False) :
    '''Same as boxplot_spectra(vel_plot=True), but OVERPLOT all the lines on 1 plot, no subplots'''
    if colortab :  mycol=colortab
    else : mycol = standard_colors1()
    restwave = wave / (1.0 + redshift)
    plt.figure(figsize=figsize)
    for ii, dum in enumerate(line_label) :
        print(ii, "Trying to plot ", line_label[ii], " at ", line_center[ii])
        vel = spec.convert_restwave_to_velocity(restwave, line_center[ii])   # velocity in km/s
        in_window = vel.between((vwin1), vwin2)
        plt.step(vel[in_window], fnu[in_window], color=mycol[ii], label=line_label[ii])
        plt.xlim(vwin1, vwin2)
        plt.ylim(0, 1.4)
        xy = (0.1,  0.26 - float(ii)/len(line_label)*0.3)
        #print "DEBUG, xy is ", xy
        #plt.annotate( line_label[ii], xy, xycoords="axes fraction", color=hist.get_color())
        plt.legend()
        plt.xlabel("rest-frame velocity (km/s)")
    plt.plot( (0.,0.), plt.ylim(),  color=color2, linewidth=2)  # plot tics at zero velocity
    plt.plot( plt.xlim(), (1.0,1.0), color=color2)  # plot unity continuum

def annotate_echelle() :
    plt.annotate("f_nu (erg/s/cm^2/Hz)",  xy=(0.05,0.055), color="black", xycoords="figure fraction")
    plt.annotate("Line color coding:", xy=(0.05,0.043), color="black", xycoords="figure fraction")
    plt.annotate("Photosphere", xy=(0.05,0.03), color="blue", xycoords="figure fraction")
    plt.annotate("Emission",    xy=(0.17,0.03), color="red", xycoords="figure fraction")
    plt.annotate("ISM",         xy=(0.25,0.03), color="green", xycoords="figure fraction")
    plt.annotate("Wind",        xy=(0.32,0.03), color="cyan", xycoords="figure fraction")
    plt.annotate("Fine structure", xy=(0.4,0.03), color="purple", xycoords="figure fraction")
    plt.annotate("Intervening", xy=(0.55,0.03), color="orange", xycoords="figure fraction")

def echelle_spectrum(the_dfs, the_zzs, LL=(), Npages=4, Npanels=24, plotsize=(11,16), outfile="multipanel.pdf", title="", norm_by_cont=False, plot_cont=False, apply_bad=False, waverange=(), colwave='wave', colfnu='fnu', colfnu_u='fnu_u', colcont='fnu_autocont', colortab=False, plotx2=True, ylim=(), topfid=(1.1,0.3), annotate=annotate_echelle, verbose=False) :
    ''' Plot an echelle(ette) spectrum with several pages and many panels for page.  Based on multipanel-spectrum-ids.py,
    but modular. and with more features.  Inputs:
    the_dfs:    an array of dataframes containing spectra, to plot. Usually this is just one dataframe, one spectrum,
                but want to allow overplotting of multiple spectra.  First spectrum will be used to set the wavelength range.
    the_zzs:    an array of redshifts
    LL:         (Optional). a linelist to plot features.  At present, takes redshift from the_zzs[0].
    Npages:     pages of plots in the output PDF.  Will split evenly.
    Npanels:    Total number of panels to plot, across all pages
    plotsize:   (Optional) Size of the plot
    outfile:    Name of output PDF file.
    norm_by_cont: (Optional), Normalize by continuum?
    plot_cont:  (Optional) Plot the continuum, too?
    apply_bad:  (Optional) Apply bad pixel map?
    waverange:  (Optional) Only plot this wavelength range, rather than full range. Format is (wave_lo, wave_hi).
    colwave, colfnu, colfnu_u, colcont:  columns in the dataframes to use for wave, fnu, fnu_uncert, continuum
    colortab:   (Optional) color table to use, to replace default colortab
    plotx2:     Plot upper x-axis, the rest-frame wavelength?  Turn off when lower x is already rest-frame
    ylim:       (Optional), over-ride my auto-sensing of the ylims for each subplot
    topfid:     (Optional) fiddle parameters to adjust tightness of y axis.  first scales the median, second scales the IQR
    annotate:   (Optional) Function that will annotate the first page.  For plot customization.'''
    
    contcolor = '0.75'
    if colortab :  linecolor=colortab
    else : linecolor = standard_colors2()
    pp = PdfPages(outfile)  # output
    # Set up the wavelength ranges.  Each panel has same # of pixels, not same dlambda, since dispersion may change.
    if not len(waverange) in (0, 2) : raise Exception("Error: len(waverange) must be 0 or 2")
    if len(waverange) :  # If user selected wavelength range, modify index, but keep same # of pixels per panel
        first_index = the_dfs[0][the_dfs[0][colwave].between(waverange[0], waverange[0]*1.01)][colwave].index[0]
        last_index  = the_dfs[0][the_dfs[0][colwave].between(waverange[1], waverange[1]*1.01)][colwave].index[-1]
    else : 
        first_index = 0
        last_index  = len(the_dfs[0][colwave])-1
    ind = (np.linspace(first_index, last_index, num=Npanels+1)).astype(np.int)
    start = the_dfs[0][colwave][ind[0:-1]].values
    end   = the_dfs[0][colwave][ind[1:]].values

    max_per_page = np.int(np.float64(Npanels) / Npages)
    the_format = max_per_page*100 + 11  # makes label: 311 for 3 rows, 1 col, start at 1

    for df in the_dfs :
        if norm_by_cont :   df['normby'] = df[colcont]  # easy to plot whether or not it's continuum normalized
        else :              df['normby'] = 1.0
    for kk in range(0, Npanels) :
        if kk % max_per_page == 0 :   # Start a new page
            current_page = kk /  max_per_page + 1  # should be int division
            print("Starting page", current_page, "of", Npages)
            fig    = plt.figure(figsize=plotsize)
        subit = host_subplot(the_format + kk % max_per_page)
        if verbose : print("Plotting panel", kk, "for wavelengths", start[kk], "to",end[kk])
        for ss, df in enumerate(the_dfs) :
            if apply_bad : subset = df[df[colwave].between(start[kk], end[kk]) & ~df['badmask']]
            else:          subset = df[df[colwave].between(start[kk], end[kk])]

            plt.plot(subset[colwave], subset[colfnu]/subset['normby'],   color=linecolor[ss],  linestyle='steps', linewidth=1)
            plt.plot(subset[colwave], subset[colfnu_u]/subset['normby'], color=linecolor[ss], linestyle='steps', linewidth=1) # plot 1 sigma uncert spectrum
            if plot_cont :  plt.plot(subset[colwave], subset[colcont].astype(np.float)/subset['normby'], contcolor, linestyle='steps', zorder=1, linewidth=1) # plot the continuum
            if ss == 0 :  # If the first df, set the plot ranges
                top = (subset[colfnu]/subset['normby']).median()*topfid[0] + util.IQR(subset[colfnu]/subset['normby'])*topfid[1]
                #print "DEBUGGING top", kk, top, (subset[colfnu]/subset['normby']).median(), + util.IQR(subset[colfnu]/subset['normby'])
        if len(ylim) == 2:   plt.ylim(ylim[0], ylim[1])
        else:                plt.ylim(0, top)  # trying to fix weird autoscaling from bad pixels
        plt.xlim(start[kk], end[kk])
        if len(LL) : mage.plot_linelist(LL, the_zzs[0])   # Plot the line IDs.  Use the first redshift for z.
        if plotx2: 
            upper = subit.twiny()  # make upper x-axis in rest wave (systemic, or of abs lines)
            upper.set_xlim( start[kk]/(1.0+the_zzs[0]), end[kk]/(1.0+the_zzs[0]))
            upper.locator_params(axis='x', nbins=5)
            subit.locator_params(axis='x', nbins=5)
            subit.xaxis.tick_bottom()  # don't let lower ticks be mirrored  on upper axis

        if  kk % max_per_page == (max_per_page-1) or kk == Npanels-1:   # last plot on this page
            if plotx2 : subit.set_xlabel(r"observed-frame vacuum wavelength ($\AA$)")  ## COMMENDED OUT FOR mage stack paper
            else:       subit.set_xlabel(r"rest-frame vacuum wavelength ($\AA$)")  # Just for mage stack paper
            #plt.ylabel('fnu') # fnu in cgs units: erg/s/cm^2/Hz
            plt.ylabel('relative flux') # **temp for stacked paper
            pp.savefig(bbox_inches='tight')    
            #fig.canvas.draw()
        if  kk % max_per_page == 0 and plotx2 :  # first plot on the page
            upper.set_xlabel(u"rest-frame vacuum wavelength ($\AA$)")
            
        if(kk == 0):  # first page
            if callable(annotate) : annotate()
            plt.suptitle(title)  # global title
    pp.close()
    print("   Generated PDF:  ", outfile)
    return(0)
