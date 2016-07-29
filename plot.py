''' Functions to make plots.  At present, just plotting spectra.
jrigby, may 2016 '''

from jrr import spec
import pandas
from   math import ceil
from   matplotlib import pyplot as plt
import numpy as np
from   matplotlib import pyplot as plt

color1 = 'k'     # color for spectra
color2 = '0.65'  # color for uncertainty spectra
color3 = '0.5'   # color for continuum
    
def boxplot_spectra(wave, fnu, dfnu, line_label, line_center, redshift, win, Ncol, spec_label="",figsize=(8,8), vel_plot=True) :
    '''Make a plot of flux density versus rest-frame velocity or rest-frame wavelength for several spectral lines, in a
    [Nrow x Ncol] box.
    Inputs are:
    wave:        observed wavelength (currently Angstrom)
    fnu:         cont-normalized flux density (erg/s/cm^2/Hz)  ** should this be flambda instead?.
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
        print "    Plotting ", line_label[ii], " at ", line_center[ii]
        ax = fig.add_subplot(Nrow, Ncol, ii+1)
        plt.annotate( line_label[ii], (0.3,0.9), xycoords="axes fraction")
        if(vel_plot) :
            vel = spec.convert_restwave_to_velocity(restwave, line_center[ii])   # velocity in km/s
            in_window = vel.between(-1*win, win)
            plt.step(vel[in_window], fnu[in_window], color=color1)   # assumes input is continuum-normalized
            plt.step(vel[in_window], dfnu[in_window], color=color2)  # plot uncertainty
            plt.plot( (-1*win, win), (1.0,1.0), color=color3)        # plot unity continuum. 
            plt.plot( (0., 0.), (0.0,2), color=color2, linewidth=2)  # plot tics at zero velocity
            plt.ylim(0.0, 1.5)  # May need to change these limits
            plt.xlim(-1*win, win)
        else :
            in_window = restwave.between((line_center[ii] - win), (line_center[ii] + win))
            plt.step(restwave[in_window], fnu[in_window],  color=color1)
            plt.step(restwave[in_window], dfnu[in_window], color=color2)
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

def velocity_overplot(wave, fnu, line_label, line_center, redshift, vwin1, vwin2, figsize=(8,8)) :
    '''Same as boxplot_spectra(vel_plot=True), but OVERPLOT all the lines on 1 plot, no subplots'''
    restwave = wave / (1.0 + redshift)
    plt.figure(figsize=figsize)
    for ii, dum in enumerate(line_label) :
        print "Trying to plot ", line_label[ii], " at ", line_center[ii]
        vel = spec.convert_restwave_to_velocity(restwave, line_center[ii])   # velocity in km/s
        in_window = vel.between((vwin1), vwin2)
        hist, = plt.step(vel[in_window], fnu[in_window])
        plt.xlim(vwin1, vwin2)
        plt.ylim(0, 1.4)
        xy = (0.1,  0.26 - float(ii)/len(line_label)*0.3)
        print "DEBUG, xy is ", xy
        plt.annotate( line_label[ii], xy, xycoords="axes fraction", color=hist.get_color())
        plt.xlabel("rest-frame velocity (km/s)")
    plt.plot( (0.,0.), plt.ylim(),  color=color2, linewidth=2)  # plot tics at zero velocity
    plt.plot( plt.xlim(), (1.0,1.0), color=color2)  # plot unity continuum
