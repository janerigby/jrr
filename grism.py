from __future__ import print_function
# Useful functions for dealing w HST grism spectra.  Started for S1723
# Used by S1723_working.py, grism_fitcont.py, grism_fit_spectra_v4.py
from builtins import str
from jrr import spec
from jrr import mage
from jrr import util
from jrr import  MW_EBV
from re import search, sub, split
from os.path import exists, basename
from astropy.coordinates import SkyCoord
from astropy import units as u
from numpy import sqrt
from matplotlib import pyplot as plt
import pandas
from numpy import round 

def wrap_fit_continuum(sp, LL, zz, boxcar, colwave='wave', colf='flam_cor', colfu='flam_u_cor', colcont='flamcor_autocont', label="", makeplot=True) :
    (smooth1, smooth2) =  spec.fit_autocont(sp, LL, zz, boxcar=boxcar, colf=colf,  colcont=colcont)
    notmasked = sp[~sp['contmask']]
    if makeplot: 
        plt.plot(sp[colwave],  sp[colf],    color='green', label=label)
        plt.plot(notmasked[colwave],  notmasked[colfu],   color='lightgreen')
        plt.plot(sp[colwave],  sp['contmask']*sp[colf].median(),   color='orange', label='masked')
        plt.plot(sp[colwave],  sp[colcont], color='k', label='Auto continuum fit')
        plt.ylim(sp[colf].median() * -0.1, sp[colf].median() * 5)
        plt.legend()
    return(smooth1, smooth2)

def load_linelists(linelistdir, zz, vmask=500.) :
    LL_uv  = pandas.read_csv(linelistdir + "rest_UV_emission_linelist_short.txt",      delim_whitespace=True, comment="#")
    LL_opt = pandas.read_csv(linelistdir + "rest_optical_emission_linelist_short.txt", delim_whitespace=True, comment="#")
    (spec_path, line_path) = mage.getpath('released')
    (LL_temp, zz_notused) =  mage.get_linelist(line_path + 's1723.linelist')
    LL_uvabs = LL_temp.loc[LL_temp['type'].eq("ISM")].copy(deep=True)  # This already has redshift from .linelist.  may be out of synch****
    LL_uv['zz']  = zz  ;   LL_opt['zz'] = zz   # Load the redshift
    LL_uv.sort_values( by='restwav', inplace=True)
    LL_opt.sort_values(by='restwav', inplace=True)
    LL_uvabs.sort_values(by='restwav', inplace=True)
    LL = pandas.concat([LL_uv, LL_opt, LL_uvabs], ignore_index=True)  # Merge the two linelists
    LL.sort_values(by='restwav', inplace=True)  # Sort by wavelength
    LL.reset_index(drop=True, inplace=True)
    LL['obswav'] = LL['restwav'] * (1.0 + LL['zz'])
    LL['fake_wav'] = 0
    LL['fake_v'] = 0
    LL['vmask'] = vmask  # Dummy for now
    LL.drop_duplicates(subset=('restwav', 'lab1'), inplace=True)  # drop duplicate entries, w same rest wavelength, same ion label    
    return(LL)

def get_MWreddening_S1723(internet=True) :
    # Get MW reddening E(B-V) from Green et al. 2015, using their API query_argonaut
    coords = SkyCoord(ra=260.9006916667, dec=34.199825, unit=(u.deg, u.deg))
    if internet:
        EBV_Green2015 =  MW_EBV.query(coords.ra.value, coords.dec.value, coordsys='equ', mode='sfd')  #
        val2return = list(EBV_Green2015.values())[0]
    else: val2return = 0.03415 # Stashed EBV -- temp solution while no internet on train
    return(val2return)

def get_grism_info(which_grism) :
    # R is FWHM, for a pt source.  For extended source will be morphologically broadened
    # wave_unc is relative wavelength uncertainty, in Angstroms, from wfc3 data handbook, S9.4.8
#    if   which_grism == 'G141' :   grism_info = {'R':130., 'x1':11000., 'x2':16600., 'wave_unc':9.} # what was used for S1723, S2340
    if   which_grism == 'G141' :   grism_info = {'R':130., 'x1':11000., 'x2':17100., 'wave_unc':9.}  # extending to the red for Sunburst
    elif which_grism == 'G102' :   grism_info = {'R':210., 'x1': 8000., 'x2':11500., 'wave_unc':6. }
    else : error_unknown_grism(which_grism)
    return(grism_info)

def parse_filename(grism_filename) :
    foo = split('_', grism_filename)
    mydict = { 'gname': foo[0], 'descrip': foo[1], 'roll': foo[2],  'grating': foo[3]}
    if len(foo) > 4 :  mydict['suffix']  = foo[4]
    if len(foo) == 6:  mydict['suffix2'] = foo[5]
    return(mydict)

def half_the_flux(sp, colf='flam', colfu='flam_u', colcont='cont') :
    # Michael's "bothrolls" 1D extractions for S1723 have flux x2 too high bc summed over both rolls.
    #So, divide flux by 2, and drop uncertainties by root-2.
    sp[colf]    /= 2.
    sp[colcont] /= 2.    
    sp[colfu]   /= sqrt(2)
    return(0)

def measure_linerats_fromfiles(fitfiles, fitdir, line1, line2, verbose=False) :  # given a list of fits files, return line ratios
    for fitfile in fitfiles :
        df = pandas.read_csv(fitdir + fitfile, comment="#")
        (flux1, dflux1, flux2, dflux2, fluxrat, dfluxrat) = util.linerat_from_df(df, line1, line2)
        if verbose:  print(fitfile, fluxrat, dfluxrat, flux1, dflux1, flux2, dflux2)
        else :       print(fitfile, fluxrat, dfluxrat)
    return(0)

def measure_linerats_usebothgrisms(G102fitfiles, outfilename, line1='Halpha_G141', line2='Hbeta_G102', verbose=False) : #As above, but use both grisms
    outfile = open(outfilename, 'w')
    outfile.write('  '.join(("# ratio of lines: ", line1, line2, 'verbose='+str(verbose)+'\n')))
    if verbose:  outfile.write('counter, file, fluxrat, dfluxrat, flux1, dflux1, flux2, dflux2\n')
    else :       outfile.write('file, fluxrat, dfluxrat\n')
    for ii, G102fitfile in enumerate(G102fitfiles) :
        temp2 = sub("G102", "G141", G102fitfile)
        temp1 = sub('2.fitdf', '1.fitdf', temp2)
        if exists(G102fitfile) and (exists(temp2) or exists(temp1)) :  # if files exist:
            if exists(temp2)   : G141fitfile = temp2
            elif exists( + temp1) : G141fitfile = temp1
            else :  outfile.write("WARNING, I cannot find a file like" + temp2 + temp1 + '\n')
            df1 = pandas.read_csv(G102fitfile, comment="#")
            df2 = pandas.read_csv(G141fitfile, comment="#")
            df1['linename_grism'] = df1['linename'] + '_G102'
            df2['linename_grism'] = df2['linename'] + '_G141'
            df1.set_index('linename_grism', inplace=True, drop=False) ;   df2.set_index('linename_grism', inplace=True, drop=False)
            df_all = pandas.concat((df1, df2))
            (flux1, dflux1, flux2, dflux2, fluxrat, dfluxrat) = util.linerat_from_df(df_all, line1, line2, colname='linename_grism')
            (flux1, dflux1, flux2, dflux2, fluxrat, dfluxrat) = [round(x, 5) for x in (flux1, dflux1, flux2, dflux2, fluxrat, dfluxrat)]
            shortname = sub("_wcontMWdr_meth2.fitdf", "", basename(G102fitfile))
            namedict = parse_filename(shortname)
            label = namedict['descrip'] + '-' + namedict['roll']
            if verbose:  outfile.write( '  '.join(str(x) for x in (ii, shortname, label, fluxrat, dfluxrat, flux1, dflux1, flux2, dflux2, '\n')))
            else :       outfile.write('   '.join(str(x) for x in (shortname, fluxrat, dfluxrat, '\n')))
        else :
            if verbose : print("#Files do not exist:  " + ' '.join(str(x) for x in (basename(temp1), basename(temp2))))
    #return(df_all) # as an example
    return(0)

# header gets written 3x  Needd to make it get written only once.  Also, ii needs to be unique; it's not.
