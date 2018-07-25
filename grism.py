# Useful functions for dealing w HST grism spectra.  Started for S1723
# Used by S1723_working.py, grism_fitcont.py, grism_fit_spectra_v4.py
from jrr import spec
from jrr import mage
from jrr import util
from jrr import query_argonaut
from re import search, sub
from os.path import exists
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
    LL_uv  = pandas.read_table(linelistdir + "rest_UV_emission_linelist_short.txt",      delim_whitespace=True, comment="#")
    LL_opt = pandas.read_table(linelistdir + "rest_optical_emission_linelist_short.txt", delim_whitespace=True, comment="#")
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
        EBV_Green2015 = query_argonaut.query(coords.ra.value, coords.dec.value, coordsys='equ', mode='sfd')  #
        val2return = EBV_Green2015.values()[0]
    else: val2return = 0.03415 # Stashed EBV -- temp solution while no internet on train
    return(val2return)

def get_grism_info(which_grism) :
    # R is FWHM, for a pt source.  For extended source will be morphologically broadened
    # wave_unc is relative wavelength uncertainty, in Angstroms, from wfc3 data handbook, S9.4.8
    if   which_grism == 'G141' :   grism_info = {'R':130., 'x1':11000., 'x2':16600., 'wave_unc':9.}
    elif which_grism == 'G102' :   grism_info = {'R':210., 'x1': 8000., 'x2':11500., 'wave_unc':6. }
    else : error_unknown_grism(which_grism)
    return(grism_info)

def parse_filename(grism_filename) :
    m = search('(\S+)_(\S+)_(\S+)_(\S+)_(\S+)', grism_filename)
    mydict = { 'gname': m.group(1), 'descrip': m.group(2), 'roll': m.group(3),  'grating': m.group(4), 'suffix': m.group(5)}
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
        if verbose:  print fitfile, fluxrat, dfluxrat, flux1, dflux1, flux2, dflux2
        else :       print fitfile, fluxrat, dfluxrat    
    return(0)

def measure_linerats_usebothgrisms(G102fitfiles, fitdir, outfilename, line1='Halpha_G141', line2='Hbeta_G102', verbose=False) : #As above, but use both grisms
    outfile = open(outfilename, 'a')
    outfile.write('  '.join(("# ratio of lines: ", line1, line2, 'verbose='+str(verbose)+'\n')))
    if verbose:  outfile.write('counter, file, fluxrat, dfluxrat, flux1, dflux1, flux2, dflux2\n')
    else :       outfile.write('file, fluxrat, dfluxrat\n')
    for ii, G102fitfile in enumerate(G102fitfiles) :
        temp2 = sub("G102", "G141", G102fitfile)
        temp1 = sub('2.fitdf', '1.fitdf', temp2)
        if exists(fitdir + G102fitfile) and (exists(fitdir + temp2) or exists(fitdir + temp1)) :  # if files exist:
            if exists(fitdir + temp2)   : G141fitfile = temp2
            elif exists(fitdir + temp1) : G141fitfile = temp1
            else :  outfile.write("WARNING, I cannot find a file like" + temp2 + temp1 + '\n')
            df1 = pandas.read_csv(fitdir + G102fitfile, comment="#")
            df2 = pandas.read_csv(fitdir + G141fitfile, comment="#")
            df1['linename_grism'] = df1['linename'] + '_G102'
            df2['linename_grism'] = df2['linename'] + '_G141'
            df1.set_index('linename_grism', inplace=True, drop=False) ;   df2.set_index('linename_grism', inplace=True, drop=False)
            df_all = pandas.concat((df1, df2))
            (flux1, dflux1, flux2, dflux2, fluxrat, dfluxrat) = util.linerat_from_df(df_all, line1, line2, colname='linename_grism')
            (flux1, dflux1, flux2, dflux2, fluxrat, dfluxrat) = [round(x, 5) for x in (flux1, dflux1, flux2, dflux2, fluxrat, dfluxrat)]
            if verbose:  outfile.write( '  '.join(str(x) for x in (ii, G102fitfile, fluxrat, dfluxrat, flux1, dflux1, flux2, dflux2, '\n')))
            else :       outfile.write('   '.join(str(x) for x in (G102fitfile, fluxrat, dfluxrat, '\n')))
        else :
            if verbose : outfile.write("#Files do not exist:  " + ' '.join(str(x) for x in (temp1, temp2, '\n')))
    #return(df_all) # as an example
    return(0)

# header gets written 3x  Needd to make it get written only once.  Also, ii needs to be unique; it's not.
