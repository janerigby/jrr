# Useful functions for dealing w HST grism spectra.  Started for S1723
# Used by S1723_working.py, grism_fitcont.py.
from jrr import spec
from jrr import mage
from jrr import query_argonaut
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib import pyplot as plt
import pandas


def wrap_fit_continuum(sp, LL, zz, boxcar, colwave='wave', colf='flam_cor', colfu='flam_u_cor', colcont='flamcor_autocont', label="") :
    (smooth1, smooth2) =  spec.fit_autocont(sp, LL, zz, boxcar=boxcar, colf=colf,  colcont=colcont)
    notmasked = sp[~sp['contmask']]
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
    LL_uvabs = LL_temp.loc[LL_temp['type'].eq("ISM")]  # This already has redshift from .linelist.  may be out of synch****
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
