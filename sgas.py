# Helper functions for dealing with SGAS (SDSS Giant Arcs Survey) data, that are not specific to MagE.
# Also adding lensing equation helper functions.
# jrigby, Dec 2020
from jrr import util, spec
import numpy as np
import pandas
from astropy import constants
from astropy.stats import gaussian_fwhm_to_sigma




def load_s1723_lineflux_template(indir='~/Python/jrr/Example_data/', infile='tab_s1723_measured_emission_v6.csv'):
    # The infile is in CSV format.  It was created by converting Table 1 of Rigby et al. 2020 Table 1 (in latex format)
    # to csv, using JRR_Code/s1723_convert_latex_table_csv.py.
    # JRR copied it from the S1723 paper to here (in Dropbox/SGAS/) so team has access.
    #
    # Need to change the table a bit before it can be used -- it has S_II blended rather than the indy lines,
    # and it double-counts C~III] [O III]
    # The SII doublet was blended at grism resoln. Split it, with the flux evenly distributed.
    names = ['lineID', 'wave', 'spectrograph', 'Wr', 'Wr_u', 'significance', 'flux', 'flux_u', 'dr_flux', 'dr_flux_u']  #column names
    df3 = pandas.read_csv(indir+infile, comment='#', index_col=0)
    df3.drop(index=68, inplace=True)  # Drop the SII row
    df_SII = pandas.read_csv(indir+'fake_SII.txt', names= names + ['detected',])
    df_SII['detected'] =  df_SII['detected'].astype(bool)
    df4 = pandas.concat([df3, df_SII])
    df4.reset_index(inplace=True)
    df4.drop('index', axis=1, inplace=True)
    df4 = df4.reindex()

    # Now that df is well-behaved, convert all eligible columns to float64
    floatcols = ['wave', 'Wr', 'Wr_u', 'significance', 'flux', 'flux_u', 'dr_flux', 'dr_flux_u']
    for thiscol in floatcols :
        df4[thiscol] = df4[thiscol].astype(float)

    flux_scaling = 1.0E-17  # The S1723 table's fluxes are in units of 10^-17 erg/s/cm^2
    fluxcols = ['flux', 'flux_u', 'dr_flux', 'dr_flux_u']
    for thiscol in fluxcols :
        df4[thiscol] *= flux_scaling
    return(df4)

def make_linelist_from_s1723(indir='~/Python/jrr/Example_data/', infile='tab_s1723_measured_emission_v6.csv', zz=1.3293) :
    # Make a linelist (LL in mage.py format) from the S1723 detected lines`````````
    df =  load_s1723_lineflux_template(indir, infile)
    LL = make_a_mage_linelist_from_inputlist(df['wave'], df['lineID'], zz, vmask=500., ltype='EMISSION', color='red', src='S1723_template')
    return(LL)

def make_a_mage_linelist_from_inputlist(restwaves, lineIDs, zz, vmask=500., ltype='EMISSION', color='red', src='input') :
    # Make a linelist, in the mage LL format, from any set of rest-frame wavelengths and IDs.  Keep here or move to mage.py or spec.py? 
    LL = pandas.DataFrame()
    LL['restwav'] = restwaves
    LL['lab1']    = lineIDs
    LL['zz']      = zz
    LL['type']    = ltype
    LL['color']   = color
    LL['src']     = src
    LL['obswav'] = LL['restwav'] * (1.0 + LL['zz'])  # observed wavelength for this line
    LL['fake_v']   = 0    # need this for plot_linelist if velplot
    LL['fake_wav'] = 0    # need this for Mage_plot if restframe
    LL['vmask'] =  vmask  # default window to mask the line
    return(LL) 

def make_synthetic_spectrum_based_on_S1723(zz, fwhm_kms=100, scaleby=1.0) :  # fwhm_kms is assumed line width, fwhm, in km/s.  scaleby is factor to scale fluxes
    # Convenient wrapper function to load the S1723 template, and use it to generate a simulated spectrum
    # Load the S1723 lineflux measurements into a well-behaved pandas dataframe.  Uses the local copy which has duplicate line measurements commented out
    df_s1723 = load_s1723_lineflux_template() 
    c_kms = constants.c.to('km/s').value   # Someday I'll trust astropy.units
    df_s1723['sigma'] = (fwhm_kms  / c_kms ) * gaussian_fwhm_to_sigma * df_s1723['wave']  # linewidths to use
    detected_lines =  df_s1723[df_s1723['detected']].copy(deep=True)             #solving pandas' SettingWithCopyWarning  
    print("Will scale the fluxes of S1723 by a factor of", np.round(scaleby, 5))

    # Make a synthetic spectrum based on the fluxes of S1723, redshifted and scaled
    startwave = 1300.  # Angstroms
    endwave   = 7000.  # Angstroms
    Npix      = 9.0E4   #kludge
    df_sim = pandas.DataFrame(data=pandas.Series(np.geomspace(startwave, endwave, Npix)), columns=('rest_wave',))
    df_sim['rest_flam'] = 0.0     # no continuum to start
    df_sim['rest_flam_u'] = 0.0   # no uncertainty either
    df_sim['d_restwave'] = df_sim['rest_wave'].diff()  # dlambda array, the width of 1 pixel
    df_sim['d_restwave'].iloc[0] = df_sim['d_restwave'].iloc[1] # first value is bogus
    detected_lines['obs_wave']    = detected_lines['wave'] * (1.0 + zz)
    detected_lines['scaled_flux'] = detected_lines['flux'] * scaleby   # Scale lineflux from above.  No (1+z), b/c flux is conserved
    for row in detected_lines.itertuples() :  # Make rest frame synthetic spectrum.  Add a gaussian for each detected line, at the right flux
        amplitude = row.scaled_flux / row.sigma / np.sqrt(2.*np.pi)  # scaleby already applied to scaled_flux.  No 1+z bc this is rest wave
        y_thisgauss = spec.onegaus(df_sim['rest_wave'],  amplitude, row.wave, row.sigma, 0.0)  # array to fill, AMPLITUDE, central wavelength, sigma, continuum
        df_sim['rest_flam'] +=  y_thisgauss

    spec.convert2obsframe_df(df_sim, zz, units='flam', colrw='rest_wave', colrf='rest_flam', colrf_u='rest_flam_u')
    df_sim['d_wave'] = df_sim['d_restwave'] * (1+zz)
    df_sim['wave_um'] = df_sim['wave']/1E4
    df_sim['fnu_mJy'] = spec.flam2fnu(df_sim['wave'], df_sim['flam']) /1E-23 *1E3 # convert cgs fnu to mJy
    return( df_s1723, detected_lines, df_sim)


def load_s1110_Johnson_Tab4() :
    zz_s1110 = 2.481  
    johnson_tab4 = pandas.read_csv("Johnson_2017a_Table4.txt", delim_whitespace=True, comment="#")
    # The magnitudes in Tab 4 of Johnson et al 2017a are too faint by 7.5 mag (factor of 1000), due to a scaling factor.
    # Actually I think this is more like 5 mags (~100), Feb 2021
    fix_mag_factor =  -2.5*np.log10(100.)      # Not sure this is right.  Resulting mags are crazy
    fix_mag_keys = ('m_F606W', 'm_F390W')
    for thiskey in fix_mag_keys :   johnson_tab4[ thiskey] += fix_mag_factor
    # convert (no longer screwewd up) m_AB(F606W) to flux density, correct for 1+z, and convert UV fnu to SFR following Kennicutt
    johnson_tab4['fnu606'] = johnson_tab4['m_F606W'].map(lambda x: util.mAB_to_fnu(x))    # convert AB mags to observed-frame fnu
    johnson_tab4['Lnu']    =  johnson_tab4['fnu606'].map(lambda x: util.fnu_to_Lnu(x, zz_s1110))  # convert obs fnu to rest flambda
    johnson_tab4['SFR'] = johnson_tab4['Lnu'].map(lambda x: util.Kennicutt_LUV_to_SFR(x))  # Eqn 1 of Kennicutt 1998, in Msol/yr
    return(johnson_tab4)
