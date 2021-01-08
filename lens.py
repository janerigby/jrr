from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from jrr import util

# Basic functions to work with Keren's lens models. 

def lensing_distances(cosmo, z_lens, z_src, verbose=False) :
    D_L = cosmo.angular_diameter_distance(z_lens)
    D_S = cosmo.angular_diameter_distance(z_src) 
    D_LS = cosmo.angular_diameter_distance_z1z2(z_lens,z_src)        # Should return a sensible value, since z1<z2.
    D_LS_wrong = cosmo.angular_diameter_distance_z1z2(z_src, z_lens) # Bad input not caught in error trapping; returns negative D.
    # Astropy should throw an error if z2<z1 in angular_diameter_distance_z1z2(z1, z2).  It does not. It just warns
    # in the documentation. That's dumb.  I submitted a pull request.  For now, ignore D_LS_wrong.
    if verbose: print("Testing lensing distances:", D_L, D_S, D_LS)  # , D_LS_backwards)
    return(D_L, D_S, D_LS)

def scale_lensing_deflection(cosmo, z_lens, z_src, verbose=False) :
    # Deflection maps are scaled to d_LS / d_s =1.  Need to scale by distances for actual redshifts
    (D_L, D_S, D_LS) = lensing_distances(cosmo, z_lens, z_src)
    return( (D_LS / D_S).value) 

def critical_density(cosmo, z_lens, z_src) :
    (D_L, D_S, D_LS) = lensing_distances(cosmo, z_lens, z_src)
    sigma_critical =  const.c **2  / (4. * np.pi * const.G) * D_S / (D_L * D_LS)
    return(sigma_critical)

def make_lensed_regions_file(df_in, datadir, prefix, racol, deccol, color='green', suffix='reg') :  # Helper function stays in this file
    # Helper function to make a ds9 regions file from a dataframe containing RA, DEC, ID of arcs
    df_reg = df_in.copy(deep=True)
    df_reg['text'] = 'text={' + df_reg['ID'].astype('str') + ' ' + suffix  + '}'
    df_reg['color'] = 'color=' + color
    df_reg['radius'] = '1.5'
    util.make_ds9_regions_file(datadir + prefix + suffix + '.reg', df_reg, racol=racol, deccol=deccol)
    return(df_reg)

def compute_sourceplane_positions(arcs_df, prefix,z_lens, datadir, deflection_map, cosmo, deflection_norm=1, debug=False) :
    # Compute source plane positions, given a dataframe of image plane positions and redshifts, and deflection maps in x,y
    # see test_lensing_deflections.py for a worked example for A2744 and RCS0327
    # def of deflection_norm:  The normalization for the deflection maps, which are scaled to some d_LS/d_S.  They scale linearly w d_LS/d_S.
    prev_imgpos = SkyCoord(0.0, 0.0, unit='deg') # initalize
    prev_srcpos = SkyCoord(0.0, 0.0, unit='deg') # initalize
    print("Coordinates                                 distance from previous")
    print("The distances for the first image in each family are meaningless")
    print("ID   image plane    src plane               d_img    d_src")
    df_imgplane_reg =  make_lensed_regions_file(arcs_df, datadir, prefix, 'RA', 'DEC', color='green', suffix='img') 
    src_pos_list = []
    for row in arcs_df.itertuples() :
        img_pos = SkyCoord(row.RA, row.DEC, unit='deg')
        z_src = row.zz
        thisdeflect = {}
        for key in deflection_map : #Loop through the X and Y deflection maps
            defscale = scale_lensing_deflection(cosmo, z_lens, z_src) / deflection_norm
            imval, xy = util.imval_at_coords(datadir + deflection_map[key], img_pos)
            thisdeflect[key] = imval*defscale
            if debug:
                print("  ", row.ID, "  Deflection in", key, "at", xy, "(stupid numpy yx index):", np.round(imval, 2), end='  ')
                print("scale by", np.round(defscale,3), "to get ", np.round(thisdeflect[key],2))
        src_pos =  util.offset_coords(img_pos, delta_RA=thisdeflect['x'] *1, delta_DEC=thisdeflect['y']*-1) # minus sign here, per lensing eqn?
        src_pos_list.append(util.print_skycoords_decimal(src_pos)) # save this to list, then put it in the data frame after the loop.
        print(row.ID, util.print_skycoords_decimal(img_pos), util.print_skycoords_decimal(src_pos), end=' ')
        print(prev_imgpos.separation(img_pos).to_string(unit=u.arcsec, precision=2),  prev_srcpos.separation(src_pos).to_string(unit=u.arcsec, precision=2))
        prev_imgpos = img_pos
        prev_srcpos = src_pos

    # Store the source-plane positions in the dataframe.
    arcs_df['temp'] = src_pos_list
    arcs_df[['RA_srcplane', 'DEC_srcplane']] = arcs_df['temp'].str.split(expand=True).astype('float')
    arcs_df.drop(['temp',],  inplace=True, axis=1)
    arcs_df.to_csv(prefix + '_wsrccoords.csv')
    df_srcplane_reg =  make_lensed_regions_file(arcs_df, datadir, prefix, 'RA_srcplane', 'DEC_srcplane', color='red', suffix='src') 
    return(arcs_df)
