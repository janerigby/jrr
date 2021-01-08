from astropy import constants as const
import numpy as np

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
