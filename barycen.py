''' This is code to compute and apply the barycentric velocity correction, as defined in jskycalc documentation.
Adapted from https://gist.github.com/StuartLittlefair/5aaf476c5d7b52d20aa9544cfaa936a1
Eventually such a wrapper function will get written into astropy.  In the meantime, using this stopgap.
I have verified that this code gives the same answers as jskycalc, for a few positions,
and a few observatories (Las Campanas and Keck), to within 0.01 km/s.
jrigby, Feb 2017'''

from astropy.time import Time
from astropy import coordinates
from astropy.coordinates import SkyCoord, solar_system, EarthLocation, ICRS
from astropy import units as u
from astropy import constants


def apply_barycentric_correction(df, barycor_vel,  colwav='wave', colwavnew='newwave'):
    ''' Applies the barycentric velocity correction to the wavelength of a spectrum.  Needs to be tested.'''
    A_c = constants.c  # speed of light
    df[colwavnew] = df[colwav] * (1.0 + (barycor_vel / A_c))
    return(0)

def compute_barycentric_correction(mytime, skycoord, location=None):
  """Barycentric velocity correction.  was named 'velcorr'
     Should return barycentric correction factor, in km/s, same sign convention as in jskycalc 
  
  Uses the ephemeris set with  ``astropy.coordinates.solar_system_ephemeris.set`` for corrections. 
  For more information see `~astropy.coordinates.solar_system_ephemeris`.
  
  Parameters
  ----------
  mytime : `~astropy.time.Time`
    The time of observation.
  skycoord: `~astropy.coordinates.SkyCoord`
    The sky location to calculate the correction for.
  location: `~astropy.coordinates.EarthLocation`, optional
    The location of the observatory to calculate the correction for.
    If no location is given, the ``location`` attribute of the Time
    object is used
    
  Returns
  -------
  vel_corr : `~astropy.units.Quantity`
    The velocity correction to convert to Barycentric velocities. Should be added to the original
    velocity.
  """
  
  if location is None:
    if mytime.location is None:
        raise ValueError('An EarthLocation needs to be set or passed '
                         'in to calculate bary- or heliocentric '
                         'corrections')
    location = mytime.location
    
  # ensure sky location is ICRS compatible
  if not skycoord.is_transformable_to(ICRS()):
    raise ValueError("Given skycoord is not transformable to the ICRS")
  
  ep, ev = solar_system.get_body_barycentric_posvel('earth', mytime) # ICRS position and velocity of Earth's geocenter
  op, ov = location.get_gcrs_posvel(mytime) # GCRS position and velocity of observatory
  # ICRS and GCRS are axes-aligned. Can add the velocities
  velocity = ev + ov # relies on PR5434 being merged
  
  # get unit ICRS vector in direction of SkyCoord
  sc_cartesian = skycoord.represent_as(coordinates.UnitSphericalRepresentation).represent_as(coordinates.CartesianRepresentation)
  return sc_cartesian.dot(velocity).to(u.km/u.s) # similarly requires PR5434

