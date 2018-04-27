#  Simple routines to read in BPASS2 spectra into pandas dataframes.  jrigby, Apr 2018
import pandas

def model_dir() : # Set this to where your models are
    return('/Volumes/Apps_and_Docs/JRR_Utils/BPASS_v2.1/')

def ages_setup() :  # Create a pandas dataframe of ages and column names ('coln'), as in BPASS v2.1 manual
    df_age = pandas.DataFrame([ 10**(6+0.1*(n-2)) for n in range(2,52+1)], columns=('age',))
    colnames = pandas.Series(["col"+str(n) for n in range(2,52+1)])
    df_age['colname'] = colnames
    return(df_age) 

def find_closest_age(age_to_find):
    df_age = ages_setup()
    closest_age = df_age.iloc[(df_age['age']- age_to_find).abs().argsort()[:1]]
    return(closest_age) # Returns the row of df_age thats closest in age.  

def load_spectra(filename) :  # Loads spectra for all ages.  1st col is wavelength ('wave'), in Angstroms
    bpassdir = model_dir()
    df_age = ages_setup()
    df_bpass = pandas.read_table(bpassdir + filename, header=None, delim_whitespace=True, names=['wave'] + df_age['colname'].tolist())
    return(df_bpass) 

def load_1spectrum(filename, age_to_find) : #Streamline version of above, just loads 1 age.  Faster.
    bpassdir = model_dir()
    closest_age = find_closest_age(age_to_find)
    usecols = [0, closest_age.index.values[0] + 1]  # Test this!
    names = ['wave', 'flam']
    df_bpass = pandas.read_table(bpassdir + filename, header=None, delim_whitespace=True, usecols=usecols, names=names)
    return(df_bpass)
    
