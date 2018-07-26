#  Simple routines to read in BPASS2 spectra into pandas dataframes.  jrigby, Apr 2018
import pandas

def model_dir(ver='2.1') : # Set this to where your models are
    if    ver=='2.1'   :  return('/Volumes/Apps_and_Docs/JRR_Utils/BPASS_v2.1/')
    elif  ver=='2.2'   :  return('/Volumes/Apps_and_Docs/JRR_Utils/BPASS_v2.2/')
    elif  ver=='2.2.1' :  return('/Volumes/Apps_and_Docs/JRR_Utils/BPASS_v2.2.1/')

def default_filenames(ver='2.1', style='binary') : # Default directory, model rootnames.  Default models, also what John wants for consistency w S99.
    if ver=='2.1' and 'binary' in style :
        return ('BPASSv2.1_imf135_100/',     'BPASSv2p1_imf135_100_burst_binary')  # dir, model_rootname
    if ver=='2.1' and 'single' in style :
        return ('BPASSv2.1_imf135_100/',     'BPASSv2p1_imf135_100_burst_single')  # dir, model_rootname
    if ver=='2.2' and 'binary' in style :
        return ('BPASSv2.2_bin-imf135_100/', 'BPASSv2p2_bin-imf135_100_burst_binary')  # NOT TESTED YET
    if ver=='2.2' and 'single' in style :
        return ('BPASSv2.2_sin-imf135_100/', 'BPASSv2p2_sin-imf135_100_burst_single')  # NOT TESTED YET

def ages_setup() :  # Create a pandas dataframe of ages and column names ('coln'), as in BPASS v2.1 manual.  Same for v2.2
    df_age = pandas.DataFrame([ 10**(6+0.1*(n-2)) for n in range(2,52+1)], columns=('age',))
    colnames = pandas.Series(["col"+str(n) for n in range(2,52+1)])
    df_age['colname'] = colnames
    return(df_age) 

def find_closest_age(age_to_find):
    df_age = ages_setup()
    closest_age = df_age.iloc[(df_age['age']- age_to_find).abs().argsort()[:1]]
    return(closest_age) # Returns the row of df_age thats closest in age.  

def load_spectra(filename, ver='2.1') :  # Loads spectra for all ages.  1st col is wavelength ('wave'), in Angstroms
    # Uses filename, which should look like 'BPASSv2.1_imf135_100/spectra.z020.dat.gz'
    bpassdir = model_dir(ver=ver)
    df_age = ages_setup()
    df_bpass = pandas.read_table(bpassdir + filename, header=None, delim_whitespace=True, names=['wave'] + df_age['colname'].tolist())
    return(df_bpass) 

def load_1spectrum(filename, age_to_find, ver='2.1') : #Streamline version of above, just loads 1 age (units=yr)  Faster.
    # Uses filename, which should look like 'BPASSv2.1_imf135_100/spectra.z020.dat.gz'
    bpassdir = model_dir(ver=ver)
    closest_age = find_closest_age(age_to_find)
    #print "DEBUG, closest age was", closest_age
    usecols = [0, closest_age.index.values[0] + 1] 
    names = ['wave', 'flam']
    df_bpass = pandas.read_table(bpassdir + filename, header=None, delim_whitespace=True, usecols=usecols, names=names)
    return(df_bpass)
    
def wrap_load_1spectrum(Z, age, style, ver='2.1') :  #uses parameters rather than filenames. Units of age are years.
    if 'binary' in style   : fileroot = 'spectra-bin'
    elif 'single' in style :
        if    ver=='2.1' : fileroot = 'spectra'
        elif  ver=='2.2' : fileroot = 'spectra-sin'
    else : raise Exception("Misparsed style")
    IMFdir = default_filenames(ver=ver, style=style)[0]
    middlebit_v22 = '-imf135_100'
    if   ver=='2.1' :  filename = IMFdir + fileroot + '.z' + Z + '.dat.gz'
    elif ver=='2.2' :  filename = IMFdir + fileroot + middlebit_v22 + '.z' + Z + '.dat.gz'
    #print "    DEBUGGING filename in bpass.py, filename", filename
    df_bpass2 = load_1spectrum(filename, age, ver=ver)
    return(df_bpass2)  # call as wrap_load_1spectrum('020', 1E6, -2.0, "BPASS_single")
