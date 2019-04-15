from __future__ import print_function
import jrr
mage_mode = 'released'
import matplotlib.pyplot as plt


''' This is a test suite for Jane Rigby's mage tools (jrr.mage).  Work in progress.
jrigby Apr 2019.'''

#### Set up ###############
mage_mode = 'released'
labels = ('Horseshoe', 'planckarc_h5')
###########################


(sp, resoln, dresoln, LL, zz_syst) = jrr.mage.wrap_open_spectrum(labels[0], mage_mode) 
print("Loaded 1 spectrum, it looks like this: ", labels[0], "got ", sp.head(2))

(big_sp, resoln, dresoln, big_LL, big_zz_sys, specs) = jrr.mage.open_many_spectra(mage_mode, which_list='labels', labels=labels, MWdr=True)  # open honking
print(("Loaded and processed the following spectra:", labels))

print("Here it is again, from open_many_spectra: ")
print(big_sp[labels[0]].head(2))


assert zz_syst == big_zz_sys['Horseshoe']


print("Show that we can plot spectra.")

for ii, key in enumerate(labels):
    ax = big_sp[key].plot(x='wave', y='fnu', label=key, color='black')
    big_sp[key].plot(x='wave', y='fnu_autocont', color='green', label='autocont', ax=ax) 
    plt.ylim(0,5E-28)
plt.show()

jrr.mage.plot_1line_manyspectra(1548., "CIV test", 100, mage_mode=mage_mode)  # Kinda rough
