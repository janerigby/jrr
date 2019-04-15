from __future__ import print_function
import jrr
mage_mode = 'released'
import matplotlib.pyplot as plt
labels = ('Horseshoe', 'rcs0327-E', 'SPT0310_slitA', 'planckarc_h5')
print(("Test suite, will load and process the following spectra:", labels))
(big_sp, resoln, dresoln, big_LL, big_zz_sys, specs) = jrr.mage.open_many_spectra(mage_mode, which_list='labels', labels=labels, MWdr=True)  # open honking

print("First part of a spectrum:")
print(big_sp[labels[0]].head())

print("Show that we can plot spectra.")

for ii, key in enumerate(labels):
    big_sp[key].plot(x='wave', y='fnu', label=key)
    plt.ylim(0,1E-27)
plt.show()
