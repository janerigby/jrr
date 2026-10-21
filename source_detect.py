from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval
from scipy.stats import norm
from scipy.signal import find_peaks # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.htm


#This is Rosalia's source detection algorithm, for detecting traces in 2D NIRSpec FS spectra

def wrap_detect_source_s2d(s2dfile, plot=False):
    with fits.open(s2dfile) as hdul:
        spec2d = hdul['SCI'].data[1:-1].copy()  # Read the 2D spectrum.
    detection_level = detect_source_s2d(spec2d, plot=plot)
    return(detection_level)

# Function to detect a source in a 2d JWST S2D NIRSpec spectrum  # From Rosalia O'Brien
def detect_source_s2d(spectrum_2d, plot=False):
    """
    Detects if a horizontal trace is present in a 2D (works with NaNs in data).

    Params
    ------
    spectrum_2d - arr
        2D spectrum
    plot - bool
        Make pretty plots or no

    Output
    ------
    detection_level - Float
        Detection level which sources are found 
    """
    
    # Collapse the 2D spectrum using np.nanmedian
    # axis = 1 means each row gets collapsed into one value
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'All-NaN slice encountered',
            category=RuntimeWarning)
        collapsed_med_spec = np.nanmedian(spectrum_2d, axis=1)
    
    # Estimate the background level and noise 
    background_level = np.nanmedian(collapsed_med_spec)
    mad = np.nanmedian(np.abs(collapsed_med_spec - background_level))
    sigma = mad * 1.4826 # std ≈ 1.4826 × mad (https://statisticsbyjim.com/basics/median-absolute-deviation/)
        
    # Find "peaks" in 2d speak corresponding to sources
    # For now, just identify sources that are at least 1 sigma in height
    required_height = background_level + (sigma)
    peak_positions, properties = find_peaks(collapsed_med_spec, height=required_height)
    # peak_positions --> xloc of peaks
    # collapsed_med_spec[peak_positions] --> "height" (yloc) of peaks
    
    source_detected = len(peak_positions) > 0 # Bool, if source is found
    detection_level_all = (collapsed_med_spec[peak_positions] - background_level)/sigma

    if source_detected:
        # print(np.max(detection_level_all))
        detection_level = np.max(detection_level_all)
    if not source_detected:
        detection_level = 0.0

    if plot:
        fig = plt.figure(figsize=(12, 7))

        # Make plot pretty 
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5]) # Create a grid: 2 rows, 3 columns
        ax1 = fig.add_subplot(gs[0, :]) # ax1 spans all 3 columns in the first row
        ax2 = fig.add_subplot(gs[1, 0]) # ax2 sits only in the middle column of the second row
        
        # Plot
        zscale = ZScaleInterval(contrast = 0.5)
        norm = ImageNormalize(spectrum_2d, interval=zscale)
        im = ax1.imshow(spectrum_2d, aspect='auto', origin='lower', cmap='Greys_r', norm=norm)
        ax1.set_title("2D Spectrum")
        ax1.set_xlabel("Dispersion Axis [Pixels]")
        ax1.set_ylabel("Spatial Axis [Pixels]")
        
        # Overlay locations of detected source traces
        for i, p in enumerate(peak_positions):

            leg = False # Whether or not to plot legend (only show legend if you actually find sources at > 3 sigma)

            # Only show >3 sigma sources
            if collapsed_med_spec[p] > (background_level + 3*(sigma)):
                
                # Only assign the label to the very first line drawn
                lbl = 'Detected 3-sigma Source' if i == 0 else '_nolegend_'
                
                ax1.axhline(p+3, color='red', linestyle='--', alpha=0.3, label=lbl)
                ax1.axhline(p-3, color='red', linestyle='--', alpha=0.3)

                leg = True
                
            if leg:
                ax1.legend()
        
        # Plot 1D median spatial spectra
        ax2.plot(collapsed_med_spec, color='blue', label='Collapsed Profile')
        ax2.axhline(background_level, color='gray', linestyle='--', label='Background')
        ax2.axhline(background_level + (3*sigma), color='orange', linestyle='--', label='3-Sigma Threshold')
        
        if source_detected:
            ax2.plot(peak_positions, collapsed_med_spec[peak_positions], "rx", markersize=10, label='Detected 1-Sigma Peaks')
            
        ax2.set_title("1D Spatial Profile")
        ax2.set_xlabel("Spatial Axis [Pixels]")
        ax2.set_ylabel("Median Flux [MJy/sr]")
        ax2.legend(bbox_to_anchor = (1,1), loc = 'upper left')
        
        plt.tight_layout()
        plt.show()
    
    return detection_level


# THIS IS Codex's improved version.  GPT-5.6-Sol, Oct 2026

"""Multi-scale trace detection for rectified 2-D spectra.  Allows source to be narrow or wide,
at any position within slit.  It jointly fits a linear spatial background and a positive Gaussian source
template, searches all centers and widths, and supports edge-truncated traces.
"""



def _collapsed_profile(data, error=None):
    """Median spatial profile and an approximate error on each row median."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmedian(data, axis=1)

    if error is None:
        residual = data - profile[:, None]
        scatter = 1.4826 * np.nanmedian(
            np.abs(residual - np.nanmedian(residual, axis=1)[:, None]), axis=1
        )
        count = np.sum(np.isfinite(data), axis=1)
        profile_error = 1.2533 * scatter / np.sqrt(np.maximum(count, 1))
    else:
        good = np.isfinite(data) & np.isfinite(error) & (error > 0)
        count = np.sum(good, axis=1)
        # Standard error of a median for approximately Gaussian samples.
        profile_error = 1.2533 * np.sqrt(
            np.nansum(np.where(good, error**2, np.nan), axis=1)
        ) / np.maximum(count, 1)

    valid_error = np.isfinite(profile_error) & (profile_error > 0)
    fallback = np.nanmedian(profile_error[valid_error])
    profile_error[~valid_error] = fallback
    return profile, profile_error


def _fit_template(profile, profile_error, center, sigma):
    y = np.arange(profile.size, dtype=float)
    template = np.exp(-0.5 * ((y - center) / sigma) ** 2)
    good = np.isfinite(profile) & np.isfinite(profile_error) & (profile_error > 0)
    yg = y[good]
    scale = max(profile.size - 1, 1)
    design = np.column_stack((np.ones(yg.size), (yg - y.mean()) / scale, template[good]))
    weight = 1.0 / profile_error[good] ** 2
    normal = design.T @ (weight[:, None] * design)
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None
    coefficients = covariance @ (design.T @ (weight * profile[good]))
    fitted = design @ coefficients
    chi2 = np.sum(weight * (profile[good] - fitted) ** 2)
    dof = max(good.sum() - design.shape[1], 1)
    # Inflate formal errors when the simple background/template model does not
    # describe real spatial structure or correlated calibration residuals.
    error_scale = max(1.0, np.sqrt(chi2 / dof))
    amplitude_error = np.sqrt(covariance[2, 2]) * error_scale
    raw_significance = coefficients[2] / amplitude_error

    background = coefficients[0] + coefficients[1] * (y - y.mean()) / scale
    model = background + coefficients[2] * template
    return raw_significance, coefficients[2], background, model


def _detect_source_s2d_detailed(
    spectrum_2d,
    error_2d=None,
    widths=None,
    n_blocks=6,
    plot=False,
    outfile=None,
):
    """Find the strongest narrow or extended horizontal trace.

    Parameters
    ----------
    spectrum_2d, error_2d : 2-D arrays
        Science image and optional 1-sigma errors, indexed [spatial, dispersion].
    widths : sequence, optional
        Gaussian sigma values in spatial pixels.  The default spans narrow
        through moderately extended sources.
    n_blocks : int
        Number of wavelength blocks used to assess trace persistence.
    plot : bool
        Create a diagnostic plot.
    outfile : path-like, optional
        Save the plot here.  If omitted and ``plot`` is true, display it.
    """
    data = np.asarray(spectrum_2d, dtype=float)
    error = None if error_2d is None else np.asarray(error_2d, dtype=float)
    if data.ndim != 2 or (error is not None and error.shape != data.shape):
        raise ValueError("spectrum_2d and error_2d must be matching 2-D arrays")

    ny = data.shape[0]
    if widths is None:
        widths = np.geomspace(0.7, max(0.8, ny / 4.0), 14)
    widths = np.asarray(widths, dtype=float)
    widths = widths[(widths > 0) & (widths < ny / 2.0)]
    centers = np.arange(ny, dtype=float)
    profile, profile_error = _collapsed_profile(data, error)

    best = None
    trials = 0
    for sigma in widths:
        for center in centers:
            fitted = _fit_template(profile, profile_error, center, sigma)
            trials += 1
            if fitted is None or fitted[1] <= 0:
                continue
            if best is None or fitted[0] > best[0]:
                best = (*fitted, center, sigma)
    if best is None:
        zeros = np.zeros(ny)
        return {
            "significance": np.nan, "raw_significance": np.nan,
            "center": np.nan, "sigma": np.nan, "fwhm": np.nan,
            "amplitude": 0.0, "background": zeros, "model": zeros,
            "profile": profile, "profile_error": profile_error,
            "block_significances": np.array([]), "trials": trials,
        }

    raw_sig, amplitude, background, model, center, sigma = best
    # Bonferroni correction expresses the search over centers and scales as an
    # equivalent one-sided Gaussian significance.
    false_alarm = min(1.0, norm.sf(raw_sig) * trials)
    significance = max(0.0, norm.isf(false_alarm))

    block_sigs = []
    for columns in np.array_split(np.arange(data.shape[1]), n_blocks):
        block_profile, block_error = _collapsed_profile(
            data[:, columns], None if error is None else error[:, columns]
        )
        fitted = _fit_template(block_profile, block_error, center, sigma)
        block_sigs.append(np.nan if fitted is None else fitted[0])
    block_sigs = np.asarray(block_sigs)

    result = {
        "significance": significance,
        "raw_significance": raw_sig,
        "center": center,
        "sigma": sigma,
        "fwhm": 2.35482 * sigma,
        "amplitude": amplitude,
        "background": background,
        "model": model,
        "profile": profile,
        "profile_error": profile_error,
        "block_significances": block_sigs,
        "trials": trials,
    }
    if plot:
        plot_detection(data, result, outfile=outfile)
    return result


def detect_source_s2d_new(
    spectrum_2d,
    error_2d=None,
    widths=None,
    n_blocks=6,
    plot=False,
    outfile=None,
):
    """Return one simple source-detection value.

    The value is the trials-corrected significance of the best-fitting source,
    including marginal or low-significance fits.  ``-99.0`` means that the fit
    could not be performed.
    """
    try:
        result = _detect_source_s2d_detailed(
            spectrum_2d,
            error_2d=error_2d,
            widths=widths,
            n_blocks=n_blocks,
            plot=plot,
            outfile=outfile,
        )
        if not (
            np.isfinite(result["significance"])
            and np.isfinite(result["raw_significance"])
            and np.isfinite(result["center"])
        ):
            return -99.0
        return float(result["significance"])
    except Exception:
        return -99.0


def plot_detection(spectrum_2d, result, outfile=None):
    """Plot the 2-D spectrum and fitted one-dimensional spatial profile."""
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
    image_axis = fig.add_subplot(grid[0, :])
    profile_axis = fig.add_subplot(grid[1, 0])

    normalization = ImageNormalize(spectrum_2d, interval=ZScaleInterval(contrast=0.5))
    image_axis.imshow(
        spectrum_2d, aspect="auto", origin="lower", cmap="Greys_r", norm=normalization
    )
    image_axis.axhline(result["center"], color="red", linestyle="--", alpha=0.8)
    image_axis.axhspan(
        result["center"] - result["fwhm"] / 2,
        result["center"] + result["fwhm"] / 2,
        color="red", alpha=0.12,
    )
    image_axis.set(
        title=(f"2D Spectrum: {result['significance']:.1f} sigma corrected "
               f"({result['raw_significance']:.1f} sigma raw)"),
        xlabel="Dispersion Axis [Pixels]", ylabel="Spatial Axis [Pixels]",
    )

    y = np.arange(result["profile"].size)
    profile_axis.errorbar(
        y, result["profile"], yerr=result["profile_error"], color="blue", lw=1.5,
        elinewidth=0.5, alpha=0.9, label="Collapsed profile",
    )
    profile_axis.plot(y, result["background"], "--", color="gray", label="Fitted background")
    profile_axis.plot(y, result["model"], color="red", label="Background + source model")
    profile_axis.axvspan(
        result["center"] - result["fwhm"] / 2,
        result["center"] + result["fwhm"] / 2,
        color="red", alpha=0.12, label=f"FWHM = {result['fwhm']:.1f} pixels",
    )
    profile_axis.set(
        title="1D Spatial Profile", xlabel="Spatial Axis [Pixels]",
        ylabel="Median Flux [MJy/sr]",
    )
    profile_axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    if outfile is None:
        plt.show()
    else:
        fig.savefig(outfile, dpi=180, bbox_inches="tight")
        plt.close(fig)


def wrap_detect_source_s2d_new(s2dfile, plot=False, outfile=None, **kwargs):
    """Run the detector on a JWST S2D file; return significance/limit/-99."""
    try:
        with fits.open(s2dfile) as hdul:
            science = hdul["SCI"].data[1:-1].copy()
            error = hdul["ERR"].data[1:-1].copy() if "ERR" in hdul else None
        return detect_source_s2d_new(
            science, error_2d=error, plot=plot, outfile=outfile, **kwargs
        )
    except Exception:
        return -99.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infile")
    parser.add_argument("--plot", default="source_detection2_diagnostic.png")
    args = parser.parse_args()
    answer = wrap_detect_source_s2d_new(args.infile, plot=True, outfile=args.plot)
    if answer == -99:
        print("result: -99 (fit failed)")
    else:
        print(f"result: {answer:.3f} sigma")
