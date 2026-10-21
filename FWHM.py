# =============================================================================
# Batch PSF / resolution measurement from sub-resolution bead stacks
# -----------------------------------------------------------------------------
# WHAT THIS SCRIPT DOES (overview)
#   Sub-resolution fluorescent beads act as point sources, so the size of the
#   spot you see in the image is a measure of the microscope's resolution
#   (its point spread function, PSF). For each TIFF stack this script:
#     1. reads the 3D image and its voxel size (µm per pixel in z, y, x)
#     2. finds every bead in the stack
#     3. cuts out a small 3D crop around each bead
#     4. fits a Gaussian to the intensity profile through the bead centre along
#        z, y and x, and converts the Gaussian width to a FWHM
#        (full width at half maximum = the resolution measure)
#     5. subtracts the bead's own physical size from each FWHM
#     6. saves the numbers to Excel and a check plot (images + fits) per file
#
# Folder layout (img folder sits next to this script):
#   my_folder/
#     bead_fwhm_batch.py
#     img/                  <- put your .tif/.tiff files here
#     fwhm_results.xlsx     <- created by the script
#     fit_plots/            <- created by the script
#
# Usage:
#   python bead_fwhm_batch.py                 # reads the "img" folder next to the script
#   python bead_fwhm_batch.py "C:/path/to/folder"   # optional: read a different folder instead
#
# Outputs (saved next to the script, not inside img):
#   fwhm_results.xlsx with two sheets:
#     Beads   - one row per bead (all files together), raw and bead-corrected FWHM
#     Summary - one row per file: bead counts and mean/median/SD of unflagged beads
#   fit_plots/<file>_fits.png - for each bead, XZ / XY / YZ images on top with
#     the z/y/x fits underneath, so you can check the fits by eye
# A file that can't be read (e.g. missing voxel size) is skipped and logged in
# the Summary sheet instead of stopping the whole run.
# =============================================================================

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from skimage import filters, measure

# ---------- user settings ----------
SCRIPT_DIR = Path(__file__).resolve().parent                        # folder this script lives in
FOLDER   = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR / "img - example"
PATTERNS = ("*.tif", "*.tiff")             # which files to pick up
OUT_XLSX = SCRIPT_DIR / "fwhm_results.xlsx"    # results workbook (next to the script)
PLOT_DIR = SCRIPT_DIR / "fit_plots"            # one PNG of fits per file goes here (next to the script)

BEAD_D   = 0.100                  # µm, nominal bead diameter (from the bead supplier)
# A bead is not a true point, so it blurs the measurement slightly. D_EFF is how
# much FWHM the bead itself contributes; it is removed later in bead_correct().
D_EFF    = BEAD_D * np.sqrt(3)/2  # µm, FWHM contributed by a uniformly filled sphere (use BEAD_D for conservative)
HALF_BOX = 0.5                    # µm half-width of crop around each bead (use ~0.5 for STED)
MIN_R2   = 0.93                   # flag fits worse than this (R² = goodness of fit, 1 = perfect)

# Conversion factors: metadata unit -> µm (all internal maths is done in µm)
UNIT = {"nm": 1e-3, "µm": 1.0, "um": 1.0, "micron": 1.0, "microns": 1.0, "mm": 1e3}

# Column order for the per-bead table (also used to build an empty table if nothing was measured)
BEAD_COLS = ["file", "bead", "y_px", "x_px",
             "FWHM_z_nm", "FWHM_y_nm", "FWHM_x_nm",
             "FWHM_z_corr_nm", "FWHM_y_corr_nm", "FWHM_x_corr_nm", "flags"]
RAW_COLS  = ["FWHM_z_nm", "FWHM_y_nm", "FWHM_x_nm"]                    # as measured
CORR_COLS = ["FWHM_z_corr_nm", "FWHM_y_corr_nm", "FWHM_x_corr_nm"]     # after removing bead size


# ---------- 1. read image + voxel size ----------
def load(path):
    """Return (img as float Z,Y,X array, vz, vy, vx in µm). Raises ValueError if unusable."""
    with tifffile.TiffFile(path) as tif:
        s = tif.series[0]                      # first image series in the file
        img, axes = s.asarray(), s.axes        # pixel data + axis order string (e.g. "ZYX")
        vz = vy = vx = None                    # voxel sizes, filled in below if found

        # The voxel size is stored differently depending on which software saved the file.
        if tif.is_ome:
            # OME-TIFF: voxel sizes are PhysicalSizeX/Y/Z (+ ...Unit) in the XML <Pixels> element
            px = next(e for e in ET.fromstring(tif.ome_metadata).iter() if e.tag.endswith("Pixels"))
            # f("Z") reads PhysicalSizeZ and converts it to µm using its unit; None if absent
            f = lambda a: (float(px.get(f"PhysicalSize{a}")) * UNIT.get(px.get(f"PhysicalSize{a}Unit", "µm"), 1.0)
                           if px.get(f"PhysicalSize{a}") else None)
            vz, vy, vx = f("Z"), f("Y"), f("X")

        elif tif.is_imagej:
            # ImageJ/Fiji TIFF: z spacing in the ImageJ metadata, x/y as TIFF resolution tags (pixels per unit)
            md = tif.imagej_metadata or {}
            u = UNIT.get(str(md.get("unit", "µm")).strip(), 1.0)
            if md.get("spacing") is not None: vz = float(md["spacing"])*u
            pg = tif.pages[0]
            # XResolution is stored as a fraction n/d = pixels per unit, so pixel size = d/n
            if "XResolution" in pg.tags: n, d = pg.tags["XResolution"].value; vx = d/n*u
            if "YResolution" in pg.tags: n, d = pg.tags["YResolution"].value; vy = d/n*u

    print(f"  axes: {axes}  shape: {img.shape}  voxel (µm) z,y,x: {vz}, {vy}, {vx}")
    # Without voxel sizes the FWHM can't be converted from pixels to nm, so stop for this file
    if None in (vz, vy, vx):
        raise ValueError("voxel size missing from metadata")

    img = np.squeeze(img)                      # drop length-1 axes (channel, time...)
    if img.ndim != 3:
        raise ValueError(f"expected 3D (Z,Y,X) after squeeze, got axes {axes} shape {img.shape}")
    return img.astype(float), vz, vy, vx


# ---------- 2. fitting ----------
def gauss(x, A, mu, sig, off):
    """1D Gaussian: amplitude A, centre mu, width sig, constant background off."""
    return A*np.exp(-(x-mu)**2/(2*sig**2)) + off


def fwhm_1d(profile, step):
    """Fit a Gaussian to a 1D profile; return (FWHM, R², x-axis, fit params). Units follow `step` (µm)."""
    x = np.arange(len(profile))*step           # position of each sample in µm
    # Starting guesses for the fit: amplitude above background, peak position, width of ~2 samples, background
    A0 = profile.max() - np.median(profile)
    p0 = [A0, x[np.argmax(profile)], 2*step, np.median(profile)]
    # Bounds keep the fit sensible: amplitude >= 0, centre inside the profile,
    # width between a quarter of a sample and the whole profile length
    popt, _ = curve_fit(gauss, x, profile, p0=p0, maxfev=5000,
                        bounds=([0, x[0], step/4, -np.inf], [np.inf, x[-1], x[-1]-x[0], np.inf]))
    resid = profile - gauss(x, *popt)          # data minus fit
    r2 = 1 - resid.var()/profile.var()         # goodness of fit (1 = perfect)
    return 2.3548*popt[2], r2, x, popt         # FWHM = 2*sqrt(2 ln2) * sigma


# ---------- 3. find beads in one stack and measure them ----------
def measure_beads(img, vz, vy, vx):
    """Return (list of per-bead dicts in nm, list of fit curves for plotting, list of (crop, peak) for images)."""
    # Step A: locate beads. Max projection along z collapses the stack to one 2D image,
    # a light blur (sigma = 1 px) smooths noise, and Otsu's method picks an automatic
    # brightness threshold. Each connected bright blob is labelled as one bead.
    mip = ndi.gaussian_filter(img.max(0), 1)
    lab = measure.label(mip > filters.threshold_otsu(mip))
    hy, hx = round(HALF_BOX/vy), round(HALF_BOX/vx)       # crop half-size converted from µm to pixels
    rows, fits, crops = [], [], []

    for r in measure.regionprops(lab):
        y0, x0 = map(int, r.centroid)                     # blob centre in the projection (pixels)
        # skip beads whose crop would run off the image edge
        if y0-hy < 0 or x0-hx < 0 or y0+hy >= img.shape[1] or x0+hx >= img.shape[2]:
            print(f"  bead at ({y0},{x0}) skipped: too close to edge")
            continue

        # Step B: cut out a small box around the bead (full z range, small y/x window)
        crop = img[:, y0-hy:y0+hy+1, x0-hx:x0+hx+1]
        # true 3D peak, located on a smoothed copy so single-pixel noise can't win
        z, y, x = ndi.maximum_position(ndi.gaussian_filter(crop, 1))
        # Step C: 1D profiles through the peak along each axis (from the UNSMOOTHED data,
        # so the blur used for finding the peak doesn't widen the measured FWHM).
        # Each entry is (profile, voxel step in µm).
        prof = {"z": (crop[:, y, x], vz), "y": (crop[z, :, x], vy), "x": (crop[z, y, :], vx)}

        # Step D: fit each profile and record quality flags
        out, flag = {}, []
        for ax, (p, st) in prof.items():
            try:
                fw, r2, xs, popt = fwhm_1d(p, st)
                out[ax] = fw*1000                                    # µm -> nm
                fits.append((len(rows), ax, xs, p, gauss(xs, *popt)))   # keep curve for the plot
                if r2 < MIN_R2: flag.append(f"{ax}:R2={r2:.2f}")     # poor fit
                if fw > 0.9*xs[-1]: flag.append(f"{ax}:wide")        # FWHM ~ whole crop (likely not a clean bead)
            except (RuntimeError, ValueError):                       # no convergence / invalid input
                out[ax] = np.nan
                flag.append(f"{ax}:fail")
        rows.append({"bead": len(rows), "y_px": y0, "x_px": x0,
                     "FWHM_z_nm": out["z"], "FWHM_y_nm": out["y"], "FWHM_x_nm": out["x"],
                     "flags": ";".join(flag)})                       # empty string = no warnings
        crops.append((crop, (z, y, x)))                              # keep crop + peak for the bead images
    return rows, fits, crops


def bead_correct(fwhm_nm):
    """Subtract the bead's own size in quadrature (measured² = true² + bead²); NaN if measured <= bead size."""
    # Widths from independent blurs add in quadrature, so the microscope's own FWHM is
    # sqrt(measured² - bead²). If the measured value is smaller than the bead size the
    # correction is meaningless, so NaN is returned instead of a number.
    d = D_EFF*1000                                                   # bead contribution in nm
    fwhm_nm = np.asarray(fwhm_nm, dtype=float)
    return np.where(fwhm_nm > d, np.sqrt(np.maximum(fwhm_nm**2 - d**2, 0)), np.nan)


def save_fit_plot(name, n_beads, fits, df, crops, vox):
    """Per bead: a row of XZ/XY/YZ images on top, then the z/y/x profile fits underneath."""
    vz, vy, vx = vox
    # Two plot rows per bead (images, then fits); the image row is a bit taller
    fig, axs = plt.subplots(2*n_beads, 3, figsize=(10, 4.2*n_beads), squeeze=False,
                            gridspec_kw={"height_ratios": [1.3, 1]*n_beads})

    # --- image row for each bead (axes in µm, red crosshairs mark the fitted peak) ---
    for i, (crop, (z, y, x)) in enumerate(crops):
        nz, ny, nx = crop.shape
        # Each view is a 2D slice through the peak. Tuple fields:
        # (title, 2D slice, extent [left, right, bottom, top] in µm, x-label, y-label, crosshair (h, v))
        views = [
            ("XZ", crop[:, y, :], [0, nx*vx, nz*vz, 0], "x (µm)", "z (µm)", (z*vz, x*vx)),   # side view
            ("XY", crop[z, :, :], [0, nx*vx, ny*vy, 0], "x (µm)", "y (µm)", (y*vy, x*vx)),   # top-down view
            ("YZ", crop[:, :, x], [0, ny*vy, nz*vz, 0], "y (µm)", "z (µm)", (z*vz, y*vy)),   # other side view
        ]
        for j, (title, sl, ext, xl, yl, (h, v)) in enumerate(views):
            a = axs[2*i][j]
            a.imshow(sl, cmap="gray", extent=ext, aspect="auto")     # use aspect="equal" for true proportions
            a.axhline(h, color="r", lw=0.5, alpha=0.6)               # horizontal crosshair line
            a.axvline(v, color="r", lw=0.5, alpha=0.6)               # vertical crosshair line
            a.set_title(f"bead {i} {title}", fontsize=8)
            a.set_xlabel(xl, fontsize=7); a.set_ylabel(yl, fontsize=7)
            a.tick_params(labelsize=6)

    # --- fit graphs directly underneath: dots = data, line = Gaussian fit, title = FWHM ---
    for i, ax, xs, p, fit in fits:
        j = "zyx".index(ax)                                          # column: z=0, y=1, x=2
        a = axs[2*i+1][j]
        a.plot(xs, p, "o", ms=3)
        a.plot(xs, fit, "-")
        a.set_title(f"bead {i} {ax}: {df.loc[i, RAW_COLS[j]]:.0f} nm", fontsize=8)

    fig.suptitle(name, fontsize=9)
    fig.tight_layout()
    PLOT_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOT_DIR / f"{Path(name).stem}_fits.png", dpi=150)
    plt.close(fig)                                  # free memory; many files in a row


# ---------- 4. loop over all files ----------
def main():
    # Make sure the image folder exists before looking for files in it
    if not FOLDER.is_dir():
        print(f"Image folder not found: {FOLDER.resolve()}\nCreate an 'img' folder next to the script and put your .tif files in it.")
        return

    # Collect every matching file (a set avoids duplicates), sorted by name
    files = sorted({p for pat in PATTERNS for p in FOLDER.glob(pat)})
    if not files:
        print(f"No .tif/.tiff files found in {FOLDER.resolve()}")
        return

    all_beads, summary = [], []     # all_beads: one table per file; summary: one dict per file
    for k, path in enumerate(files, 1):
        print(f"[{k}/{len(files)}] {path.name}")
        row = {"file": path.name, "status": "ok", "beads_found": 0, "beads_unflagged": 0}
        try:
            img, vz, vy, vx = load(path)
            rows, fits, crops = measure_beads(img, vz, vy, vx)
        except (ValueError, OSError, tifffile.TiffFileError) as e:   # unreadable / unusable file: log and move on
            print(f"  SKIPPED: {e}")
            row["status"] = f"skipped: {e}"
            summary.append(row)
            continue

        # Per-bead table for this file, plus the bead-size-corrected FWHM columns
        df = pd.DataFrame(rows, columns=["bead", "y_px", "x_px", *RAW_COLS, "flags"])
        for raw, corr in zip(RAW_COLS, CORR_COLS):
            df[corr] = bead_correct(df[raw])
        df.insert(0, "file", path.name)

        clean = df[df["flags"] == ""]                       # beads with no quality warnings
        row["beads_found"], row["beads_unflagged"] = len(df), len(clean)
        if len(df) == 0:
            row["status"] = "no beads found"
        for c in RAW_COLS:                                   # summary stats use unflagged beads only
            row[f"mean_{c}"] = clean[c].mean()
            row[f"median_{c}"] = clean[c].median()
            row[f"SD_{c}"] = clean[c].std(ddof=1)            # NaN if fewer than 2 beads
        for c in CORR_COLS:
            row[f"mean_{c}"] = clean[c].mean()
        summary.append(row)
        all_beads.append(df)
        print(f"  {len(df)} beads, {len(clean)} unflagged")

        if len(df):
            save_fit_plot(path.name, len(df), fits, df, crops, (vz, vy, vx))

    # Combine every file's bead table into one (or an empty table if nothing was measured)
    beads = pd.concat(all_beads, ignore_index=True) if all_beads else pd.DataFrame(columns=BEAD_COLS)
    summ = pd.DataFrame(summary)

    # ---------- 5. write Excel ----------
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        beads[BEAD_COLS].round(1).to_excel(xw, sheet_name="Beads", index=False)
        summ.round(1).to_excel(xw, sheet_name="Summary", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"                           # keep header visible
            for col in ws.columns:                           # widen columns to fit content
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 45)
    print(f"\nDone. Results: {OUT_XLSX.resolve()}")


if __name__ == "__main__":
    main()