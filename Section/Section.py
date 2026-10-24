"""
STED section comparison: BM axial profile, lateral FWHM, SNR.   (fully automatic)

FOLDER LAYOUT
    <any folder>/sted_section_analysis.py     <- this script
    <any folder>/sections/                    <- put your .tif stacks here (sub-folders too)
    <any folder>/sted_results/                <- created automatically for outputs

WHAT IT DOES AUTOMATICALLY
  * finds the .tif files in 'sections'; reads thickness and pinhole from the file name
  * reads pixel size and z-step from each file's metadata (OME / ImageJ / TIFF tags)
  * finds the basement membrane (BM) in each stack with a ridge filter, and measures
    its width on many profiles taken perpendicular to it along its whole length
  * picks a background region automatically (darkest area) for the SNR
  * saves a *_QC.png per stack showing exactly what it detected - LOOK AT THESE

Needs:  pip install numpy scipy pandas matplotlib tifffile scikit-image
Just press run. Only touch LINES below if a QC image shows the detection missed the BM.
"""
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter
from scipy.ndimage import shift as nd_shift
from scipy.optimize import OptimizeWarning, curve_fit
from skimage.filters import sato
from skimage.measure import label, profile_line, regionprops
from skimage.morphology import remove_small_objects, skeletonize

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================ SETTINGS ====================================
PREVIEW_ONLY = False       # True = only save detection overlays (no measurements)

SECTIONS_FOLDER = "sections"   # folder (next to this script) holding the .tif stacks
SKIP_WORDS = ["repeat"]        # files with these words in the name are skipped
CHANNEL = 0                    # channel to analyse if a file has several

PIXEL_SIZE_NM = None   # None = read from each file. Type a number (nm) to force it.
Z_STEP_NM = None       # None = read from each file. Type a number (nm) to force it.

# --- automatic BM detection ---
AUTO_DETECT = True
AUTO_RIDGE_SIGMAS_NM = (80, 150, 250)  # ridge-filter scales (about BM half-width)
AUTO_GUIDE_SMOOTH_NM = 80      # smoothing of the detection image (noise reduction)
AUTO_GRADIENT_SCALE_NM = 100   # scale of the gradients used to find the BM direction
AUTO_ORIENT_SCALE_NM = 500     # direction is averaged over this distance along the BM
AUTO_MIN_COHERENCE = 0.3       # skip spots where the BM direction is unclear (0-1)
AUTO_RIDGE_PERCENTILE = 92     # keep the strongest (100 - this)% ridge pixels; lower = more
AUTO_MIN_LENGTH_NM = 1500      # ignore detected ridges shorter than this
AUTO_SPACING_NM = 200          # distance between profiles along the BM
AUTO_HALF_LEN_NM = 600         # each profile extends this far either side of the BM
AUTO_MAX_PROFILES = 80
AUTO_LINE_WIDTH_PX = 3         # averaging width along the BM
AUTO_CENTRE_WINDOW_NM = 200    # BM peak must lie within this of the profile centre
BG_AUTO_SIZE_NM = 800          # side of the automatic background box

# --- manual override (only needed if a QC image shows the detection is wrong) ---
# Key = any piece of the file name, value = line ACROSS the BM (x0, y0, x1, y1) in pixels.
LINES = {
    # "0.5 AU 250nm": (120, 60, 120, 110),
}
LINE = (100, 80, 100, 120)   # used for every file only if AUTO_DETECT = False
LINE_WIDTH_PX = 5
N_PROFILES = 5
PROFILE_SPACING_PX = 6
BG_ROI = (20, 20, 40, 40)    # (x, y, w, h) background box for manual lines

# --- fitting ---
SMOOTH_SIGMA_PX = 1.0
MIN_R2 = 0.6               # fits with lower R^2 are rejected
MIN_FWHM_NM = 40           # fits narrower/wider than these are rejected
MAX_FWHM_NM = 800
IN_FOCUS_FRACTION = 0.5    # "in-focus" planes = BM signal >= this fraction of the maximum
MIN_SNR_FOR_FIT = 3        # planes with lower SNR get no FWHM (they are just noise)
# ==========================================================================

UNIT_NM = {"nm": 1.0, "nanometer": 1.0, "nanometers": 1.0,
           "um": 1e3, "\u00b5m": 1e3, "\u03bcm": 1e3, "micron": 1e3, "microns": 1e3,
           "micrometer": 1e3, "micrometers": 1e3, "mm": 1e6, "cm": 1e7,
           "centimeter": 1e7, "inch": 2.54e7, "in": 2.54e7, "m": 1e9}


# ------------------------- finding files ---------------------------------
def script_folder():
    try:
        return Path(__file__).resolve().parent
    except NameError:                      # running line-by-line in a console
        return Path.cwd()


def find_stacks():
    base = script_folder()
    sec = next((d for d in base.iterdir()
                if d.is_dir() and d.name.lower() == SECTIONS_FOLDER.lower()), None)
    if sec is None:
        raise FileNotFoundError(
            f"No '{SECTIONS_FOLDER}' folder found next to the script ({base}). "
            f"Create it and put your .tif files inside.")
    def tifs(folder, recursive):
        it = folder.rglob("*") if recursive else folder.glob("*")
        return [p for p in sorted(it) if p.suffix.lower() in (".tif", ".tiff")
                and not any(w in p.stem.lower() for w in SKIP_WORDS)]

    candidates = tifs(sec, True)
    if not candidates:
        beside = tifs(base, False)
        if beside:
            print(f"NOTE: no .tif files in '{sec.name}', using the {len(beside)} .tif file(s) "
                  f"found next to the script instead. Move them into '{sec.name}' to silence this.")
            candidates, sec = beside, base
        else:
            raise FileNotFoundError(
                f"No .tif files found in {sec} (or next to the script in {base}).\n"
                f"Contents of '{sec.name}': {[p.name for p in sec.iterdir()] or 'empty'}\n"
                f"Copy your exported .tif stacks into that folder. If they are .lif/.lof files, "
                f"export them as TIFF first.")
    found = []
    for p in candidates:
        t = re.search(r"(\d+)\s*nm", p.stem, re.IGNORECASE)
        ph = (re.search(r"(?:PH|pinhole)[_\s-]*(\d+(?:\.\d+)?)", p.stem, re.IGNORECASE)
              or re.search(r"(\d+(?:\.\d+)?)\s*AU", p.stem, re.IGNORECASE))
        found.append((str(p), int(t.group(1)) if t else np.nan,
                      float(ph.group(1)) if ph else np.nan))
    print(f"Looking in: {sec}")
    for f, t, ph in found:
        bad = "" if np.isfinite(t) and np.isfinite(ph) else "   <-- could not read thickness/pinhole"
        print(f"  {Path(f).name}: {t} nm, {ph} AU{bad}")
    return base, found


# ------------------------- calibration -----------------------------------
def to_nm(value, unit):
    k = UNIT_NM.get(str(unit).strip().lower())
    return float(value) * k if k else None


def read_calibration(path):
    """Return dict(px_nm, z_nm, source). Values are None if not found in the file."""
    px = z = None
    src = []
    with tifffile.TiffFile(path) as tf:
        if tf.is_ome and tf.ome_metadata:                      # 1) OME-TIFF
            try:
                for el in ET.fromstring(tf.ome_metadata).iter():
                    if el.tag.split("}")[-1] == "Pixels":
                        a = el.attrib
                        if "PhysicalSizeX" in a:
                            px = to_nm(a["PhysicalSizeX"], a.get("PhysicalSizeXUnit", "\u00b5m"))
                        if "PhysicalSizeZ" in a:
                            z = to_nm(a["PhysicalSizeZ"], a.get("PhysicalSizeZUnit", "\u00b5m"))
                        break
                if px or z:
                    src.append("OME")
            except Exception:  # noqa: BLE001, S110
                pass
        ij = tf.imagej_metadata if tf.is_imagej else None      # 2) ImageJ / Fiji
        if ij and z is None and ij.get("spacing") and ij.get("unit"):
            z = to_nm(abs(float(ij["spacing"])), ij["unit"])
            if z:
                src.append("ImageJ z-spacing")
        page = tf.pages[0]                                     # 3) TIFF resolution tags
        if px is None and "XResolution" in page.tags:
            num, den = page.tags["XResolution"].value
            if num:
                unit = None
                if ij and ij.get("unit"):
                    unit = ij["unit"]
                elif "ResolutionUnit" in page.tags:
                    v = page.tags["ResolutionUnit"].value
                    unit = {2: "inch", 3: "cm"}.get(int(getattr(v, "value", v)))
                px = to_nm(den / num, unit) if unit else None
                if px:
                    src.append("TIFF tags")
    return {"px_nm": px, "z_nm": z, "source": ", ".join(src) if src else "none found"}


def get_calibration(path):
    cal = read_calibration(path)
    px = PIXEL_SIZE_NM if PIXEL_SIZE_NM is not None else cal["px_nm"]
    z = Z_STEP_NM if Z_STEP_NM is not None else cal["z_nm"]
    if px is None or z is None:
        missing = " and ".join(n for n, v in (("pixel size", px), ("z-step", z)) if v is None)
        raise ValueError(f"no {missing} in the file's metadata. Set PIXEL_SIZE_NM / "
                         f"Z_STEP_NM at the top of the script (Leica: LAS X > image properties).")
    for name, v, lo, hi in (("pixel size", px, 1, 1000), ("z-step", z, 10, 5000)):
        if not lo <= v <= hi:
            print(f"  WARNING {Path(path).name}: {name} = {v:.1f} nm looks odd - check units.")
    return px, z, cal["source"]


# ------------------------- loading ---------------------------------------
def load_stack(path):
    """Return array ordered (Z, Y, X); extra axes (channel, time...) are reduced."""
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        arr, axes = s.asarray().astype(float), s.axes
    if "Z" not in axes:                     # unknown / time axis used as the z axis
        for cand in "TQ":
            if cand in axes:
                axes = axes.replace(cand, "Z", 1)
                break
    for a in list(axes):
        if a in "ZYX":
            continue
        i = axes.index(a)
        idx = CHANNEL if a == "C" else 0
        arr = np.take(arr, min(idx, arr.shape[i] - 1), axis=i)
        axes = axes.replace(a, "", 1)
    if "Z" not in axes:
        arr, axes = arr[None], "Z" + axes
    return np.moveaxis(arr, [axes.index(a) for a in "ZYX"], [0, 1, 2])


# ------------------------- BM geometry -----------------------------------
def parallel_lines(line, n, spacing):
    x0, y0, x1, y1 = line
    d = np.array([x1 - x0, y1 - y0], float)
    normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    offsets = (np.arange(n) - (n - 1) / 2) * spacing
    return [((x0 + o * normal[0], y0 + o * normal[1]),
             (x1 + o * normal[0], y1 + o * normal[1])) for o in offsets]


def manual_geometry(line):
    return {"mode": "manual", "segments": parallel_lines(line, N_PROFILES, PROFILE_SPACING_PX),
                "bg": BG_ROI, "width": LINE_WIDTH_PX, "window": None, "main": line}


def auto_geometry(stack, px):
    """Find the BM with a ridge filter; return profile segments perpendicular to it."""
    # guide image = mean of the brighter half of the planes
    p99 = np.array([np.percentile(pl, 99) for pl in stack])
    guide = gaussian_filter(stack[p99 >= np.median(p99)].mean(axis=0),
                            max(1.0, AUTO_GUIDE_SMOOTH_NM / px))
    H, W = guide.shape

    sig = [max(1.0, s / px) for s in AUTO_RIDGE_SIGMAS_NM]
    ridge = sato(guide, sigmas=sig, black_ridges=False)
    mask = remove_small_objects(ridge > np.percentile(ridge, AUTO_RIDGE_PERCENTILE), 20)
    lab = label(skeletonize(mask), connectivity=2)
    min_len = AUTO_MIN_LENGTH_NM / px
    keep = np.zeros(lab.shape, bool)
    for r in regionprops(lab):
        if r.area >= min_len:
            keep[lab == r.label] = True
    pts = np.argwhere(keep)
    if len(pts) == 0:
        raise ValueError("automatic BM detection found nothing (add this file to LINES, or "
                         "lower AUTO_RIDGE_PERCENTILE / AUTO_MIN_LENGTH_NM)")

    spacing = max(2.0, AUTO_SPACING_NM / px)       # thin the points out evenly
    chosen = []
    for i in np.random.default_rng(0).permutation(len(pts)):
        if all(np.hypot(*(pts[i] - q)) >= spacing for q in chosen):
            chosen.append(pts[i])

    half = AUTO_HALF_LEN_NM / px
    # local BM direction from the structure tensor of the image gradients, averaged
    # over AUTO_ORIENT_SCALE_NM so single noisy pixels cannot twist the profile lines
    sd, rho = max(1.0, AUTO_GRADIENT_SCALE_NM / px), max(2.0, AUTO_ORIENT_SCALE_NM / px)
    gy, gx = gaussian_filter(guide, sd, order=(1, 0)), gaussian_filter(guide, sd, order=(0, 1))
    Jxx, Jyy, Jxy = (gaussian_filter(gx * gx, rho), gaussian_filter(gy * gy, rho),
                     gaussian_filter(gx * gy, rho))
    theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)            # direction ACROSS the BM
    coherence = np.hypot(Jxx - Jyy, 2 * Jxy) / (Jxx + Jyy + 1e-12)
    segs = []
    for r, c in chosen:
        if coherence[r, c] < AUTO_MIN_COHERENCE:
            continue
        dx, dy = np.cos(theta[r, c]), np.sin(theta[r, c])
        a, b = (c - dx * half, r - dy * half), (c + dx * half, r + dy * half)
        if all(0 <= p[0] <= W - 1 and 0 <= p[1] <= H - 1 for p in (a, b)):
            segs.append((a, b))
    if len(segs) < 5:
        raise ValueError(f"only {len(segs)} usable BM profiles found (add this file to LINES)")
    if len(segs) > AUTO_MAX_PROFILES:
        segs = [segs[i] for i in np.linspace(0, len(segs) - 1, AUTO_MAX_PROFILES).astype(int)]

    win = int(max(10, BG_AUTO_SIZE_NM / px))       # background = darkest box
    m = uniform_filter(guide, size=win, mode="nearest")
    h = win // 2
    r0, c0 = np.unravel_index(np.argmin(m[h:H - h, h:W - h]), (H - 2 * h, W - 2 * h))
    return {"mode": "auto", "segments": segs, "bg": (c0, r0, win, win),
                "width": AUTO_LINE_WIDTH_PX, "window": AUTO_CENTRE_WINDOW_NM / px, "main": None}


def find_geometry(path, stack, px):
    name = Path(path).name
    line = next((v for k, v in LINES.items() if k in name), None)
    if line is None and not AUTO_DETECT:
        line = LINE
    return manual_geometry(line) if line is not None else auto_geometry(stack, px)


# ------------------------- fitting helpers -------------------------------
def get_profiles(plane, geom):
    """One intensity profile per segment, all resampled to the same length."""
    (a0, b0) = geom["segments"][0]
    n = round(np.hypot(b0[0] - a0[0], b0[1] - a0[1])) + 1
    out = []
    for a, b in geom["segments"]:
        p = profile_line(plane, (a[1], a[0]), (b[1], b[0]), linewidth=geom["width"],
                         mode="reflect")
        out.append(np.interp(np.linspace(0, len(p) - 1, n), np.arange(len(p)), p))
    return out


def gauss(x, amp, mu, sigma, offset):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset


def fit_gauss(profile, px_nm, window=None):
    """Bounded Gaussian+offset fit. Returns (FWHM nm, R2, params) or (nan, nan, None)."""
    n = len(profile)
    x = np.arange(n)
    amp0 = profile.max() - profile.min()
    if amp0 <= 0:
        return np.nan, np.nan, None
    lo, hi = (n // 2 - window, n // 2 + window) if window else (0, n - 1)
    lo, hi = max(0, lo), min(n - 1, hi)
    s = gaussian_filter1d(profile, SMOOTH_SIGMA_PX)
    mu0 = lo + np.argmax(s[int(lo):int(hi) + 1])
    try:
        popt, _ = curve_fit(gauss, x, profile, p0=[amp0, mu0, 2.0, profile.min()],
                            bounds=([0, lo, 0.5, -np.inf], [np.inf, hi, n / 3, np.inf]),
                            maxfev=5000)
    except Exception:  # noqa: BLE001
        return np.nan, np.nan, None
    r2 = 1 - np.var(profile - gauss(x, *popt)) / np.var(profile) if np.var(profile) > 0 else np.nan
    return 2.3548 * popt[2] * px_nm, r2, popt


def aligned_mean(profiles, window=None):
    """Shift each profile so its smoothed peak is centred, then average (noise-robust)."""
    n = len(profiles[0])
    centre = n // 2
    lo, hi = (max(0, int(centre - window)), min(n, int(centre + window) + 1)) if window else (0, n)
    out = []
    for p in profiles:
        pk = lo + np.argmax(gaussian_filter1d(p, SMOOTH_SIGMA_PX)[lo:hi])
        out.append(nd_shift(p, centre - pk, order=1, mode="nearest"))
    return np.mean(out, axis=0)


def fwhm_halfmax(y, step):
    """Model-free FWHM from half-maximum crossings (works for plateaus)."""
    y = np.asarray(y, float)
    y = y - np.nanmin(y)
    if not np.isfinite(y).all() or y.max() <= 0:
        return np.nan
    half = y.max() / 2
    above = np.where(y >= half)[0]
    i0, i1 = above[0], above[-1]
    left = i0 - 1 + (half - y[i0 - 1]) / (y[i0] - y[i0 - 1]) if i0 > 0 else i0
    right = i1 + (y[i1] - half) / (y[i1] - y[i1 + 1]) if i1 < len(y) - 1 else i1
    return (right - left) * step


def draw_overlay(ax, img, geom, title, px=None):
    lo, hi = np.percentile(img, [1, 99.7])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    for a, b in geom["segments"]:
        ax.plot([a[0], b[0]], [a[1], b[1]], c="orange", lw=0.6)
    if geom["main"] is not None:
        m = geom["main"]
        ax.plot([m[0], m[2]], [m[1], m[3]], c="red", lw=1.2)
    bx, by, bw, bh = geom["bg"]
    ax.add_patch(plt.Rectangle((bx, by), bw, bh, fill=False, ec="cyan", lw=1.2))
    if px:                                          # scale bar, bottom-right
        width_nm = max(w for w in (200, 500, 1000, 2000, 5000) if w <= max(200, img.shape[1] * px / 4))
        x1, y1 = img.shape[1] - 0.04 * img.shape[1], img.shape[0] - 0.05 * img.shape[0]
        ax.plot([x1 - width_nm / px, x1], [y1, y1], c="white", lw=3)
        label = f"{width_nm / 1000:g} \u00b5m" if width_nm >= 1000 else f"{width_nm} nm"
        ax.text(x1 - width_nm / px / 2, y1 - 0.02 * img.shape[0], label, c="white",
                ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


# ------------------------- preview ---------------------------------------
def preview(path, out):
    stack = load_stack(path)
    try:
        px = get_calibration(path)[0]
    except ValueError:
        px = PIXEL_SIZE_NM or 20.0
        print(f"  (preview of {Path(path).name} assumes a {px:.0f} nm pixel)")
    geom = find_geometry(path, stack, px)
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_overlay(ax, stack.max(axis=0), geom,
                 f"{Path(path).name}\n{geom['mode']} detection, {len(geom['segments'])} profiles "
                 f"(orange), background (cyan)", px)
    fig.tight_layout()
    fig.savefig(out / f"{Path(path).stem}_preview.png", dpi=130)
    plt.close(fig)


# ------------------------- analysis --------------------------------------
def analyse_stack(path, thickness, pinhole, out):
    px, zstep, cal_src = get_calibration(path)
    stack = load_stack(path)
    geom = find_geometry(path, stack, px)
    nz = stack.shape[0]
    z_nm = np.arange(nz) * zstep
    bx, by, bw, bh = geom["bg"]
    win = geom["window"]

    rows, mean_profiles, mean_fits = [], [], []
    for z in range(nz):
        plane = stack[z]
        bg = plane[by:by + bh, bx:bx + bw]
        profiles = get_profiles(plane, geom)

        good = []                                   # individual fits -> spread (SD)
        for p in profiles:
            f, r2, _ = fit_gauss(p, px, win)
            if np.isfinite(f) and MIN_FWHM_NM < f < MAX_FWHM_NM and r2 >= MIN_R2:
                good.append(f)

        mp = aligned_mean(profiles, win)            # fit of aligned mean -> main value
        f_avg, r2_avg, popt = fit_gauss(mp, px, win)
        signal = gaussian_filter1d(mp, SMOOTH_SIGMA_PX).max() - bg.mean()
        snr = signal / bg.std() if bg.std() > 0 else np.nan
        ok = (np.isfinite(f_avg) and MIN_FWHM_NM < f_avg < MAX_FWHM_NM and r2_avg >= MIN_R2
              and snr >= MIN_SNR_FOR_FIT)
        rows.append({"z": z, "z_nm": z_nm[z], "whole_image_sum": plane.sum(), "bm_signal": signal,
                         "fwhm_nm": f_avg if ok else np.nan, "fit_R2": r2_avg,
                         "fwhm_sd_nm": np.std(good) if len(good) > 1 else np.nan,
                         "n_profiles": len(profiles), "n_good_individual_fits": len(good),
                         "snr": snr})
        mean_profiles.append(mp)
        mean_fits.append(popt if ok else None)

    df = pd.DataFrame(rows)
    best = int(df.bm_signal.idxmax())
    in_focus = df.bm_signal >= IN_FOCUS_FRACTION * df.bm_signal.max()
    axial_fwhm = fwhm_halfmax(df.bm_signal.clip(lower=0).values, zstep)
    stem = Path(path).stem
    df.to_csv(out / f"{stem}_per_plane.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    draw_overlay(ax[0, 0], stack[best], geom,
                 f"best plane z={z_nm[best]:.0f} nm | {geom['mode']} detection, "
                 f"{len(geom['segments'])} profiles (orange); background box (cyan)", px)
    x = (np.arange(len(mean_profiles[best])) - len(mean_profiles[best]) // 2) * px
    ax[0, 1].plot(x, mean_profiles[best], "k.-", label="aligned mean profile")
    if mean_fits[best] is not None:
        xp = np.arange(len(mean_profiles[best]))
        ax[0, 1].plot(x, gauss(xp, *mean_fits[best]), "r-", label=f"fit {df.fwhm_nm[best]:.0f} nm")
    else:
        ax[0, 1].text(0.05, 0.9, "FIT FAILED", transform=ax[0, 1].transAxes, color="r")
    ax[0, 1].set(title="Intensity profile across the BM (best plane)",
                 xlabel="Distance from BM centre (nm)", ylabel="Intensity (counts)")
    ax[0, 1].legend()
    ax[1, 0].plot(z_nm, df.bm_signal, "o-")
    ax[1, 0].axvline(z_nm[best], ls="--", c="grey")
    ax[1, 0].set(title=f"BM signal through the stack (axial FWHM {axial_fwhm:.0f} nm)",
                 xlabel="z position in stack (nm)", ylabel="BM peak above background (counts)")
    ax[1, 1].errorbar(z_nm, df.fwhm_nm, df.fwhm_sd_nm, fmt="o-")
    ax[1, 1].set(title="BM width at each z plane",
                 xlabel="z position in stack (nm)", ylabel="BM width, FWHM (nm) \u00b1 SD of profiles")
    fig.suptitle(f"{Path(path).name}\npixel {px:.1f} nm, z-step {zstep:.0f} nm ({cal_src})", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{stem}_QC.png", dpi=130)
    plt.close(fig)

    b = df.iloc[best]
    return {"file": Path(path).name, "thickness_nm": thickness, "pinhole_AU": pinhole,
                "pixel_nm": px, "z_step_nm": zstep, "detection": geom["mode"], "n_profiles": len(geom["segments"]),
                "best_z_nm": z_nm[best], "axial_FWHM_nm": axial_fwhm,
                "FWHM_at_best_nm": b.fwhm_nm, "FWHM_SD_nm": b.fwhm_sd_nm,
                "FWHM_infocus_median_nm": df.fwhm_nm[in_focus].median(),
                "n_infocus_planes": int(in_focus.sum()),
                "n_infocus_good_fits": int(df.fwhm_nm[in_focus].notna().sum()),
                "SNR_at_best": b.snr, "BM_signal_at_best": b.bm_signal}


def bleach_check(path_first, path_repeat):
    """Same field imaged before and after: % loss of BM signal at the best plane.
    The BM is detected on the first stack and the same profiles are used on both."""
    px = get_calibration(path_first)[0]
    s1, s2 = load_stack(path_first), load_stack(path_repeat)
    geom = find_geometry(path_first, s1, px)
    bx, by, bw, bh = geom["bg"]
    vals = []
    for s in (s1, s2):
        sig = [gaussian_filter1d(aligned_mean(get_profiles(pl, geom), geom["window"]),
                                 SMOOTH_SIGMA_PX).max() - pl[by:by + bh, bx:bx + bw].mean()
               for pl in s]
        vals.append(max(sig))
    loss = 100 * (1 - vals[1] / vals[0])
    print(f"Bleaching: {loss:.1f}% loss of BM signal ({vals[0]:.1f} -> {vals[1]:.1f})")
    return loss


# ------------------------- main ------------------------------------------
if __name__ == "__main__":
    base, files = find_stacks()
    out = base / "sted_results"
    out.mkdir(exist_ok=True)

    cal_rows = []
    for p, _, _ in files:
        try:
            px, z, src = get_calibration(p)
        except ValueError as e:
            px = z = np.nan
            src = f"MISSING: {e}"
        cal_rows.append({"file": Path(p).name, "pixel_nm": px, "z_step_nm": z, "source": src})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(out / "calibration.csv", index=False)
    print("\nCalibration read from files:")
    print(cal.round(2).to_string(index=False))

    if PREVIEW_ONLY:
        for p, _, _ in files:
            try:
                preview(p, out)
            except Exception as e:  # noqa: BLE001
                print(f"PREVIEW FAILED {Path(p).name}: {e}")
        print(f"\nPreviews saved in {out}")
    else:
        results = []
        for p, t, ph in files:
            try:
                results.append(analyse_stack(p, t, ph, out))
            except Exception as e:  # noqa: BLE001
                print(f"SKIPPED {Path(p).name}: {e}")
        if not results:
            raise SystemExit("No stacks could be analysed - see the SKIPPED messages above.")
        table = pd.DataFrame(results)
        table.to_csv(out / "section_comparison.csv", index=False)
        print("\n" + table.round(1).to_string(index=False))

        print("\nPer-thickness summary:")
        print(table.groupby("thickness_nm").agg(
            FWHM_infocus_median=("FWHM_infocus_median_nm", "mean"),
            FWHM_range_over_pinholes=("FWHM_infocus_median_nm", lambda s: s.max() - s.min()),
            SNR=("SNR_at_best", "mean"),
            fits_ok=("n_infocus_good_fits", "sum")).round(1))

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for t, g in table.groupby("thickness_nm"):
            g = g.sort_values("pinhole_AU")
            ax[0].plot(g.pinhole_AU, g.FWHM_infocus_median_nm, "o-", label=f"{t} nm")
            ax[1].plot(g.pinhole_AU, g.SNR_at_best, "o-", label=f"{t} nm")
        pinholes = sorted(table.pinhole_AU.dropna().unique())
        ax[0].set(title="BM width vs pinhole", xlabel="Pinhole size (Airy units, AU)",
                  ylabel="BM width, FWHM (nm)\n(median of in-focus planes)", xticks=pinholes)
        ax[1].set(title="Signal quality vs pinhole", xlabel="Pinhole size (Airy units, AU)",
                  ylabel="SNR at best plane\n(BM peak above background / background SD)",
                  xticks=pinholes)
        for a in ax:
            a.grid(alpha=0.3)
        ax[0].legend(title="Section thickness")
        fig.tight_layout()
        fig.savefig(out / "comparison.png", dpi=150)
        plt.show()

        # bleach_check("sections/first.tif", "sections/first_repeat.tif")  # uncomment, set names