# Bead PSF / Resolution Measurement

Batch measurement of microscope resolution from sub-resolution fluorescent bead stacks. The script finds every bead in each 3D TIFF stack, fits a Gaussian to the intensity profile through the bead along z, y and x, and reports the full width at half maximum (FWHM) in nanometres. It also removes the bead's own size from the result and writes everything to one Excel workbook, along with fit-check plots.

## Why

Sub-resolution beads act as point sources, so the size of the spot in the image is a direct measure of the microscope's point spread function (PSF). Measuring many beads across many files by hand is slow, so this script automates it and keeps a visual record of every fit so you can check the numbers by eye.

## What it does

For each TIFF stack in the `img` folder:

1. Reads the 3D image and its voxel size (µm per pixel in z, y, x) from the file metadata.
2. Finds beads by max-projecting along z, smoothing, and applying an Otsu threshold. Each bright blob is treated as one bead.
3. Crops a small 3D box around each bead and locates the true 3D intensity peak.
4. Fits a 1D Gaussian (with constant background) to the z, y and x profiles through the peak and converts the width to a FWHM.
5. Subtracts the bead's physical size in quadrature to estimate the microscope's own FWHM.
6. Saves the results to Excel and a plot per file.

## Requirements

- Python 3.9 or newer
- numpy, scipy, pandas, matplotlib, scikit-image, tifffile, openpyxl

```bash
pip install numpy scipy pandas matplotlib scikit-image tifffile openpyxl
```

## Folder layout

```
my_folder/
├── bead_fwhm_batch.py
├── img/                  <- put your .tif / .tiff stacks here
├── fwhm_results.xlsx     <- created by the script
└── fit_plots/            <- created by the script
```

## Usage

Put your bead stacks in an `img` folder next to the script, then run:

```bash
python bead_fwhm_batch.py
```

The `img` folder is found relative to the script, so it works from any terminal location. To read a different folder instead:

```bash
python bead_fwhm_batch.py "C:/path/to/folder"
```

Results are always saved next to the script, not inside the image folder.

## Input requirements

- 3D stacks (Z, Y, X) saved as `.tif` or `.tiff`. Singleton axes such as channel or time are dropped automatically, but the data must be 3D after that.
- Voxel size must be stored in the file metadata, in either:
  - **OME-TIFF**: `PhysicalSizeX/Y/Z` with units, or
  - **ImageJ / Fiji TIFF**: z spacing plus x/y resolution tags.
- Beads should be well separated. Touching beads merge into one blob and will usually be flagged.

Files that can't be used (for example, missing voxel size) are skipped and logged in the Summary sheet, and the rest of the batch continues.

## Settings

Edit these at the top of the script:

| Setting | Default | Meaning |
|---|---|---|
| `BEAD_D` | `0.100` | Nominal bead diameter in µm |
| `D_EFF` | `BEAD_D * √3/2` | FWHM contributed by the bead itself (about 87 nm for 100 nm beads). Use `BEAD_D` for a conservative correction |
| `HALF_BOX` | `0.5` | Half-width of the crop around each bead in µm |
| `MIN_R2` | `0.95` | Fits with R² below this are flagged |
| `PATTERNS` | `*.tif`, `*.tiff` | File types to pick up |

Set `BEAD_D` to match your beads. For example, `0.040` for 40 nm beads.

## Output

### `fwhm_results.xlsx`

**Beads sheet** (one row per bead, all files together)

| Column | Description |
|---|---|
| `file`, `bead` | Source file and bead index within it |
| `y_px`, `x_px` | Bead position in pixels |
| `FWHM_z_nm`, `FWHM_y_nm`, `FWHM_x_nm` | Measured FWHM |
| `FWHM_*_corr_nm` | FWHM after removing the bead size (NaN if the measured value is smaller than the bead) |
| `flags` | Quality warnings (empty means none) |

**Summary sheet** (one row per file)

Status, number of beads found, number unflagged, and the mean, median and SD of the raw FWHM per axis, plus the mean of the corrected FWHM. Summary statistics use unflagged beads only.

### Quality flags

| Flag | Meaning |
|---|---|
| `z:R2=0.85` | Poor Gaussian fit on that axis |
| `x:wide` | FWHM spans nearly the whole crop (often a clump or out-of-focus bead) |
| `y:fail` | Fit did not converge |

### `fit_plots/<file>_fits.png`

One PNG per file. For each bead, the top row shows XZ, XY and YZ views through the fitted peak (axes in µm, red crosshairs at the peak) and the row below shows the z, y and x profiles with the Gaussian fit overlaid and the FWHM in the title. By default the images use `aspect="auto"`, so z can look stretched. Change it to `"equal"` in `save_fit_plot` for true proportions.

## Notes and limitations

- Only the first image series in each TIFF is read.
- Beads too close to the image edge for a full crop are skipped.
- Bead finding uses a single global Otsu threshold, so images with very uneven brightness or dense bead clusters may need manual checking.
- Always check the fit plots, especially for flagged beads, before reporting numbers.
- Workbook values are rounded to 0.1 nm.
