import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(42)

# ---- Parameters -------------------------------------------------------
px_xy, px_z = 20.0, 30.0                 # voxel size (nm)
shape = (64, 128, 128)                   # Z, Y, X
bead_diam_nm = 100.0
fwhm_xy_nm, fwhm_z_nm = 85.0, 180.0      # set to your TauSTED / confocal values
peak_counts, background = 1000, 5
ss = 4                                   # supersampling factor for smooth bead edges
# -----------------------------------------------------------------------

# 1. Single bead centre (voxel units) at the middle of the volume
centres = np.array([[s // 2 for s in shape]], dtype=float)

# 2. Render each bead locally with supersampling (fast and anti-aliased)
vol = np.zeros(shape, np.float32)
half = (4, 5, 5)                         # window half-size in voxels
off = (np.arange(ss) + 0.5) / ss - 0.5   # sub-voxel sample offsets
scale = np.array([px_z, px_xy, px_xy])

for c in centres:
    lo = np.floor(c).astype(int) - half
    hi = np.floor(c).astype(int) + half + 1
    axes = []
    for a in range(3):
        idx = np.arange(lo[a], hi[a])
        axes.append(((idx[:, None] + off[None, :]).ravel() - c[a]) * scale[a])
    Z, Y, X = np.meshgrid(*axes, indexing="ij")
    inside = (np.sqrt(Z**2 + Y**2 + X**2) <= bead_diam_nm / 2).astype(np.float32)
    n = hi - lo
    patch = inside.reshape(n[0], ss, n[1], ss, n[2], ss).mean(axis=(1, 3, 5))
    vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += patch

# 3. Optical blur (FWHM in nm -> sigma in voxels)
f = 2 * np.sqrt(2 * np.log(2))
sigma = (fwhm_z_nm / f / px_z, fwhm_xy_nm / f / px_xy, fwhm_xy_nm / f / px_xy)
img = gaussian_filter(vol, sigma)

# 4. Scale, add background, apply Poisson noise
img = img / img.max() * peak_counts + background
img = rng.poisson(img).astype(np.uint16)

# 5. Save TIFF (with Fiji-readable calibration) and ground truth
tiff.imwrite(
    "simulated_bead.tif", img, imagej=True,
    resolution=(1 / (px_xy * 1e-3), 1 / (px_xy * 1e-3)),
    metadata={"axes": "ZYX", "spacing": px_z * 1e-3, "unit": "um"},
)
np.savetxt("bead_positions.csv", centres * scale, delimiter=",",
           header="z_nm,y_nm,x_nm", comments="")
print(f"Saved {len(centres)} bead -> simulated_bead.tif, bead_positions.csv")