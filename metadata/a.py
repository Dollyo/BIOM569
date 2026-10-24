#!/usr/bin/env python3
"""
Extract acquisition metadata from Leica .lif files (LAS X, Stellaris STED /
TauSTED) into a CSV: one row per image series (one row per channel for
multi-channel series whose channels are separate sequence settings, e.g. a
"Mix" series holding Confocal / STED / TauSTED / Xtend channels).

Usage
-----
    python lif_metadata.py file1.lif [file2.lif ...] -o metadata.csv --operator "Dan"
    python lif_metadata.py file.lif            # no args -> file picker (or edit DEFAULT_FILES)
    python lif_metadata.py file.lif --no-peak  # skip peak-photon calculation
    python lif_metadata.py file.lif --dump [--series NAME] [--grep REGEX]

Dependencies: standard library only. Peak photons additionally uses
`readlif` + `numpy` if installed (pip install readlif numpy); otherwise blank.

Fields that cannot be in the file (Sample, Feature, Sub feature, Number,
Parameter changed vs previous run, Notes, Mean FWHM) are left blank.
"""
import argparse
import csv
import re
import struct
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

COLUMNS = [
    "Date", "Operator", "File name", "Sample", "Feature", "Sub feature", "Number",
    "Dye", "Objective", "Immersion media", "Mode", "Exc. line (nm)",
    "STED line (nm)", "Exc. power (%)", "STED power (%)", "3D STED strength (%)",
    "Gate start (ns)", "Gate end (ns)", "TauSTED strength (%)", "Xtend strength (%)",
    "Emission wavelength for pinhole (nm)", "Pinhole (AU)", "Detection window (nm)",
    "Scan speed (Hz)", "Bidirectional (Off/On)", "Pixel size (nm)", "Size x-y-z",
    "Voxel size (nm)", "Zoom", "Stack number", "Stack step size (nm)",
    "Line avg.", "Line accum.", "Frame avg.", "Frame accum.", "Peak photons",
    "Parameter changed vs previous run", "Notes", "Mean FWHM (nm)",
]


# --------------------------------------------------------------------------
# LIF header parsing (no external deps)
# --------------------------------------------------------------------------
def read_lif_header(path):
    """Return the XML header of a .lif file as an ElementTree root."""
    with open(path, "rb") as f:
        (test,) = struct.unpack("<I", f.read(4))
        if test != 0x70:
            raise ValueError("Not a LIF file (bad test value 0x70)")
        (hdr_len,) = struct.unpack("<I", f.read(4))
        if f.read(1) != b"\x2a":
            raise ValueError("Not a LIF file (bad second test value 0x2A)")
        (n_chars,) = struct.unpack("<I", f.read(4))
        xml = f.read(n_chars * 2).decode("utf-16-le")
    return ET.fromstring(xml)


def iter_images(root):
    """Yield (parent_names, name, Element, Image) for every series (document order)."""
    def walk(el, trail):
        for child in el.findall("./Children/Element"):
            name = child.get("Name", "")
            image = child.find("./Data/Image")
            if image is not None:
                yield trail, name, child, image
            yield from walk(child, trail + [name])
    for top in root.findall("./Element"):
        top_name = top.get("Name", "")
        image = top.find("./Data/Image")
        if image is not None:
            yield [], top_name, top, image
        yield from walk(top, [top_name])


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt(x, sig=5):
    """Compact number formatting (6.599, 17.22, 25, ...)."""
    if x is None or x == "":
        return ""
    if isinstance(x, str):
        f = to_float(x)
        if f is None:
            return x
        x = f
    return f"{x:.{sig}g}"


def get_dims(image):
    dims = {}
    for d in image.findall("./ImageDescription/Dimensions/DimensionDescription"):
        dims[int(d.get("DimID"))] = (int(d.get("NumberOfElements")), to_float(d.get("Length")) or 0.0)
    return dims


def step_nm(n, length_m):
    # Leica stores the physical length spanning (n - 1) steps.
    if n and n > 1 and length_m:
        return abs(length_m) / (n - 1) * 1e9
    return None


def get_timestamps(image):
    """List of local-time strings, one per timestamp token in TimeStampList."""
    ts = image.find("./TimeStampList")
    out = []
    if ts is None or not (ts.text or "").strip():
        return out
    for tok in ts.text.split():
        try:
            ticks = (int(tok.split(":")[0], 16) << 32 | int(tok.split(":")[1], 16)) if ":" in tok else int(tok, 16)
            dt = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks / 10)
            out.append(dt.astimezone().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            out.append("")
    return out


def clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


# --------------------------------------------------------------------------
# Hardware-setting extraction
# --------------------------------------------------------------------------
def hardware_setting(image):
    for a in image.findall("./Attachment"):
        if a.get("Name") == "HardwareSetting":
            return a
    return None


def setting_blocks(hw):
    """(top, [list_entries]) : top = the setting the image was acquired with,
    list_entries = the ATLConfocalSettingDefinition of every sequence in the
    LDM block (one per channel for multi-sequence 'Mix'-style series)."""
    top = hw.find("./ATLConfocalSettingDefinition")
    lst = []
    for blk in hw.iter("LDM_Block_Sequential_List"):
        lst.extend(blk.findall("./ATLConfocalSettingDefinition"))
    return top, lst


def active_detector(defn):
    for d in defn.iter("Detector"):
        if d.get("IsActive") == "1":
            return d
    return None


def laser_lines(defn):
    """-> (exc [(wl, pct)], sted [(wl, pct)]) for lines that are visible AND >0 %,
    plus the STED depletion wavelength."""
    exc, sted = [], []
    for aotf in defn.iter("Aotf"):
        is_sted = "sted" in (aotf.get("LightSourceName") or "").lower()
        for ls in aotf.iter("LaserLineSetting"):
            p = to_float(ls.get("IntensityDev")) or 0.0
            if ls.get("IsVisible") == "1" and p > 0:
                (sted if is_sted else exc).append((ls.get("LaserLine"), p))
    dep = defn.find(".//STED_DepletionLine")
    return exc, sted, (dep.get("Wavelength") if dep is not None else "")


def detection_window(defn, det):
    """Detection window (nm) of the active detector's channel (MultiBand)."""
    if det is None:
        return ""
    ch = det.get("Channel")
    for mb in defn.iter("MultiBand"):
        if mb.get("Channel") == ch:
            lo, hi = to_float(mb.get("LeftWorld")), to_float(mb.get("RightWorld"))
            if lo is not None and hi is not None:
                return f"{lo:.1f}-{hi:.1f}".replace(".0-", "-").replace(".0", "") if False else f"{round(lo)}-{round(hi)}"
    return ""


def beam_3d(defn, fallback=None):
    """3D STED strength is not stored as a number in the file. Returns 0 when only
    the xy vortex beam (STED Beam 1) is selected, else the beam-selection name."""
    for d in (defn, fallback):
        if d is None:
            continue
        for w in d.iter("Wheel"):
            if w.get("FilterWheelName") == "STED Beam Selection":
                name = clean(w.get("FilterName"))
                return "0" if name.endswith("1") else name
    return ""


def extract_setting(defn, fallback):
    """Fields that live inside one ATLConfocalSettingDefinition. `fallback` is the
    top-level definition used for attributes a list entry doesn't repeat."""
    def attr(name):
        v = defn.get(name)
        return v if v not in (None, "") else fallback.get(name)

    r = {}
    det = active_detector(defn)
    exc, sted, depl = laser_lines(defn)
    sted_pct = sum(p for _, p in sted)
    r["Dye"] = (det.get("DyeName") if det is not None else "") or ""
    r["Exc. line (nm)"] = " | ".join(w for w, _ in exc)
    r["Exc. power (%)"] = " | ".join(fmt(p, 4) for _, p in exc)
    r["STED line (nm)"] = (" | ".join(w for w, _ in sted) or depl) if sted_pct > 0 else ""
    r["STED power (%)"] = fmt(sted_pct, 4) if (sted or depl) else ""
    r["3D STED strength (%)"] = beam_3d(defn, fallback) if sted_pct > 0 else ""

    # Gating / TauSTED
    ic = det.find(".//ImageChannel") if det is not None else None
    gmode = ic.get("GatingMode") if ic is not None else ""
    tsd = det.find(".//TauStedDefinition") if det is not None else None
    r["Gate start (ns)"] = r["Gate end (ns)"] = ""
    if gmode == "3":                                    # hardware time gate
        dg = det.find(".//DigitalGating")
        if dg is not None:
            r["Gate start (ns)"] = fmt((to_float(dg.get("Begin")) or 0) * 1e9, 4)
            r["Gate end (ns)"] = fmt((to_float(dg.get("End")) or 0) * 1e9, 4)
    r["TauSTED strength (%)"] = r["Xtend strength (%)"] = ""
    xtend = False
    if gmode == "9" and tsd is not None:                # TauSTED
        r["TauSTED strength (%)"] = fmt((to_float(tsd.get("Strength")) or 0) * 100, 5)
        if tsd.get("TimegateActive") == "1":
            r["Gate start (ns)"] = fmt((to_float(tsd.get("TimegateBegin")) or 0) * 1e9, 4)
            r["Gate end (ns)"] = fmt((to_float(tsd.get("TimegateEnd")) or 0) * 1e9, 4)
        if tsd.get("Deconvolution") == "1":             # TauSTED + deconvolution = Xtend
            xtend = True
            try:
                r["Xtend strength (%)"] = ET.fromstring(tsd.get("DeconvolutionParameters")).findtext("Strength") or ""
            except Exception:
                pass

    # Mode label
    sted_on = sted_pct > 0
    if gmode == "9":
        r["Mode"] = "TauSTED Xtend" if xtend else "TauSTED"
    elif gmode == "3":
        r["Mode"] = "Gated STED" if sted_on else "Gated confocal"
    else:
        r["Mode"] = "STED" if sted_on else "Confocal"

    r["Emission wavelength for pinhole (nm)"] = fmt(attr("EmissionWavelengthForPinholeAiryCalculation"))
    r["Pinhole (AU)"] = fmt(attr("PinholeAiry"), 3)
    r["Detection window (nm)"] = detection_window(defn if defn.find(".//MultiBand") is not None else fallback, det)
    r["Scan speed (Hz)"] = fmt(attr("ScanSpeed"))
    r["Bidirectional (Off/On)"] = "On" if (attr("ScanDirectionXName") or "").lower().startswith("bi") else "Off"
    r["Zoom"] = fmt(attr("Zoom"), 4)
    r["Line avg."] = fmt(attr("LineAverage")) or "1"
    r["Line accum."] = fmt(attr("Line_Accumulation") or attr("LineAccumulation")) or "1"
    r["Frame avg."] = fmt(attr("FrameAverage")) or "1"
    r["Frame accum."] = fmt(attr("FrameAccumulation")) or "1"
    r["Objective"] = clean(attr("ObjectiveName"))
    r["Immersion media"] = clean(attr("Immersion"))
    return r


# --------------------------------------------------------------------------
# Peak photons (optional; needs readlif + numpy)
# --------------------------------------------------------------------------
class PeakReader:
    def __init__(self, path):
        self.lif = None
        try:
            from readlif.reader import LifFile
            self.lif = LifFile(str(path))
            self.names = [i["name"] for i in self.lif.image_list]
        except Exception:
            pass

    def peak(self, name, index, ch):
        if self.lif is None:
            return ""
        try:
            import numpy as np
            i = self.names.index(name) if self.names.count(name) == 1 else index
            img = self.lif.get_image(i)
            best = 0
            for z in range(img.dims.z):
                best = max(best, int(np.asarray(img.get_frame(z=z, t=0, c=ch)).max()))
            return best
        except Exception:
            return ""


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------
def build_rows(lif_path, trail, name, image, index, args, peak_reader):
    dims = get_dims(image)
    nx, lx = dims.get(1, (None, 0))
    ny, ly = dims.get(2, (None, 0))
    nz, lz = dims.get(3, (1, 0))
    px, py, pz = step_nm(nx, lx), step_nm(ny, ly), step_nm(nz, lz)
    nch = len(image.findall("./ImageDescription/Channels/ChannelDescription")) or 1
    times = get_timestamps(image)

    hw = hardware_setting(image)
    top, lst = setting_blocks(hw) if hw is not None else (None, [])
    if top is None:
        return []
    # one row per channel when each channel is its own sequence entry
    per_channel = nch > 1 and len(lst) == nch
    defs = [(d, f" [ch{i + 1} {d.get('UserSettingName', '').strip()}]") for i, d in enumerate(lst)] \
        if per_channel else [(top, "")]

    rows = []
    for ch, (defn, suffix) in enumerate(defs):
        row = {c: "" for c in COLUMNS}
        row.update(extract_setting(defn, top))
        row["Date"] = times[ch] if per_channel and ch < len(times) else (times[0] if times else "")
        row["Operator"] = args.operator
        row["File name"] = f"{Path(lif_path).name} / {'/'.join(trail + [name])}{suffix}"
        if nx:
            row["Size x-y-z"] = f"{nx} x {ny} x {nz}"
        if px:
            row["Pixel size (nm)"] = fmt(px, 4)
            row["Voxel size (nm)"] = " x ".join([fmt(px, 4), fmt(py, 4), fmt(pz, 4) if nz and nz > 1 and pz else "-"])
        row["Stack number"] = nz if nz else 1
        if pz and nz and nz > 1:
            row["Stack step size (nm)"] = fmt(pz, 4)
        if args.peak:
            row["Peak photons"] = peak_reader.peak(name, index, ch)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Dump mode (debugging)
# --------------------------------------------------------------------------
def dump(image, title, grep=None, out=None):
    from collections import Counter
    rx = re.compile(grep, re.I) if grep else None
    counts, order = Counter(), []
    hw = hardware_setting(image)
    for el in (hw.iter() if hw is not None else []):
        for k, v in el.attrib.items():
            line = f"{el.tag}.{k} = {v[:200]}"
            if rx and not rx.search(line):
                continue
            if line not in counts:
                order.append(line)
            counts[line] += 1
    text = f"\n=== {title} ===\n" + "\n".join(l + (f"   (x{counts[l]})" if counts[l] > 1 else "") for l in order)
    print(text)
    if out is not None:
        out.write(text + "\n")


# Optionally hard-code paths here to run with a bare `%run lif_metadata.py`
DEFAULT_FILES = []   # e.g. [r"C:\Users\you\STED\data\run1.lif"]


def pick_files():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        return list(filedialog.askopenfilenames(title="Select .lif files",
                                                filetypes=[("Leica LIF", "*.lif")]))
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help=".lif files (omit to pick files in a dialog)")
    ap.add_argument("-o", "--output", default="lif_metadata.csv")
    ap.add_argument("--operator", default="")
    ap.add_argument("--no-peak", dest="peak", action="store_false", help="skip peak-photon calculation")
    ap.add_argument("--dump", action="store_true", help="print HardwareSetting attributes (saved to lif_dump.txt) and exit")
    ap.add_argument("--series", default=None, help="with --dump: only the series with this exact name")
    ap.add_argument("--grep", default=None, help="with --dump: only lines matching this regex")
    args = ap.parse_args()

    if not args.files:
        args.files = list(DEFAULT_FILES) or pick_files()
        if not args.files:
            sys.exit("No .lif files selected.")

    rows = []
    dump_out = open("lif_dump.txt", "w", encoding="utf-8") if args.dump else None
    for f in args.files:
        try:
            root = read_lif_header(f)
        except Exception as e:
            print(f"[skip] {f}: {e}", file=sys.stderr)
            continue
        peak_reader = PeakReader(f) if (args.peak and not args.dump) else None
        for i, (trail, name, _, image) in enumerate(iter_images(root)):
            if args.dump:
                if args.series and name != args.series:
                    continue
                dump(image, f"{Path(f).name} / {'/'.join(trail + [name])}", args.grep, dump_out)
            else:
                rows.extend(build_rows(f, trail, name, image, i, args, peak_reader))
    if args.dump:
        dump_out.close()
        print("\nSaved to lif_dump.txt")
        return
    with open(args.output, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()