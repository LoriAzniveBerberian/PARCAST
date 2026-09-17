#!/usr/bin/env python3
"""
PARCAST - shared I/O, QC, dark-segment and cast-detection helpers.

Imported by parcast_EdzPAR_figures.py, parcast_EsPAR_figures.py and
parcast_noise_floor.py so the three scripts cannot drift apart on calibration
constants, QC bounds or dark-floor logic. Keep this file next to them.

CALIBRATION lives here and nowhere else.
"""
import os
import glob
import re
import numpy as np
import pandas as pd
from scipy import stats

__version__ = "6.7"   # bump when the day-level block or a calibration constant changes

# 6.4  IECF corrected from 1.32 to 1.25 for SQ-500 serial 6210. See CALIBRATION below.
# 6.5  Time axis rebuilt when several samples share one timestamp (second-resolution
#      firmware), so cast detection works on those files.
# 6.6  Figure typesetting: K, E and z in math italic, subscripts, PAR and units in
#      roman, matching cops_kd_figures. Added fig_depth_profile and fig_logx_profile.
# 6.7  SITES table and clock_shift_h, so site coordinates, time zone and the
#      logger-clock correction are defined once for every notebook and script.

# ============================================================================
# CALIBRATION  (single source of truth for all three scripts)
# ============================================================================
ED_SENSITIVITY = 100.0      # umol m-2 s-1 per mV, E_dzPAR serial 6210 (SQ-500-SS)
ES_SENSITIVITY = 100.0      # umol m-2 s-1 per mV, E_sPAR serial 6211 (SQ-500-SS)

# Apogee immersion effect correction factor for the SQ-500 series:
#   1.25 for serial numbers 2876 and above
#   1.32 for serial numbers 0-2875
# E_dzPAR is serial 6210, so 1.25 applies.
# Source: SQ-500 owner's manual p.17, rev 23-April-2026.
# The 1.32 value in Blonquist et al. was measured on SQ-500 serials 1049-1053
# (their Appendix A), all below the cutoff, so it does not apply to this unit.
IECF           = 1.25
LSB_mV         = 0.0078125  # ADS1115 GAIN_SIXTEEN: 1 LSB in mV
SAMPLE_RATE_HZ = 8.0        # nominal logging rate; overridden by the CSV header

# IECF applies ONLY to the in-water sensor. E_sPAR sits in air, so load_surface
# applies ES_SENSITIVITY alone and never multiplies by IECF.
#
# The factor was derived under a direct-beam lamp at normal incidence in a
# black-walled tub (Blonquist et al., their Fig. 1). Underwater the radiance
# distribution broadens with depth and becomes increasingly diffuse, a geometry it was
# not characterised for. Together with the SQ-500 cosine spec (+/-2 % at 45 deg,
# +/-5 % at 75 deg zenith) this is the dominant uncertainty on absolute in-water Ed,
# and it is larger and less quantified than the factor itself.
# check_header() warns if the firmware disagrees.

# ---- QC bounds -------------------------------------------------------------
# NOTE: both sides are bounded. The v4 scripts tested only `depth_m > -10` and
# `pressure_mbar > 0`, so a single I2C glitch row (depth 230,337 m, pressure
# 2.3e7 mbar) passed QC and was picked up as a cast.
PRESSURE_OK_MBAR = (800.0, 4000.0)
DEPTH_OK_M       = (-2.0, 50.0)
TEMP_MIN_C       = -10.0

IMU_UP_AXIS = "accel_y_ms2"

# ---- dark-segment detection ------------------------------------------------
DARK_MAX_UMOL       = 5.0    # a sample is "dark" if |PPFD(air)| is below this
DARK_MIN_DURATION_S = 60.0   # a dark run must last at least this long
DARK_EDGE_TRIM_S    = 5.0    # trim each end (cap on/off transitions)
DARK_BRIDGE_GAP_S   = 2.0    # bridge brief non-dark blips shorter than this


# ============================================================================
# TIME AXIS
# ============================================================================
def timestamp_resolution(iso_time):
    """Smallest non-zero step in a datetime column, in seconds.

    0.001 for millisecond firmware, 1.0 for second-resolution firmware.
    """
    ns = pd.to_datetime(iso_time).to_numpy("datetime64[ns]").astype("int64")
    d = np.diff(np.unique(ns))
    return float(d.min()) / 1e9 if d.size else np.nan


def build_time_axis(iso_time, nominal_hz=None):
    """Strictly increasing elapsed-seconds axis for one logged record.

    Returns (t_s, fs_hz, mode).

    mode "logged"        timestamps resolve individual samples (millisecond
                         firmware) and are used as written.
    mode "reconstructed" several samples share one timestamp, so the samples in
                         each one-second block are spaced evenly across it. The
                         block size sets the spacing, so 4 Hz and 8 Hz files
                         both come out right without being told the rate.

    find_casts takes np.gradient(depth, t_s). A repeated timestamp gives a zero
    step there, which returns inf velocities and zero detected casts.
    """
    ns = pd.to_datetime(iso_time).to_numpy("datetime64[ns]").astype("int64")
    n_dup = np.count_nonzero(np.diff(ns) == 0)

    if n_dup <= 0.01 * max(1, len(ns) - 1):
        t = (ns - ns[0]) / 1e9
        mode = "logged"
    else:
        sec = pd.Series(ns // 1_000_000_000)
        k = sec.groupby(sec).cumcount().to_numpy()      # position within the block
        n = sec.map(sec.value_counts()).to_numpy()      # samples in the block
        base = sec.to_numpy()
        t = (base - base[0]).astype(float) + k / n
        mode = "reconstructed"

    t = np.asarray(t, dtype=float)

    # a remaining flat step would still put an inf into np.gradient
    dt = np.diff(t)
    if (dt <= 0).any():
        step = np.median(dt[dt > 0]) if (dt > 0).any() else 1.0
        t = np.maximum.accumulate(t + np.arange(len(t)) * step * 1e-6)
        dt = np.diff(t)

    med = np.median(dt) if dt.size else np.nan
    fs = 1.0 / med if np.isfinite(med) and med > 0 else (nominal_hz or SAMPLE_RATE_HZ)
    return t, float(fs), mode


# ============================================================================
# HEADER
# ============================================================================
def read_header(path):
    """Parse the leading '# key=value' lines the v5 firmware writes.

    Returns {} for pre-v5 files, which have no header at all.
    """
    meta = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            for k, v in re.findall(r"(\w+)=([^\s]+)", line):
                meta[k] = v
    return meta


def check_header(meta, expect_immersion=True, quiet=False):
    """Print firmware provenance and warn where it disagrees with this module.

    Returns (sample_hz, effective_lsb_mV).
    """
    if not meta:
        if not quiet:
            print("  no '#' header (pre-v5 firmware) - module constants used as-is")
        return SAMPLE_RATE_HZ, LSB_mV

    over = int(meta.get("par_oversample", 1))
    lsb  = float(meta.get("ads_lsb_mV", LSB_mV))
    fs   = float(meta.get("sample_rate_hz", SAMPLE_RATE_HZ))
    if not quiet:
        print(f"  firmware {meta.get('firmware','?')}  serial {meta.get('sq500_serial','?')}  "
              f"{meta.get('ads_gain','?')}  oversample x{over}  {fs:.1f} Hz")

    cal = float(meta.get("sq500_cal_factor", np.nan))
    if not quiet and np.isfinite(cal) and abs(cal - ED_SENSITIVITY) > 1e-6:
        print(f"  ** cal factor: header {cal} vs module ED_SENSITIVITY {ED_SENSITIVITY}")

    hdr_iecf = meta.get("apply_iecf_in_post")
    if (not quiet and expect_immersion and hdr_iecf is not None
            and abs(float(hdr_iecf) - IECF) > 1e-6):
        print(f"  ** IECF: header {hdr_iecf} vs module IECF {IECF} -> absolute Ed differs by "
              f"{100*(IECF/float(hdr_iecf)-1):+.1f}% (Kd unaffected)")

    return fs, lsb / max(1, over)


def quantization_step_umol(meta, sensitivity=ED_SENSITIVITY):
    """Effective PPFD resolution of one ADC step, accounting for oversampling."""
    _, eff_lsb = check_header(meta, quiet=True)
    return sensitivity * eff_lsb


# ============================================================================
# LOADERS
# ============================================================================
def load_profile(path, apply_immersion=True):
    """Read one in-water (E_dzPAR) file: QC, tilt, Ed, elapsed seconds."""
    meta = read_header(path)
    # stable sort, so samples sharing one timestamp keep their logged order
    df = (pd.read_csv(path, comment="#", parse_dates=["iso_time"])
            .sort_values("iso_time", kind="stable").reset_index(drop=True))
    n0 = len(df)

    keep = (df.pressure_mbar.between(*PRESSURE_OK_MBAR)
            & df.depth_m.between(*DEPTH_OK_M)
            & (df.water_temp_C > TEMP_MIN_C))
    df = df[keep].reset_index(drop=True)
    df.attrs["glitch_rows_dropped"] = n0 - len(df)
    df.attrs["header"] = meta

    mag = np.sqrt(df.accel_x_ms2**2 + df.accel_y_ms2**2 + df.accel_z_ms2**2)
    df["tilt_deg"] = np.degrees(np.arccos(np.clip(df[IMU_UP_AXIS] / mag, -1, 1)))

    df["umol_air"]    = df["par_mV"] * ED_SENSITIVITY          # pre-immersion scale
    df["Ed"]          = df["umol_air"] * (IECF if apply_immersion else 1.0)
    df["t_s"], df.attrs["fs"], df.attrs["time_mode"] = build_time_axis(df.iso_time)
    df["depth_raw_m"] = df["depth_m"]
    return df


def load_surface(path, es_source="mv"):
    """Read one surface (E_sPAR) file. Column names vary between firmware builds,
    so time and PAR columns are detected rather than assumed. In air: NO immersion.
    """
    meta = read_header(path)
    df = pd.read_csv(path, comment="#")

    tcol = None
    for c in (["iso_time", "datetime", "time", "timestamp"] + list(df.columns)):
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.5:
                tcol = c; df[c] = parsed; break
    if tcol is None:
        raise ValueError(f"no datetime column found (columns: {list(df.columns)})")

    mv_names   = ["par_mV", "par_mv", "parmV", "millivolts", "mV", "mv"]
    ppfd_names = ["par_uMol_m2_s", "ppfd_umol_m2_s", "ppfd", "par_uMol", "umol_m2_s"]
    mv_col   = next((c for c in mv_names   if c in df.columns), None)
    ppfd_col = next((c for c in ppfd_names if c in df.columns), None)

    df = (df.rename(columns={tcol: "iso_time"})
            .dropna(subset=["iso_time"]).sort_values("iso_time", kind="stable")
            .reset_index(drop=True))

    if es_source == "ppfd":
        if ppfd_col is None:
            raise ValueError(f"es_source='ppfd' but no PPFD column ({list(df.columns)})")
        df["Es"] = df[ppfd_col].astype(float); par_used = ppfd_col
    else:
        if mv_col is None:
            raise ValueError(f"es_source='mv' but no millivolt column ({list(df.columns)})")
        df["par_mV"] = df[mv_col].astype(float)
        df["Es"] = df["par_mV"] * ES_SENSITIVITY; par_used = mv_col

    n0 = len(df)
    # v5: keep negatives. A capped/night sensor legitimately reads slightly below
    # zero and dropping those biases the dark offset high. Drop NaN only.
    df = df[df["Es"].notna()].reset_index(drop=True)
    df["t_s"], df.attrs["fs"], df.attrs["time_mode"] = build_time_axis(df.iso_time)

    df.attrs["bad_rows_dropped"]  = n0 - len(df)
    df.attrs["time_col_detected"] = tcol
    df.attrs["par_col_detected"]  = f"{par_used} (es_source={es_source})"
    df.attrs["header"]            = meta
    return df


def estimate_fs(df, meta=None):
    """Sample rate: header if present, else measured from the rebuilt time axis.

    iso_time.diff() is not used first because second-resolution firmware gives a
    median step of exactly 0 at any rate above 1 Hz.
    """
    if meta and "sample_rate_hz" in meta:
        return float(meta["sample_rate_hz"])
    if "fs" in df.attrs:
        return float(df.attrs["fs"])
    dt = df.iso_time.diff().dt.total_seconds().median()
    return 1.0 / dt if dt and dt > 0 else SAMPLE_RATE_HZ


# ============================================================================
# DARK SEGMENTS
# ============================================================================
def find_dark_runs(df, fs, col="umol_air", max_umol=DARK_MAX_UMOL,
                   min_dur_s=DARK_MIN_DURATION_S, trim_s=DARK_EDGE_TRIM_S,
                   bridge_s=DARK_BRIDGE_GAP_S):
    """All sustained dark runs, longest first. Returns a list of (i0, i1) indices.

    v5 returns EVERY qualifying run, not just the longest: a deployment bracketed
    by dark at both ends gives two independent estimates of the dark offset, and
    the pair is what makes the barometric baseline correction possible.
    """
    dark = (df[col].abs() < max_umol).to_numpy()
    n = len(df)
    runs, i = [], 0
    while i < n:
        if not dark[i]:
            i += 1; continue
        j = i
        while j + 1 < n and dark[j + 1]:
            j += 1
        runs.append([i, j]); i = j + 1
    if not runs:
        return []

    bridge = int(bridge_s * fs)
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= bridge:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    trim, out = int(trim_s * fs), []
    for a, b in merged:
        if (b - a) < min_dur_s * fs:
            continue
        a2, b2 = a + trim, b - trim
        if b2 - a2 < min_dur_s * fs:
            a2, b2 = a, b
        out.append((int(a2), int(b2)))
    return sorted(out, key=lambda r: r[1] - r[0], reverse=True)


def dark_table(df, fs, col="umol_air"):
    """Per-run mean / SD / distinct ADC codes for every dark segment.

    n_adc_codes is the diagnostic worth watching: a floor sitting on 3-5 codes is
    quantization-limited and its SD is an upper bound only. Many codes means the
    floor is genuinely analog.
    """
    rows = []
    for a, b in sorted(find_dark_runs(df, fs, col=col)):
        seg = df.iloc[a:b + 1]
        x = seg[col]
        codes = (seg["par_adc_counts"].nunique() if "par_adc_counts" in seg
                 else seg["par_mV"].nunique())
        rows.append(dict(i0=a, i1=b, t0_s=seg.t_s.iloc[0], t1_s=seg.t_s.iloc[-1],
                         dur_min=len(seg) / fs / 60, mean=x.mean(), sd=x.std(ddof=1),
                         depth_med=(seg.depth_raw_m.median() if "depth_raw_m" in seg else np.nan),
                         n_adc_codes=int(codes)))
    return pd.DataFrame(rows)


def resolve_min_ed(dark_tbl, n_sigma=3.0, apply_immersion=True, fallback=5.0):
    """Detection floor from the file's own dark runs: dark mean + n_sigma * SD.

    Replaces the hand-set MIN_ED. With MIN_ED = 0 (or a floor set below the real
    noise), samples one quantization step above zero drop ln(Ed) ~ -3 into the
    regression; with MIN_ED fixed too high, clear water gets truncated.
    """
    if dark_tbl is None or dark_tbl.empty:
        return fallback, False
    thr = float((dark_tbl["mean"] + n_sigma * dark_tbl["sd"]).max())
    return thr * (IECF if apply_immersion else 1.0), True


# ============================================================================
# DEPTH BASELINE
# ============================================================================
def baseline_model(profile, dark_tbl):
    """Out-of-water depth offset as a function of elapsed time.

    The Bar30 reads ABSOLUTE pressure, so the dry zero walks with barometric
    drift and sits far off zero at altitude (cf. the Lake Tahoe negative depths).
    During a capped dark run the instrument is on deck in air, so its median
    depth IS the offset. Two dark runs -> linear interpolation in time; one ->
    held constant; none -> shallowest 5% of samples.

    Returns a callable: t_s -> offset in metres.
    """
    if dark_tbl is None or dark_tbl.empty or dark_tbl.depth_med.isna().all():
        quiet = profile[profile.depth_raw_m < profile.depth_raw_m.quantile(0.05)]
        off = float(quiet.depth_raw_m.median()) if len(quiet) else 0.0
        return lambda t: np.full_like(np.asarray(t, float), off), "shallowest 5% of samples"

    d  = dark_tbl.dropna(subset=["depth_med"]).sort_values("t0_s")
    tc = ((d.t0_s + d.t1_s) / 2).to_numpy()
    of = d.depth_med.to_numpy()
    if len(tc) == 1:
        return (lambda t: np.full_like(np.asarray(t, float), float(of[0])),
                "single dark run, held constant")
    return (lambda t: np.interp(np.asarray(t, float), tc, of),
            f"{len(tc)} dark runs, interpolated in time")


def cast_baseline(profile, i0, i1, base_fn):
    """Offset to subtract from one cast, evaluated at its midpoint."""
    t_mid = 0.5 * (profile.t_s.iloc[i0] + profile.t_s.iloc[i1])
    return float(np.atleast_1d(base_fn(t_mid))[0])


# ============================================================================
# CAST DETECTION
# ============================================================================
def find_casts(profile, fs, direction="down", vel_thr=0.02, merge_gap_s=15.0,
               min_span_m=2.0, min_dur_s=15.0):
    """Segment on sustained vertical motion of a median-smoothed depth trace.

    Returns a list of (i0, i1) index pairs.

    Replaces the rolling depth-std 'profiling' flag. On a slow cast (~0.04 m/s)
    a std threshold discards the near-surface samples - the high-Ed end that
    anchors the Kd fit - and splitting on every direction change then fragments
    what remains at each bob and pause. Requiring sustained motion, then a
    minimum span AND duration, keeps whole casts and rejects surface bobbing,
    bottom holds and parked/capped periods.
    """
    n = max(3, int(2 * fs))
    d = profile["depth_m"].rolling(n, center=True, min_periods=3).median()
    t = profile["t_s"].to_numpy()
    v = (pd.Series(np.gradient(d.to_numpy(), t))
           .rolling(max(3, int(3 * fs)), center=True, min_periods=3).median())

    sign = 1 if direction == "down" else -1
    idx = np.flatnonzero(((sign * v) > vel_thr).to_numpy())
    if idx.size == 0:
        return []

    brk    = np.flatnonzero(np.diff(t[idx]) > merge_gap_s)
    starts = np.r_[idx[0], idx[brk + 1]]
    ends   = np.r_[idx[brk], idx[-1]]

    casts = []
    for s, e in zip(starts, ends):
        span = d.iloc[s:e + 1].max() - d.iloc[s:e + 1].min()
        if span >= min_span_m and (t[e] - t[s]) >= min_dur_s:
            casts.append((int(s), int(e)))
    return casts


def manual_casts(profile, start, end):
    m = (profile.iso_time >= pd.Timestamp(start)) & (profile.iso_time <= pd.Timestamp(end))
    if not m.any():
        return []
    ii = np.flatnonzero(m.to_numpy())
    return [(int(ii[0]), int(ii[-1]))]


# ============================================================================
# Kd
# ============================================================================
def fit_kd_per_cast(profile, casts, min_ed, base_fn=None, value_col="Ed",
                    min_pts=30, min_span_m=2.0):
    """One regression of ln(Ed) on depth per cast. All samples, no binning.

    Each cast is ONE replicate: the honest uncertainty is the spread ACROSS
    casts. Pooling every sample into a single regression gives a nominal SE
    roughly 10x too narrow because the samples are autocorrelated.
    """
    rows, fits = [], {}
    for i, (a, b) in enumerate(casts, 1):
        seg = profile.iloc[a:b + 1].copy()
        base = cast_baseline(profile, a, b, base_fn) if base_fn is not None else 0.0
        seg["depth_m"] = seg["depth_raw_m"] - base

        g = seg[(seg[value_col] > min_ed) & (seg.depth_m > 0)].copy()
        if len(g) < min_pts or (g.depth_m.max() - g.depth_m.min()) < min_span_m:
            print(f"  cast {i}: skipped (n={len(g)}, "
                  f"span={g.depth_m.max()-g.depth_m.min() if len(g) else 0:.2f} m)")
            continue

        g["lnE"] = np.log(g[value_col])
        f = stats.linregress(g.depth_m, g.lnE)
        lab = len(rows) + 1
        fits[lab] = dict(Kd=-f.slope, r2=f.rvalue**2, slope=f.slope,
                         intercept=f.intercept, slope_se=f.stderr, samples=g)
        rows.append(dict(cast=lab, t_start=seg.iso_time.min(), baseline_m=base,
                         z0=g.depth_m.min(), z1=g.depth_m.max(), zmax=seg.depth_m.max(),
                         n_pts=len(g), tilt_med=seg.tilt_deg.median(),
                         Kd=-f.slope, r2=f.rvalue**2, slope_se=f.stderr))
    return pd.DataFrame(rows), fits


def kd_ensemble_stats(tbl):
    """Honest Kd: each cast = one replicate, uncertainty = spread ACROSS casts."""
    if tbl is None or len(tbl) == 0:
        return {}
    k = tbl["Kd"].to_numpy(); n = len(k)
    mean = k.mean()
    sd   = k.std(ddof=1) if n > 1 else np.nan
    se   = sd / np.sqrt(n) if n > 1 else np.nan
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    return dict(n=n, mean=mean, sd=sd, se=se,
                ci_lo=mean - tcrit * se if n > 1 else np.nan,
                ci_hi=mean + tcrit * se if n > 1 else np.nan,
                median=float(np.median(k)),
                rng=float(k.max() - k.min()) if n > 1 else np.nan,
                cv=100 * sd / mean if n > 1 else np.nan)


# ============================================================================
# OUTPUT PATHS
# ============================================================================
def fig_base():
    return os.environ.get(
        "PARCAST_FIG_BASE",
        "/Users/loriberberian/Desktop/PARcast/data_processing/figures")


def make_saver(outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    import matplotlib.pyplot as plt

    def savefig(fig, shortname):
        path = os.path.join(outdir, f"{shortname}_{stem}.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print("saved", path)
    return savefig


# ============================================================================
# DAY-LEVEL WORKFLOW  (v6)
# ----------------------------------------------------------------------------
# Everything below operates on a whole field day rather than one file. Added
# for the v6 notebooks; nothing above this line was changed, so the v5
# notebooks and the parcast_*_figures.py scripts are unaffected.
#
# pvlib is imported lazily inside the solar functions, so this module still
# imports cleanly without it.
# ============================================================================

N_WATER    = 1.34                     # refractive index of seawater over PAR
# Percent-PAR reference depths. Keep these INSIDE the shallowest profile of
# the day, or the value is extrapolated from the fitted Kd rather than
# measured. Deepest samples on 11 Aug 2026 were 4.51-6.85 m, so 3 m is the
# deepest defensible reference. compute_metrics() masks anything beyond a
# station's own z_max regardless.
REF_DEPTHS = (1.0, 2.0, 3.0)

# Cast-segmentation defaults for v6. MERGE_GAP_S was 15 s, which bridged
# genuinely separate drops into one; MIN_DUR_S was 15 s, which rejected
# descents faster than ~0.28 m/s. Both are relaxed here.
MERGE_GAP_S_DEFAULT = 8.0
MIN_DUR_S_DEFAULT   = 10.0

# Publication figure style, matched to cops_kd_figures so the PARCAST and C-OPS
# panels on the poster read as one set: bold titles, axis labels and tick
# labels, no grid, left and bottom spines only, 300 dpi.
INK = "#1A1A1A"

PUB_RC = {
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 2.2, "axes.edgecolor": INK,

    "font.family": "DejaVu Sans",
    "font.size": 20, "font.weight": "bold",
    "text.color": INK, "axes.labelcolor": INK,
    "axes.titlesize": 24, "axes.titleweight": "bold", "axes.titlepad": 16,
    "axes.labelsize": 26, "axes.labelweight": "bold", "axes.labelpad": 12,
    "figure.titlesize": 28, "figure.titleweight": "bold",

    # Tick labels: the numbers on the axes
    "xtick.labelsize": 22, "ytick.labelsize": 22,
    "xtick.color": INK, "ytick.color": INK,

    # Tick marks themselves, scaled up to match the numerals
    "xtick.major.size": 10, "ytick.major.size": 10,
    "xtick.major.width": 2.2, "ytick.major.width": 2.2,
    "xtick.minor.size": 5, "ytick.minor.size": 5,
    "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
    "xtick.major.pad": 8, "ytick.major.pad": 8,
    "xtick.direction": "out", "ytick.direction": "out",

    "legend.fontsize": 18, "legend.frameon": False,
    "lines.linewidth": 2.6, "lines.markersize": 12,

    # Math text in the same bold sans serif as the labels around it. Letters in
    # math mode are italic (K, E, z); \mathrm{} and digits are roman.
    "mathtext.fontset": "custom",
    "mathtext.it": "DejaVu Sans:italic:bold",
    "mathtext.rm": "DejaVu Sans:bold",
    "mathtext.bf": "DejaVu Sans:bold",
    "mathtext.default": "it",
}

# Symbol strings for figure text. K, E and z are math italic; subscripts, PAR,
# operators and units are roman. Every figure builds its labels from these.
KD         = r"$K_\mathrm{d}$"
KD_PAR     = r"$K_\mathrm{d}(\mathrm{PAR})$"
ED         = r"$E_\mathrm{d}$"
ED_PAR     = r"$E_\mathrm{d}(\mathrm{PAR})$"
ED_Z_PAR   = r"$E_\mathrm{d}(z,\mathrm{PAR})$"
ES         = r"$E_\mathrm{s}$"
ES_PAR     = r"$E_\mathrm{s}(\mathrm{PAR})$"
LN_ED      = r"$\ln E_\mathrm{d}$"
THETA_W    = r"$\theta_\mathrm{w}$"
COS_W      = r"$\cos\theta_\mathrm{w}$"
PATH_FAC   = r"$1/\cos\theta_\mathrm{w}$"
KD_COS_W   = r"$K_\mathrm{d}\cos\theta_\mathrm{w}$"
R2         = r"$r^2$"
PER_M      = r"(m$^{-1}$)"
PPFD_UNITS = "(\u00b5mol m$^{-2}$ s$^{-1}$)"
DEPTH      = "Depth (m)"

# Legends are set in LEGEND_WEIGHT, as in the C-OPS figures. A legend holding a
# math symbol is set bold, so its words match the weight of the symbol.
LEGEND_WEIGHT = "normal"


# ============================================================================
# DEPLOYMENT SITES
# ----------------------------------------------------------------------------
# Coordinates and time zone for each field site, defined once so the notebooks
# and scripts cannot disagree. Solar zenith angle changes by about 0.1 deg per
# 0.1 deg of position, so a site centroid is accurate enough for the geometry;
# the time zone is not forgiving in the same way, since an hour of clock error
# moves the sun by roughly 15 deg.
#
# lat and lon are decimal degrees, north and east positive.
# tz is an IANA name. "America/Texas" is not one of them and raises at runtime.
# clock is what the logger's RTC was set to on that deployment, and is what
# clock_shift_h() converts from.
SITES = {
    "galveston": dict(
        name="Galveston, Gulf of Mexico", lat=29.31, lon=-94.77,
        tz="America/Chicago", clock="local",
        note="NASA SARP cruise, 8 June 2026. APPROXIMATE, replace with the "
             "cruise position from the field notes."),
    "laguna": dict(
        name="Laguna Beach, California", lat=33.5427, lon=-117.7854,
        tz="America/Los_Angeles", clock="local",
        note="Kelp forest, 4-5 June 2026. APPROXIMATE town centroid, replace "
             "with the dive site position."),
    "tahoe": dict(
        name="North Lake Tahoe, Nevada", lat=39.1988, lon=-119.9300,
        tz="America/Los_Angeles", clock="local",
        note="14 July 2026, Sand Harbor. APPROXIMATE. Surface pressure at "
             "1,897 m is about 805 mbar, so PRESSURE_OK_MBAR has to be widened "
             "for this site."),
    "sbc": dict(
        name="Santa Barbara Channel", lat=34.2114, lon=-119.9371,
        tz="America/Los_Angeles", clock="utc",
        note="Plumes and Blooms pb353, 5 August 2026. Station position; set "
             "per station if you want the geometry station by station."),
    "malibu": dict(
        name="Malibu, California", lat=34.0375, lon=-118.6775,
        tz="America/Los_Angeles", clock="utc",
        note="Surfrider Beach, 11 August 2026."),
}


def site(key):
    """One entry of SITES, by key. Raises with the valid keys rather than a KeyError."""
    try:
        return dict(SITES[key])
    except KeyError:
        raise KeyError(f"unknown site {key!r}. Known sites: {', '.join(SITES)}") from None


def clock_shift_h(tz, when, clock="utc"):
    """Hours to add to a logged timestamp to put it on local clock time.

    clock="local"  the RTC was already on site time, so the shift is 0.
    clock="utc"    the RTC was on UTC, so the shift is the site's offset from
                   UTC on that date. Daylight saving is why the date matters and
                   why this is not a constant: PDT is -7 and PST is -8, and the
                   Gulf coast is -5 in summer, not -7.

    when is any date inside the deployment. Pass the file's first timestamp.
    """
    if clock == "local":
        return 0.0
    if clock != "utc":
        raise ValueError(f"clock must be 'utc' or 'local', got {clock!r}")
    ts = pd.Timestamp(when)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts.tz_convert(tz).utcoffset().total_seconds() / 3600.0


def first_timestamp(path, col=None):
    """First timestamp in a logged file, for picking the UTC offset on that date."""
    df = pd.read_csv(path, comment="#", nrows=1)
    if col is None:
        col = next((c for c in ("iso_time", "datetime", "time", "timestamp")
                    if c in df.columns), df.columns[0])
    return pd.to_datetime(df[col].iloc[0])


def site_table():
    """SITES as a DataFrame, for printing in a notebook."""
    return pd.DataFrame(SITES).T.rename_axis("key").reset_index()


# ---------------------------------------------------------------- solar -----
def solar_geometry(times, lat, lon, tz, n_water=N_WATER):
    """Solar zenith, azimuth, in-water refracted angle and path factor.

    Naive timestamps are localised to `tz` explicitly. Handing local time to a
    solar routine that expects UTC returns a plausible-looking answer that is
    silently hours wrong.

    Uses the Solar Position Algorithm of Reda & Andreas (2004) via pvlib,
    stated uncertainty +/- 0.0003 deg.

    path_factor = 1/cos(theta_w) is the extra photon path per unit depth, and
    is the geometric predictor Kd scales with (Kirk 1984) - not the in-air
    angle, which refraction compresses.
    """
    import pvlib
    t = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(times)))
    if t.tz is None:
        t = t.tz_localize(tz)
    sp  = pvlib.solarposition.spa_python(t, lat, lon, altitude=0)
    sza = sp["apparent_zenith"].to_numpy()
    tw  = np.arcsin(np.sin(np.radians(np.clip(sza, 0, 89.99))) / n_water)
    return pd.DataFrame({
        "sza":         sza,
        "elevation":   sp["apparent_elevation"].to_numpy(),
        "azimuth":     sp["azimuth"].to_numpy(),
        "theta_w":     np.degrees(tw),
        "path_factor": 1.0 / np.cos(tw),
    })


def solar_noon(date, lat, lon, tz):
    """Solar transit for a calendar day, returned tz-naive local."""
    import pvlib
    loc = pvlib.location.Location(lat, lon, tz, 0, "site")
    day = pd.DatetimeIndex([pd.Timestamp(date).normalize()]).tz_localize(tz)
    r = loc.get_sun_rise_set_transit(day, method="spa").iloc[0]
    return {k: r[k].tz_localize(None) for k in ("sunrise", "transit", "sunset")}


def fresnel_reflectance(sza_deg, n_water=N_WATER):
    """Unpolarised reflectance at the air-water interface, fraction 0-1.

    ~2% overhead, ~35% at 80 deg, 100% at grazing. This is the physical reason
    the direct beam - and with it wave focusing - disappears at low sun.
    """
    ti = np.radians(np.clip(np.asarray(sza_deg, float), 0, 89.999))
    tt = np.arcsin(np.sin(ti) / n_water)
    rs = ((np.cos(ti) - n_water*np.cos(tt)) / (np.cos(ti) + n_water*np.cos(tt)))**2
    rp = ((n_water*np.cos(ti) - np.cos(tt)) / (n_water*np.cos(ti) + np.cos(tt)))**2
    return (rs + rp) / 2.0


# ------------------------------------------------------------ one station ---
def process_station(path, lat, lon, tz, tz_shift_h=0.0, apply_immersion=True,
                    auto_min_ed=True, n_sigma_dark=3.0, min_ed_manual=5.0,
                    baseline_correct=True, direction="down", vel_thr=0.02,
                    merge_gap_s=MERGE_GAP_S_DEFAULT, min_span_m=2.0,
                    min_dur_s=MIN_DUR_S_DEFAULT, min_pts=30, quiet=True):
    """Run one in-water file through the full chain.

    tz_shift_h shifts iso_time on load. The E_dzPAR logger RTC was set to UTC
    while the surface logger used local time, so -7 (PDT) puts them on the same
    clock. unix_time is a true epoch and is left untouched.

    Returns a dict with the profile, dark table, regression cutoff, casts, the
    per-cast Kd table (with solar geometry attached) and the fits.
    """
    prof = load_profile(path, apply_immersion=apply_immersion)
    if tz_shift_h:
        prof["iso_time"] = prof["iso_time"] + pd.Timedelta(hours=tz_shift_h)

    fs, _ = check_header(prof.attrs["header"], expect_immersion=apply_immersion,
                         quiet=quiet)
    if not prof.attrs["header"]:
        fs = estimate_fs(prof)                  # pre-v5 file: rate from the data
    dark  = dark_table(prof, fs)
    min_ed, auto = (resolve_min_ed(dark, n_sigma_dark, apply_immersion,
                                   fallback=min_ed_manual)
                    if auto_min_ed else (min_ed_manual, False))
    base_fn, base_how = baseline_model(prof, dark)
    if not baseline_correct:
        base_fn, base_how = None, "disabled"

    casts = find_casts(prof, fs, direction=direction, vel_thr=vel_thr,
                       merge_gap_s=merge_gap_s, min_span_m=min_span_m,
                       min_dur_s=min_dur_s)

    kd, fits = fit_kd_per_cast(prof, casts, min_ed, base_fn=base_fn,
                               min_pts=min_pts, min_span_m=min_span_m)

    s = dict(path=path, stem=os.path.splitext(os.path.basename(path))[0],
             profile=prof, fs=fs, dark=dark, min_ed=min_ed, auto_min_ed=auto,
             base_fn=base_fn, base_how=base_how, casts=casts,
             kd_table=kd, kd_fits=fits, stats=kd_ensemble_stats(kd),
             t0=prof.iso_time.iloc[0], apply_immersion=apply_immersion,
             min_pts=min_pts, min_span_m=min_span_m)

    if len(kd):
        mids = [r["samples"].iso_time.min() +
                (r["samples"].iso_time.max() - r["samples"].iso_time.min())/2
                for r in fits.values()]
        geo = solar_geometry(pd.DatetimeIndex(mids), lat, lon, tz)
        kd = pd.concat([kd.reset_index(drop=True), geo.reset_index(drop=True)],
                       axis=1)
        kd["t_mid"]   = pd.DatetimeIndex(mids)
        kd["Kd_norm"] = kd.Kd / kd.path_factor
        # %PAR only where the reference depth was actually sampled; beyond
        # z1 the value would be an extrapolation of the fitted exponential.
        for z in REF_DEPTHS:
            kd[f"pct_{z:g}m"] = np.where(z <= kd.z1, 100*np.exp(-kd.Kd*z), np.nan)

        # Light level reached at the DEEPEST measured sample. Unlike z_1% or
        # z_10%, this is bounded by the data: on 11 Aug 2026 z_1% worked out at
        # 14-26 m from profiles of 4.5-6.9 m, a 2-6x extrapolation of an
        # exponential and wholly unconstrained.
        kd["pct_at_zmax"]   = 100*np.exp(-kd.Kd*kd.z1)
        kd["optical_depth"] = kd.Kd * kd.z1
        s["kd_table"] = kd

        t_mid = kd.t_mid.min() + (kd.t_mid.max() - kd.t_mid.min())/2
        g = solar_geometry(t_mid, lat, lon, tz).iloc[0]
        s.update(t_mid=t_mid, sza=g.sza, azimuth=g.azimuth,
                 theta_w=g.theta_w, path_factor=g.path_factor)
    else:
        s.update(t_mid=prof.iso_time.iloc[len(prof)//2], sza=np.nan,
                 azimuth=np.nan, theta_w=np.nan, path_factor=np.nan)

    s["in_cast"] = np.zeros(len(prof), bool)
    for a, b in casts:
        s["in_cast"][a:b+1] = True
    return s


def load_day(ed_dir, lat, lon, tz, local_date=None, file_filter="",
             pattern="E_dzPAR", verbose=True, require_casts=True, **kw):
    """Process every in-water file for one field day.

    Files are grouped by their CORRECTED local date, not by filename. The
    logger names a file from the UTC date, so casts run after 17:00 PDT are
    written as the NEXT day - e.g. 20260812_0007 and _0008 belong to the
    11 August field day once shifted by tz_shift_h.

    require_casts=True drops files with no qualifying down-cast - a
    dark-only file, or a deployment where detection found nothing. Such a
    station has no Kd and no solar geometry (SZA is NaN), so it would render
    as an empty panel in every figure. Set False to keep them for inspection.

    verbose=True prints one line per file showing the filename date, the
    corrected local date, and whether it was kept - so nothing is dropped
    silently.
    """
    files = sorted(glob.glob(os.path.join(ed_dir, "*.CSV")) +
                   glob.glob(os.path.join(ed_dir, "*.csv")))
    files = [f for f in files if pattern in os.path.basename(f)]
    if file_filter:
        files = [f for f in files if file_filter in os.path.basename(f)]

    if verbose:
        print(f"{len(files)} file(s) matching '{pattern}'"
              + (f" and '{file_filter}'" if file_filter else "")
              + f" in {ed_dir}")
        if local_date:
            print(f"keeping local date {local_date}\n")
        print(f"{'file':<28} {'logged':<16} {'local':<16} {'casts':>6}  status")
        print("-" * 82)

    out, skipped = [], []
    for f in files:
        name = os.path.basename(f)
        try:
            s = process_station(f, lat, lon, tz, **kw)
        except Exception as e:
            skipped.append((name, f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"{name:<28} {'':<16} {'':<16} {'':>6}  FAILED "
                      f"{type(e).__name__}")
            continue

        shift = kw.get("tz_shift_h", 0.0)
        logged = s["t0"] - pd.Timedelta(hours=shift)      # as the logger wrote it
        keep = (not local_date) or s["t0"].strftime("%Y%m%d") == str(local_date)

        no_casts = len(s["kd_table"]) == 0
        if keep and require_casts and no_casts:
            keep = False
            reason = "no qualifying casts (dark-only file?)"
        else:
            reason = f"local date {s['t0']:%Y-%m-%d}"

        if verbose:
            note = "kept" if keep else f"skipped ({reason})"
            if keep and logged.strftime("%Y%m%d") != s["t0"].strftime("%Y%m%d"):
                note += "  <- crossed midnight UTC"
            print(f"{name:<28} {logged:%Y-%m-%d %H:%M}  {s['t0']:%Y-%m-%d %H:%M}  "
                  f"{len(s['casts']):6d}  {note}")

        if keep:
            out.append(s)
        else:
            skipped.append((name, reason))

    out = sorted(out, key=lambda s: s["t0"])
    if verbose:
        print(f"\n{len(out)} station(s) kept, {len(skipped)} skipped")
        if not out:
            print("** nothing kept. Check LOCAL_DATE against the 'local' column "
                  "above, or set LOCAL_DATE='' to keep everything.")
    return out


def station_table(stations, noon=None, r2_min=0.98, cv_max=10.0):
    """One row per station: mean Kd, spread, geometry, pass/fail."""
    rows = []
    for s in stations:
        st, kd = s["stats"], s["kd_table"]
        r = dict(station=f"{s['t0']:%H:%M}", stem=s["stem"], t_mid=s["t_mid"],
                 sza=s["sza"], theta_w=s["theta_w"], path_factor=s["path_factor"],
                 n=st.get("n", 0), Kd=st.get("mean", np.nan),
                 Kd_sd=st.get("sd", np.nan), cv=st.get("cv", np.nan),
                 r2_min=kd.r2.min() if len(kd) else np.nan,
                 z_max=kd.z1.max() if len(kd) else np.nan,
                 optical_depth=kd.optical_depth.mean() if len(kd) else np.nan,
                 pct_at_zmax=kd.pct_at_zmax.mean() if len(kd) else np.nan,
                 tilt=kd.tilt_med.median() if len(kd) else np.nan,
                 min_ed=s["min_ed"], n_dark=len(s["dark"]))
        if len(kd):
            r["Kd_norm"] = kd.Kd_norm.mean()
            for z in REF_DEPTHS:
                r[f"pct_{z:g}m"] = kd[f"pct_{z:g}m"].mean()
        if noon is not None:
            r["limb"] = "AM" if s["t_mid"] < noon else "PM"
            r["hrs_from_noon"] = (s["t_mid"] - noon).total_seconds()/3600
        r["ok"] = bool(np.isfinite(r["r2_min"]) and r["r2_min"] >= r2_min
                       and np.isfinite(r["cv"]) and r["cv"] <= cv_max)
        rows.append(r)
    out = pd.DataFrame(rows)
    return out.dropna(subset=["sza"]).reset_index(drop=True) if len(out) else out


def depth_groups(s, gap_min=2.0):
    """Split one station's casts into groups separated by long pauses.

    A station occupied at several depths in succession appears as one file;
    grouping by time gap recovers the separate occupations.
    """
    cs = s["casts"]
    if not cs:
        return []
    tm = s["profile"].t_s.values / 60
    groups, cur = [], [cs[0]]
    for prev, nxt in zip(cs[:-1], cs[1:]):
        if tm[nxt[0]] - tm[prev[1]] > gap_min:
            groups.append(cur); cur = [nxt]
        else:
            cur.append(nxt)
    groups.append(cur)
    return groups


def fluctuation_cv(s, bins=None, value_col="Ed"):
    """CV of E_d within depth bins, pooled across a station's casts.

    Zaneveld et al. (2001): near-surface irradiance fluctuation under waves is
    the real light field, not instrument noise, and it decays with depth as
    beams from successive waves cross and average.
    """
    if not s["kd_fits"]:
        return pd.Series(dtype=float)
    if bins is None:
        bins = np.arange(0, 8.0, 0.5)
    seg = pd.concat([r["samples"] for r in s["kd_fits"].values()])
    g = seg.groupby(pd.cut(seg.depth_m, bins), observed=True)[value_col]
    cv = (100 * g.std() / g.mean()).dropna()
    cv.index = [iv.mid for iv in cv.index]
    return cv


def cloud_index(surf, lat, lon, tz, model="ineichen", pctile=98,
                fit_window=None, clear_thr=0.85):
    """Measured surface PAR / modelled clear-sky curve.

    The clear-sky model returns BROADBAND irradiance, not PAR. Because this is
    a ratio the units cancel and only the shape of the curve is used - do not
    read `clear_ref` as a PAR prediction.

    fit_window : ("13:00", "16:00") to fit the scale over a period known to be
        clear. This matters: broken cloud produces edge enhancement, where
        light scattered off cloud edges briefly pushes measured irradiance
        ABOVE clear-sky. If those spikes land in the percentile, the reference
        is inflated and the rest of the day is scored as cloudier than it was.
    """
    import pvlib
    loc = pvlib.location.Location(lat, lon, tz, 0, "site")
    idx = pd.DatetimeIndex(surf["iso_time"])
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    cs = loc.get_clearsky(idx, model=model)["ghi"].to_numpy()
    es = surf["Es"].to_numpy(float)

    ok = cs > 50
    mask = ok
    if fit_window:
        d0 = surf["iso_time"].iloc[0].strftime("%Y-%m-%d")
        t0 = pd.Timestamp(f"{d0} {fit_window[0]}")
        t1 = pd.Timestamp(f"{d0} {fit_window[1]}")
        w = ok & (surf["iso_time"] >= t0).values & (surf["iso_time"] <= t1).values
        if w.sum() >= 100:
            mask = w

    scale     = np.nanpercentile(es[mask]/cs[mask], pctile) if mask.any() else 1.0
    scale_all = np.nanpercentile(es[ok]/cs[ok], pctile) if ok.any() else 1.0
    ref = cs * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(ref > 0, es/ref, np.nan)

    out = pd.DataFrame({"iso_time": surf["iso_time"].values, "Es": es,
                        "clear_ref": ref, "cloud_index": np.clip(ci, 0, 1.2)})
    out["sky_class"] = pd.cut(out.cloud_index, [-0.01, 0.35, 0.65, clear_thr, 2.0],
                              labels=["overcast", "heavy cloud", "broken", "clear"])
    out.attrs.update(scale=scale, scale_all=scale_all, model=model)
    return out


# ============================================================================
# DAY-LEVEL FIGURES (v6)
# ----------------------------------------------------------------------------
# Each returns a matplotlib Figure and does not save or close it, so the
# notebook stays in charge of both.
# ============================================================================
def _plt():
    import matplotlib.pyplot as plt
    return plt


def _finish(fig):
    """Apply the legend weight rule to every legend in the figure."""
    legends = [a.get_legend() for a in fig.axes] + list(fig.legends)
    for leg in legends:
        if leg is None:
            continue
        texts = list(leg.get_texts()) + [leg.get_title()]
        weight = "bold" if any("$" in t.get_text() for t in texts) else LEGEND_WEIGHT
        for t in texts:
            t.set_fontweight(weight)
    return fig


def _log_x_ticks(ax):
    """Plain log-axis tick labels spaced so the 22 pt numerals do not collide.

    Four labels per decade over a narrow range, 1 and 3 per decade up to two and
    a half decades, one per decade beyond that. Unlabelled minor ticks mark the
    rest. A clear-water profile can span well under one decade, where labelling
    only the decade marks leaves the axis with a single number on it.
    """
    from matplotlib import ticker as mticker
    lo, hi = ax.get_xlim()
    decades = np.log10(hi / lo) if lo > 0 and hi > lo else 3.0
    if decades <= 1.2:
        subs = (1., 2., 3., 5.)
    elif decades <= 2.5:
        subs = (1., 3.)
    else:
        subs = (1.,)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=subs, numticks=12))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10),
                                                  numticks=12))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def fig_timeline_grid(stations, tilt_ok_deg=5.0, pad_min=1.5, panel_w=6.2,
                      height=14.5):
    """One column per station: E_d, depth and tilt through the casting window.

    Cropped to the casts plus `pad_min` either side. Most of each record is
    dark time and transit; plotting it whole compresses the part you want to
    see into a sliver. Set pad_min=None to plot the full record.
    """
    plt = _plt()
    stations = [s for s in stations if s["casts"]]     # skip empty panels
    if not stations:
        raise ValueError("no station has any detected cast")
    n = len(stations)
    fig, ax = plt.subplots(3, n, figsize=(panel_w*n, height), sharey="row",
                           gridspec_kw={"hspace": 0.30, "wspace": 0.20,
                                        "height_ratios": [1.25, 1, 0.75]})
    ax = np.atleast_2d(ax)
    if n == 1:
        ax = ax.reshape(3, 1)

    for j, s in enumerate(stations):
        p = s["profile"]; tm = p.t_s.values/60
        aE, aD, aT = ax[0, j], ax[1, j], ax[2, j]

        if pad_min is not None and s["casts"]:
            lo = tm[s["casts"][0][0]] - pad_min
            hi = tm[s["casts"][-1][1]] + pad_min
        else:
            lo, hi = tm[0], tm[-1]
        m = (tm >= lo) & (tm <= hi)
        t = tm[m] - lo

        aE.plot(t, p.Ed.values[m], color="#c0392b", lw=0.7)
        aE.axhline(s["min_ed"], color="k", ls=":", lw=1.2)
        aE.set_yscale("log")
        aD.plot(t, p.depth_raw_m.values[m], color="#2e6f95", lw=0.9)
        aT.plot(t, p.tilt_deg.values[m], color="#6c3483", lw=0.7)
        aT.axhline(tilt_ok_deg, color="red", ls=":", lw=1.2)

        for a, b in s["casts"]:
            for a_ in (aE, aD, aT):
                a_.axvspan(tm[a]-lo, tm[b]-lo, color="#2e6f95", alpha=0.15, lw=0)

        sza = s.get("sza", np.nan)
        aE.set_title(f"{s['t0']:%H:%M}   SZA {sza:.0f}°\n{len(s['casts'])} casts",
                     fontsize=22)
        aT.set_xlabel("Minutes")
        for a_ in (aE, aD, aT):
            a_.set_xlim(0, hi-lo); a_.locator_params(axis="x", nbins=3)
        for a_ in (aD, aT):                       # not aE: it is log-scaled
            a_.locator_params(axis="y", nbins=5)
        if j == 0:
            aE.set_ylabel(f"{ED_PAR}\n{PPFD_UNITS}")
            aD.set_ylabel("Depth (m)")
            aT.set_ylabel("Tilt (°)")

    ax[1, 0].invert_yaxis()
    fig.suptitle("Casting windows with detected down casts shaded", y=0.985)
    _finish(fig)
    return fig


def fig_profile_grid(stations, ncol=3, panel_w=8.0, panel_h=8.8, cbar=True,
                     cmap="turbo"):
    """Log-x depth profiles, one panel per station, colored by cast order.

    The color scale is normalised WITHIN each station, so it shows relative
    order (first to last cast) rather than an absolute cast number - stations
    have different numbers of casts. The shared colorbar is labelled
    accordingly; set cbar=False to omit it.
    """
    plt = _plt()
    from matplotlib import ticker as mticker
    import matplotlib as mpl
    stations = [s for s in stations if s["kd_fits"]]   # skip empty panels
    n = len(stations); ncol = min(ncol, n); nrow = int(np.ceil(n/ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(panel_w*ncol, panel_h*nrow),
                            squeeze=False, sharey=True)
    cm = plt.get_cmap(cmap)

    for k, s in enumerate(stations):
        a = axs[k//ncol][k%ncol]
        N = max(1, len(s["kd_fits"]))
        for i, r in enumerate(s["kd_fits"].values()):
            seg = r["samples"]
            a.scatter(seg.Ed, seg.depth_m, s=30, alpha=0.6, color=cm(i/max(1, N-1)))
        a.axvline(s["min_ed"], color=INK, ls=":", lw=2,
                  label="regression cutoff" if k == 0 else None)
        a.set_xscale("log")
        _log_x_ticks(a)
        kd = s["stats"].get("mean", np.nan)
        a.set_title(f"{s['t0']:%H:%M}   SZA {s.get('sza', np.nan):.0f}°\n"
                    f"{KD} = {kd:.3f} m$^{{-1}}$   $n$ = {N}", fontsize=22)
        a.set_xlabel(f"{ED_Z_PAR} {PPFD_UNITS}")
        if k % ncol == 0:
            a.set_ylabel(DEPTH)
        if k == 0:
            a.legend(loc="upper left")
    axs[0][0].invert_yaxis()
    empty = [(k//ncol, k%ncol) for k in range(n, nrow*ncol)]
    for r, c in empty:
        axs[r][c].axis("off")

    fig.suptitle("Depth profiles by station", y=1.00)
    fig.tight_layout()

    if cbar:
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=mpl.colors.Normalize(0, 1))
        if empty:
            # tuck it into the last unused slot, bottom right of the grid
            host = axs[empty[-1][0]][empty[-1][1]]
            box = host.get_position()
            cax = fig.add_axes([box.x0 + 0.06*box.width, box.y0 + 0.42*box.height,
                                0.68*box.width, 0.055*box.height])
        else:
            # full grid, no spare slot - hang it below the bottom-right panel,
            # clear of that panel's x-label
            host = axs[nrow-1][ncol-1]
            box = host.get_position()
            fig.canvas.draw()
            lab = host.xaxis.label.get_window_extent()
            y_lab = fig.transFigure.inverted().transform((0, lab.y0))[1]
            cax = fig.add_axes([box.x0 + 0.15*box.width, y_lab - 0.06,
                                0.70*box.width, 0.012])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["first", "last"])
        cb.set_label("Cast order within station", labelpad=10, fontsize=19)
        cb.ax.tick_params(labelsize=17)
        cb.outline.set_linewidth(1.5)
    _finish(fig)
    return fig


def cast_frames(s, value_col="Ed"):
    """Every sample of each fitted cast, flagged by whether it entered the fit.

    fit_kd_per_cast keeps only samples above the regression cutoff, which is right
    for the regression. The samples below it are real readings resolved to a few
    ADS1115 codes, so the figures draw them rather than drop them. The accept test
    and cast numbering mirror fit_kd_per_cast, so cast N is the same cast in both.
    """
    prof, min_ed = s["profile"], s["min_ed"]
    min_pts, min_span_m = s.get("min_pts", 30), s.get("min_span_m", 2.0)
    out = {}
    for a, b in s["casts"]:
        seg = prof.iloc[a:b + 1].copy()
        base = cast_baseline(prof, a, b, s["base_fn"]) if s["base_fn"] is not None else 0.0
        seg["depth_m"] = seg["depth_raw_m"] - base
        seg = seg[seg.depth_m > 0].copy()
        g = seg[seg[value_col] > min_ed]
        if len(g) < min_pts or (g.depth_m.max() - g.depth_m.min()) < min_span_m:
            continue
        seg["in_fit"] = seg[value_col] > min_ed
        out[len(out) + 1] = seg
    return out


def adc_step_ed(s):
    """One ADS1115 step in E_d units: sensitivity x effective LSB, times IECF in water."""
    step = quantization_step_umol(s["profile"].attrs.get("header", {}))
    return step * (IECF if s.get("apply_immersion", True) else 1.0)


def fig_depth_profile(s, tilt_ok_deg=5.0, title=None, figsize=(9.5, 11.0),
                      cmap="viridis"):
    """E_d against depth for every sample in the file, cast samples colored by tilt.

    Light grey  samples outside a detected down cast (parked, drifting, upcasts).
    Mid grey    cast samples at or above tilt_ok_deg.
    Colored     cast samples below tilt_ok_deg, on a 0 to tilt_ok_deg scale.

    Depth is baseline corrected with the station's dry zero model, so the surface
    sits at 0 m as it does in fig_logx_profile.
    """
    plt = _plt()
    p = s["profile"]
    base = s["base_fn"](p.t_s.to_numpy()) if s["base_fn"] is not None else 0.0
    z = p.depth_raw_m.to_numpy() - base
    cast = s["in_cast"]
    tilted = cast & (p.tilt_deg.to_numpy() >= tilt_ok_deg)
    level = cast & (p.tilt_deg.to_numpy() < tilt_ok_deg)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(p.Ed[~cast], z[~cast], s=14, color="#D9D9D9", lw=0, zorder=1)
    ax.scatter(p.Ed[tilted], z[tilted], s=20, color="#A6A6A6", lw=0, zorder=2)
    sc = ax.scatter(p.Ed[level], z[level], s=26, c=p.tilt_deg[level], cmap=cmap,
                    vmin=0, vmax=tilt_ok_deg, lw=0, zorder=3)
    ax.invert_yaxis()
    ax.set_xlabel(f"{ED_Z_PAR} {PPFD_UNITS}")
    ax.set_ylabel(DEPTH)
    # the date is in the title because two field days can start at the same clock time
    ax.set_title(title if title is not None
                 else f"PAR depth profile {s['t0']:%d %b %H:%M}")

    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.03)
    cb.set_label("Tilt (deg)")
    cb.outline.set_linewidth(2.0)
    cb.ax.tick_params(width=2.0, length=8)

    # proxy markers, so the colored class shows one representative color
    from matplotlib.lines import Line2D
    key = [("#D9D9D9", f"parked or drifting"),
           ("#A6A6A6", f"cast tilt above {tilt_ok_deg:g} deg"),
           (plt.get_cmap(cmap)(0.75), f"cast tilt below {tilt_ok_deg:g} deg")]
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=11, color=c, label=l)
                       for c, l in key],
              loc="lower right", handletextpad=0.3)
    fig.tight_layout()
    _finish(fig)
    return fig


def fig_logx_profile(s, show_step="auto", step_decades=1.0, show_excluded=True,
                     show_cutoff=False, title=None, figsize=(9.5, 11.0),
                     legend_loc="lower right"):
    """Log E_d against depth, one color per cast, every cast sample shown.

    Filled markers entered the K_d fit. Hollow markers are below the regression
    cutoff and were not fitted; they are drawn because what limits them is the
    ADS1115 step size, not the SQ-500. Samples at or below 0 cannot be drawn on a
    log axis; count them with cast_frames(s).

    show_step="auto" draws the ADS1115 step only when the dimmest sample is within
    step_decades of it, which is the case the line is there to make: a profile that
    ran into the resolution of the converter. In clear water at 16x oversampling the
    step sits two or three decades below the data, and drawing it there stretches
    the axis until the profile is a sliver at the right-hand edge. True and False
    force it on or off.
    """
    plt = _plt()
    cm = plt.get_cmap("tab10")
    frames = cast_frames(s)

    fig, ax = plt.subplots(figsize=figsize)
    for cid, seg in frames.items():
        c = cm((cid - 1) % 10)
        used = seg[seg.in_fit]
        ax.scatter(used.Ed, used.depth_m, s=26, color=c, alpha=0.65, lw=0,
                   label=f"cast {cid}")
        low = seg[(~seg.in_fit) & (seg.Ed > 0)]
        if show_excluded and len(low):
            ax.scatter(low.Ed, low.depth_m, s=26, facecolors="none", edgecolors=c,
                       linewidths=1.4, alpha=0.75)

    step = adc_step_ed(s)
    lows = [f.Ed[f.Ed > 0].min() for f in frames.values() if (f.Ed > 0).any()]
    e_min = min(lows) if lows else np.nan
    draw_step = show_step
    if show_step == "auto":
        draw_step = bool(np.isfinite(e_min) and step > 0
                         and e_min / step <= 10.0 ** step_decades)
    if draw_step:
        ax.axvline(step, color="#7F7F7F", ls="--", lw=2.5,
                   label=f"ADS1115 step {step:.2f}")
    if show_cutoff:
        ax.axvline(s["min_ed"], color=INK, ls=":", lw=2.5,
                   label=f"regression cutoff {s['min_ed']:.1f}")
    if show_excluded and any((~f.in_fit).any() for f in frames.values()):
        ax.scatter([], [], s=26, facecolors="none", edgecolors="#444444",
                   linewidths=1.4, label="below cutoff, not fitted")

    ax.set_xscale("log")
    _log_x_ticks(ax)          # plain numerals, spaced for the 22 pt tick font
    ax.invert_yaxis()
    ax.set_xlabel(f"{ED_Z_PAR} {PPFD_UNITS}")
    ax.set_ylabel(DEPTH)
    ax.set_title(title if title is not None
                 else f"PAR depth profile log scale {s['t0']:%d %b %H:%M}")
    if frames:
        # A profile runs from bright and shallow at the top right to dim and deep at
        # the bottom left, so the bottom right corner is the one that stays empty.
        # The top left only looks empty until the legend is wide enough to reach the
        # near-surface samples. White fill with no edge keeps the step line and the
        # deepest points from showing through the text.
        leg = ax.legend(loc=legend_loc, ncol=2 if len(frames) > 6 else 1,
                        markerscale=2.2, columnspacing=0.8, handletextpad=0.3,
                        frameon=True, framealpha=1.0, facecolor="white",
                        edgecolor="none")
        leg.set_zorder(5)
    fig.tight_layout()
    _finish(fig)
    return fig


def fig_fits_grid(stations, panel_w=6.0, height=12.5):
    """Per-cast ln(E_d) fits (top) and residuals (bottom).

    The fan-out between fit lines is the honest uncertainty - not the standard
    error of any single fit, which is roughly an order of magnitude too narrow
    because samples within a cast are autocorrelated.
    """
    plt = _plt()
    stations = [s for s in stations if s["kd_fits"]]   # skip empty panels
    n = len(stations)
    fig, axs = plt.subplots(2, n, figsize=(panel_w*n, height), squeeze=False,
                            gridspec_kw={"hspace": 0.40, "wspace": 0.22})
    for j, s in enumerate(stations):
        aL, aR = axs[0][j], axs[1][j]
        cm = plt.get_cmap("turbo"); N = max(1, len(s["kd_fits"]))
        for i, r in enumerate(s["kd_fits"].values()):
            seg = r["samples"]; col = cm(i/max(1, N-1))
            aL.scatter(seg.depth_m, seg.lnE, s=24, color=col, alpha=0.35)
            xs = np.linspace(seg.depth_m.min(), seg.depth_m.max(), 30)
            aL.plot(xs, r["intercept"] + r["slope"]*xs, color=col, lw=1.5)
            aR.scatter(seg.depth_m, seg.lnE - (r["intercept"] + r["slope"]*seg.depth_m),
                       s=24, color=col, alpha=0.35)
        aR.axhline(0, color="k", lw=0.8, ls=":")
        aL.set_title(f"{s['t0']:%H:%M}   SZA {s.get('sza', np.nan):.0f}°", fontsize=22)
        aR.set_xlabel(DEPTH)
        if j == 0:
            aL.set_ylabel(LN_ED); aR.set_ylabel(f"Residual {LN_ED}")
    fig.suptitle("Per cast fits above and residuals below", y=0.98)
    _finish(fig)
    return fig


def fig_kd_day(tbl, figsize=(21, 8.4)):
    """Kd through the day, and Kd against the in-water path factor.

    The dashed line is what geometry alone predicts, anchored at the station
    closest to solar noon. Points above it need a water-column explanation.
    """
    plt = _plt()
    import matplotlib.dates as mdates
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize)

    a1.errorbar(tbl.t_mid, tbl.Kd, yerr=tbl.Kd_sd, fmt="o-", ms=9, capsize=4,
                color="#2e6f95", lw=1.5)
    for r in tbl.itertuples():
        a1.annotate(f"{r.sza:.0f}°", (r.t_mid, r.Kd), fontsize=17, fontweight="bold",
                    xytext=(7, 9), textcoords="offset points")
    a1.set_xlabel("Time (local)"); a1.set_ylabel(f"{KD_PAR} {PER_M}")
    a1.set_title(f"{KD_PAR} through the day")
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if "limb" in tbl:
        for lb, col, mk in [("AM", "#2e6f95", "o"), ("PM", "#c0392b", "s")]:
            d = tbl[tbl.limb == lb]
            if len(d):
                a2.errorbar(d.path_factor, d.Kd, yerr=d.Kd_sd, fmt=mk, ms=10,
                            capsize=4, color=col, lw=0, elinewidth=1.5, label=lb)
    else:
        a2.errorbar(tbl.path_factor, tbl.Kd, yerr=tbl.Kd_sd, fmt="o", ms=10,
                    capsize=4, color="#2e6f95", lw=0, elinewidth=1.5)
    ref = tbl.loc[tbl.sza.idxmin()]
    xs = np.linspace(tbl.path_factor.min(), tbl.path_factor.max(), 30)
    a2.plot(xs, ref.Kd*xs/ref.path_factor, color="#888780", ls="--", lw=1.8,
            label="geometry alone")
    a2.set_xlabel(f"Path factor {PATH_FAC}")
    a2.locator_params(axis="x", nbins=5)
    a2.set_ylabel(f"{KD_PAR} {PER_M}")
    a2.set_title(f"{KD_PAR} vs in water path factor"); a2.legend(loc="upper left")
    fig.tight_layout()
    _finish(fig)
    return fig


def fig_operational_range(tbl, r2_min=0.98, cv_max=10.0, tilt_ok_deg=5.0,
                          figsize=(24, 7.8)):
    """Replicate CV and worst per-cast r2 against solar zenith angle."""
    plt = _plt()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=figsize)
    col = ["#2e6f95" if o else "#c0392b" for o in tbl.ok]

    a1.scatter(tbl.sza, tbl.cv, s=240, c=col, zorder=3)
    a1.axhline(cv_max, color="red", ls="--", lw=1.5, label=f"CV ≤ {cv_max:g}%")
    a1.set_xlabel("Solar zenith angle (deg)")
    a1.set_ylabel(f"CV of {KD} across casts (%)")
    a1.set_title("Repeatability"); a1.legend(loc="upper left")

    a2.scatter(tbl.sza, tbl.r2_min, s=240, c=col, zorder=3)
    a2.axhline(r2_min, color="red", ls="--", lw=1.5, label=f"{R2} ≥ {r2_min}")
    a2.set_xlabel("Solar zenith angle (deg)"); a2.set_ylabel(f"Worst per cast {R2}")
    a2.set_title("Fit quality"); a2.legend(loc="lower left")

    a3.scatter(tbl.tilt, tbl.Kd, s=240, color="#6c3483", zorder=3)
    a3.axvline(tilt_ok_deg, color="red", ls=":", lw=1.5, label=f"{tilt_ok_deg:.0f}°")
    a3.set_xlabel("Median cast tilt (deg)"); a3.set_ylabel(f"{KD_PAR} {PER_M}")
    a3.set_title(f"{KD_PAR} vs tilt"); a3.legend(loc="upper right")

    if tbl.ok.any():
        for a_ in (a1, a2):
            a_.axvspan(0, tbl[tbl.ok].sza.max(), color="green", alpha=0.07)
    fig.tight_layout()
    _finish(fig)
    return fig


def fig_focusing(stations, tbl, bins=None, figsize=(21, 10.5)):
    """Fluctuation intensity vs depth, and near-surface CV vs sun angle."""
    plt = _plt()
    fig, (b1, b2) = plt.subplots(1, 2, figsize=figsize)
    cm = plt.get_cmap("plasma")
    tbl = tbl.dropna(subset=["sza"])
    stations = [s for s in stations if s["kd_fits"]]
    szas = tbl.sza.values
    rng = max(1e-9, np.ptp(szas))
    rows = []
    for s, sz in zip(stations, szas):
        cv = fluctuation_cv(s, bins)
        if not len(cv):
            continue
        b1.plot(cv.values, cv.index, "o-", ms=10,
                color=cm((sz - np.nanmin(szas))/rng),
                label=f"{s['t0']:%H:%M}  {sz:.0f}°")
        shallow = cv[[i < 2 for i in cv.index]].mean()
        rows.append(dict(station=f"{s['t0']:%H:%M}", sza=round(sz, 1),
                         cv_upper2m=round(shallow, 1), cv_all=round(cv.mean(), 1)))
    b1.invert_yaxis()
    b1.set_xlabel(f"CV of {ED} within depth bin (%)"); b1.set_ylabel(DEPTH)
    b1.set_title("Fluctuation intensity vs depth")
    # Below the axes: the profiles fill the panel, so any in-axes placement
    # ends up with lines running through the legend.
    b1.legend(title="Station and SZA", loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=15,
              title_fontsize=16, columnspacing=1.4, handlelength=1.6)

    fl = pd.DataFrame(rows)
    b2.scatter(fl.sza, fl.cv_upper2m, s=260, color="#c0392b", zorder=3)
    for r in fl.itertuples():
        b2.annotate(r.station, (r.sza, r.cv_upper2m), fontsize=17, fontweight="bold",
                    xytext=(8, 8), textcoords="offset points")
    b2.set_xlabel("Solar zenith angle (deg)"); b2.set_ylabel("Mean CV, upper 2 m (%)")
    b2.set_title("Near surface fluctuation vs sun angle")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.34)      # room for the legend below b1
    _finish(fig)
    return fig, fl


def fig_multidepth(s, lat, lon, tz, gap_min=2.0, bins=None):
    """Panels for a station occupied at several depths, plus a CV comparison.

    Sun angle is effectively constant within a station, so a difference in Kd
    between groups is not geometric. Zaneveld et al. (2001) predict it: a
    profile truncated above the wave-focusing crossover is dominated by the
    focal region, so its fitted Kd is biased relative to one reaching the
    damped water below.
    """
    plt = _plt()
    from matplotlib import ticker as mticker
    gs = depth_groups(s, gap_min)
    if len(gs) < 2:
        return None, pd.DataFrame()

    cid = {c: i for i, c in enumerate(s["casts"], 1)}
    fig, axs = plt.subplots(1, len(gs)+1, figsize=(8.0*(len(gs)+1), 9.8),
                            gridspec_kw={"wspace": 0.40})
    cmg = plt.get_cmap("plasma")
    rows = []

    for gi, (g, ax) in enumerate(zip(gs, axs[:-1]), 1):
        ids = [cid[c] for c in g if cid[c] in s["kd_fits"]]
        if not ids:
            continue
        sub  = s["kd_table"][s["kd_table"].cast.isin(ids)]
        segs = pd.concat([s["kd_fits"][i]["samples"] for i in ids])
        tmid = segs.iso_time.min() + (segs.iso_time.max()-segs.iso_time.min())/2
        geo  = solar_geometry(tmid, lat, lon, tz).iloc[0]

        b  = segs.groupby(pd.cut(segs.depth_m,
                                 bins if bins is not None else np.arange(0, 8, .5)),
                          observed=True).Ed
        cv = (100*b.std()/b.mean()).dropna()
        cv.index = [iv.mid for iv in cv.index]

        cm = plt.get_cmap("turbo")
        for i, c in enumerate(ids):
            seg = s["kd_fits"][c]["samples"]
            ax.scatter(seg.Ed, seg.depth_m, s=34, alpha=0.6,
                       color=cm(i/max(1, len(ids)-1)))
        ax.set_xscale("log"); ax.invert_yaxis()
        # plain, widely spaced decade labels - the default log minor ticks
        # collide at this font size (3x10^2 running into 4x10^2)
        _log_x_ticks(ax)
        ax.set_title(f"Group {gi}   {segs.iso_time.min():%H:%M} to "
                     f"{segs.iso_time.max():%H:%M}\nSZA {geo.sza:.1f}°   "
                     f"max {sub.z1.max():.1f} m\n"
                     f"{KD} = {sub.Kd.mean():.3f} m$^{{-1}}$", fontsize=20)
        ax.set_xlabel(f"{ED_Z_PAR} {PPFD_UNITS}")
        if gi == 1:
            ax.set_ylabel(DEPTH)

        axs[-1].plot(cv.values, cv.index, "o-", ms=11,
                     color=cmg((gi-1)/max(1, len(gs)-1)),
                     label=f"Group {gi} ({sub.z1.max():.1f} m)")

        rows.append(dict(station=f"{s['t0']:%H:%M}", group=gi,
                         t_start=segs.iso_time.min(), sza=round(geo.sza, 2),
                         n_casts=len(ids), z_max=round(sub.z1.max(), 2),
                         Kd=round(sub.Kd.mean(), 4),
                         Kd_sd=round(sub.Kd.std(), 4) if len(ids) > 1 else np.nan,
                         cv=round(100*sub.Kd.std()/sub.Kd.mean(), 1) if len(ids) > 1 else np.nan,
                         r2_min=round(sub.r2.min(), 3),
                         cv_upper2m=round(cv[[i < 2 for i in cv.index]].mean(), 1)))

    axs[-1].invert_yaxis()
    axs[-1].set_xlabel(f"CV of {ED} in depth bin (%)")
    axs[-1].set_ylabel(DEPTH)
    axs[-1].set_title("Fluctuation intensity")
    axs[-1].legend(loc="upper left", fontsize=16)
    fig.suptitle(f"Station {s['t0']:%H:%M} occupied at multiple depths", y=1.09)
    fig.subplots_adjust(top=0.74, bottom=0.14)
    _finish(fig)
    return fig, pd.DataFrame(rows)


# ============================================================================
# SURFACE REFERENCE — day-level (v6)
# ============================================================================
def load_surface_local(path, es_source="mv", tz_shift_h=0.0):
    """load_surface, then shift iso_time by tz_shift_h.

    The E_sPAR surface logger recorded LOCAL time, so the shift is normally 0;
    the E_dzPAR profiler RTC used UTC and needs -7 (PDT). Set this if your
    surface unit also logged UTC.
    """
    s = load_surface(path, es_source=es_source)
    if tz_shift_h:
        s["iso_time"] = s["iso_time"] + pd.Timedelta(hours=tz_shift_h)
    return s


def surface_qc(surf, gap_factor=5.0, clear_sky_max=2500.0):
    """Dropouts, implausible peaks and negative samples.

    Negatives are kept, not filtered: a capped or night sensor legitimately
    reads slightly below zero, and removing them biases the dark offset high.
    """
    dt = surf.iso_time.diff().dt.total_seconds()
    thr = gap_factor * dt.median()
    gaps = surf.loc[dt > thr, ["iso_time"]].assign(gap_s=dt[dt > thr].values)
    return dict(dt=dt, dt_median=dt.median(), gap_thr=thr, gaps=gaps,
                n_negative=int((surf.Es < 0).sum()),
                peak=float(surf.Es.max()),
                peak_implausible=bool(surf.Es.max() > clear_sky_max))


def sky_steadiness(surf, window_s=30.0, lit_frac=0.02):
    """Rolling CV of E_s, masked to lit samples.

    This is the assumption the single-instrument Kd fit leans on: if the sky
    moved while the profiler was descending, that slope is contaminated rather
    than interesting. At night the rolling mean approaches zero and the CV
    explodes, so dark samples are masked out.
    """
    dt_s = surf.iso_time.diff().dt.total_seconds().median()
    win = max(3, int(round(window_s / dt_s))) if dt_s and dt_s > 0 else 30
    roll = surf.Es.rolling(win, center=True, min_periods=max(3, win//3))
    lit = surf.Es > max(10.0, lit_frac * surf.Es.max())
    return (100.0 * roll.std() / roll.mean()).where(lit), win, lit


def pair_stations(surf, ed_dir, tz_shift_h=-7.0, local_date=None,
                  pattern="E_dzPAR"):
    """Coverage and sky condition during each in-water station.

    A surface reference that does not span a cast window cannot normalise it.
    Es_cv_pct is the cast-validity flag: high values mean the sky moved
    mid-profile.
    """
    files = sorted(glob.glob(os.path.join(ed_dir, "*.CSV")) +
                   glob.glob(os.path.join(ed_dir, "*.csv")))
    files = [f for f in files if pattern in os.path.basename(f)]
    a0, a1 = surf.iso_time.iloc[0], surf.iso_time.iloc[-1]
    has_ci = "cloud_index" in surf.columns

    rows = []
    for f in files:
        try:
            prof = load_profile(f)
        except Exception:
            continue
        t = prof.iso_time + pd.Timedelta(hours=tz_shift_h)
        b0, b1 = t.iloc[0], t.iloc[-1]
        if local_date and b0.strftime("%Y%m%d") != str(local_date):
            continue
        ov = (min(a1, b1) - max(a0, b0)).total_seconds()
        cov = 100 * ov / (b1 - b0).total_seconds() if ov > 0 else 0.0
        w = surf[(surf.iso_time >= b0) & (surf.iso_time <= b1)]
        rows.append(dict(
            station=f"{b0:%H:%M}", file=os.path.basename(f), start=b0, end=b1,
            coverage_pct=round(cov),
            Es_mean=round(w.Es.mean(), 1) if len(w) else np.nan,
            Es_cv_pct=round(100*w.Es.std()/w.Es.mean(), 1) if len(w) > 1 else np.nan,
            cloud_index=round(w.cloud_index.mean(), 3) if (has_ci and len(w)) else np.nan,
            sza=round(w.sza.mean(), 2) if ("sza" in w and len(w)) else np.nan))
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def detect_burnoff(surf, resample="2min", lag=5):
    """Largest sustained rise in the cloud index — the marine-layer transition."""
    s = (surf.set_index("iso_time")["cloud_index"]
              .resample(resample).mean().interpolate())
    step = s.diff(lag)
    if not step.notna().any() or len(s) < 10:
        return None
    t = step.idxmax()
    return dict(time=t, rise=float(step.max()), series=s,
                before=float(s.loc[:t].tail(10).mean()),
                after=float(s.loc[t:].head(10).mean()))


# ---------------------------------------------------------------- figures ---
def fig_surface_timeline(surf, stations_tbl=None, gaps=None, figsize=(21, 12.0)):
    """Incident PAR with the clear-sky reference, plus sun angle below."""
    plt = _plt()
    import matplotlib.dates as mdates
    fig, (aT, aB) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1],
                                              "hspace": 0.10})
    if "clear_ref" in surf:
        aT.plot(surf.iso_time, surf.clear_ref, color="#2e6f95", lw=2.2, ls="--",
                label="clear-sky reference (shape only)")
    aT.plot(surf.iso_time, surf.Es, color="#e8a33d", lw=1.2, label=f"measured {ES}")
    if gaps is not None:
        for t in gaps.iso_time:
            aT.axvline(t, color="#c0392b", ls=":", lw=1.2)
    if stations_tbl is not None:
        for r in stations_tbl.itertuples():
            aT.axvspan(r.start, r.end, color="#2e6f95", alpha=0.16, lw=0)
            aT.annotate(r.station, (r.start, surf.Es.max()*0.98), fontsize=18,
                        rotation=90, va="top", fontweight="bold")
    aT.set_ylabel(f"{ES_PAR}\n{PPFD_UNITS}")
    aT.set_title("Surface incident PAR with casting windows")
    aT.legend(loc="lower center")

    aB.plot(surf.iso_time, surf.sza, color="#6c3483", lw=2.2)
    aB.axhline(60, color="#888780", ls=":", lw=1.5)
    aB.invert_yaxis()
    if stations_tbl is not None:
        for r in stations_tbl.itertuples():
            aB.axvspan(r.start, r.end, color="#2e6f95", alpha=0.16, lw=0)
    aB.set_ylabel("Solar zenith\nangle (deg)"); aB.set_xlabel("Time (local)")
    aB.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for a_ in (aT, aB):
        a_.margins(x=0.01)
    _finish(fig)
    return fig


def fig_cloud_index(surf, cv, stations_tbl=None, clear_thr=0.85,
                    steady_cv_pct=5.0, figsize=(21, 12.0)):
    """Cloud index and short-term sky variability."""
    plt = _plt()
    import matplotlib.dates as mdates
    lit = surf.Es > max(10.0, 0.02*surf.Es.max())
    fig, (aT, aB) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                 gridspec_kw={"hspace": 0.10})
    aT.plot(surf.iso_time, surf.cloud_index.where(lit), color="#6c3483", lw=1.4)
    aT.axhline(clear_thr, color="#c0392b", ls="--", lw=2, label=f"clear ≥ {clear_thr}")
    aT.set_ylim(0, 1.2); aT.set_ylabel("Cloud index")
    aT.set_title("Cloud index and short-term sky variability")
    aT.legend(loc="lower right")

    aB.plot(surf.iso_time, cv, color="#c0392b", lw=1.2)
    aB.axhline(steady_cv_pct, color="#2e6f95", ls="--", lw=2,
               label=f"{steady_cv_pct:.0f}% (steady sky)")
    aB.set_ylabel("Rolling CV (%)"); aB.set_xlabel("Time (local)")
    aB.legend(loc="upper right")
    aB.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    if stations_tbl is not None:
        for r in stations_tbl.itertuples():
            for a_ in (aT, aB):
                a_.axvspan(r.start, r.end, color="#2e6f95", alpha=0.16, lw=0)
    for a_ in (aT, aB):
        a_.margins(x=0.01)
    _finish(fig)
    return fig


def fig_par_vs_sza(surf, stations_tbl=None, noon=None, clear_thr=0.85,
                   figsize=(21, 8.8)):
    """Incident PAR against sun angle, and sky condition at each station."""
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize)
    if noon is not None:
        parts = [(surf[surf.iso_time < noon], "#2e6f95", "AM"),
                 (surf[surf.iso_time >= noon], "#c0392b", "PM")]
    else:
        parts = [(surf, "#2e6f95", None)]
    for d, col, lab in parts:
        d = d[d.sza < 90]
        a1.scatter(d.sza, d.Es, s=34, alpha=0.5, color=col, label=lab)
    a1.set_xlabel("Solar zenith angle (deg)")
    a1.set_ylabel(f"{ES_PAR} {PPFD_UNITS}")
    a1.set_title("Incident PAR vs sun angle")
    if noon is not None:
        a1.legend(markerscale=4)

    if stations_tbl is not None:
        for r in stations_tbl.itertuples():
            w = surf[(surf.iso_time >= r.start) & (surf.iso_time <= r.end)]
            if not len(w):
                continue
            a2.scatter(w.sza.mean(), w.cloud_index.mean(), s=240,
                       color="#6c3483", zorder=3)
            a2.annotate(r.station, (w.sza.mean(), w.cloud_index.mean()),
                        fontsize=18, xytext=(10, 9), textcoords="offset points",
                        fontweight="bold")
    a2.axhline(clear_thr, color="#c0392b", ls="--", lw=2, label=f"clear ≥ {clear_thr}")
    a2.set_ylim(0, 1.2)
    a2.set_xlabel("Solar zenith angle (deg)"); a2.set_ylabel("Cloud index")
    a2.set_title("Sky condition at each station"); a2.legend(loc="lower left")
    fig.tight_layout()
    _finish(fig)
    return fig


def fig_burnoff(burn, stations_tbl=None, clear_thr=0.85, figsize=(21, 7.5)):
    """Cloud index with the detected marine-layer transition marked."""
    plt = _plt()
    import matplotlib.dates as mdates
    s = burn["series"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s.index, s.values, color="#6c3483", lw=2.2)
    ax.axvline(burn["time"], color="#c0392b", lw=2.5, ls="--",
               label=f"transition {burn['time']:%H:%M}")
    ax.axhline(clear_thr, color="#888780", ls=":", lw=1.5)
    if stations_tbl is not None:
        for r in stations_tbl.itertuples():
            ax.axvspan(r.start, r.end, color="#2e6f95", alpha=0.16, lw=0)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Cloud index"); ax.set_xlabel("Time (local)")
    ax.set_title("Marine layer transition"); ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.margins(x=0.01)
    _finish(fig)
    return fig


def fig_surface_distribution(surf, dark_tbl=None, panel_w=9.2, panel_h=7.8):
    """Distribution of incident PAR, and the dark-run histograms beside it."""
    plt = _plt()
    n_dark = len(dark_tbl) if dark_tbl is not None else 0
    fig, axs = plt.subplots(1, 1 + n_dark,
                            figsize=(panel_w*(1+n_dark), panel_h), squeeze=False)
    a = axs[0][0]
    a.hist(surf.Es, bins=60, color="#e8a33d", edgecolor="#a8721f", linewidth=0.6)
    a.axvline(surf.Es.median(), color="#c0392b", lw=2.5,
              label=f"median {surf.Es.median():.0f}")
    a.set_xlabel(f"{ES_PAR} {PPFD_UNITS}"); a.set_ylabel("Count")
    a.set_title("Surface PAR distribution"); a.legend(loc="upper left")

    for j, r in enumerate(dark_tbl.itertuples() if n_dark else [], start=1):
        seg = surf.iloc[int(r.i0):int(r.i1)+1]
        a = axs[0][j]
        a.hist(seg.Es, bins=50, color="#6c3483", alpha=0.85)
        a.axvline(seg.Es.mean(), color="#c0392b", lw=2.5, label=f"mean {r.mean:.3f}")
        a.set_title(f"Dark {j}: {r.dur_min:.0f} min\nSD {r.sd:.3f}, "
                    f"{int(r.n_adc_codes)} codes", fontsize=21)
        a.set_xlabel(f"{ES_PAR} {PPFD_UNITS}"); a.set_ylabel("Count")
        a.legend()
    fig.tight_layout()
    _finish(fig)
    return fig


def fig_correction(tbl, clear_col="cloud_index", clear_thr=0.85, figsize=(24, 8.2)):
    """Does the geometric normalisation collapse the scatter?

    Kd is an apparent optical property: it depends on the light field as well
    as on the water. Kirk (1984) gives a leading 1/mu_0 term, mu_0 = cos(theta_w),
    which is pure slant path. Gordon (1989) showed that multiplying it out
    yields a quantity much closer to a property of the water itself:

        Kd_norm = Kd * cos(theta_w) = Kd / path_factor

    This is NOT a correction for instrument error. It places every cast on a
    common illumination geometry so that water-column change becomes visible.

    Left   : Kd against path factor, with the geometry-only prediction.
    Middle : raw and normalised Kd against sun angle.
    Right  : the residual after normalisation - if flat, the Kirk 1/mu_0 model
             is sufficient and no additional correction is warranted; a rise
             above some angle is where the instrument itself starts to fail.

    Under overcast the light field is diffuse and mu_0 is not set by the sun,
    so the fit uses clear-sky stations where enough of them exist.
    """
    plt = _plt()
    d = tbl.dropna(subset=["Kd", "path_factor"]).copy()

    used, which = d, "all stations"
    if clear_col in d and (d[clear_col] >= clear_thr).sum() >= 3:
        used  = d[d[clear_col] >= clear_thr]
        which = f"clear-sky only ({clear_col} >= {clear_thr})"

    fit = stats.linregress(used.path_factor, used.Kd) if len(used) >= 3 else None

    cv_raw  = 100*d.Kd.std()/d.Kd.mean()
    cv_norm = 100*d.Kd_norm.std()/d.Kd_norm.mean()

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=figsize)

    a1.errorbar(d.path_factor, d.Kd, yerr=d.Kd_sd, fmt="o", ms=10, capsize=4,
                color="#2e6f95", lw=0, elinewidth=1.5, label="measured")
    ref = d.loc[d.sza.idxmin()]
    xs = np.linspace(1.0, d.path_factor.max()*1.02, 40)
    a1.plot(xs, ref.Kd_norm*xs, color="#888780", ls="--", lw=2,
            label="geometry alone")
    if fit is not None:
        a1.plot(xs, fit.intercept + fit.slope*xs, color="#c0392b", lw=2,
                label=f"fit {R2} = {fit.rvalue**2:.3f}")
    a1.set_xlabel(f"Path factor {PATH_FAC}")
    a1.locator_params(axis="x", nbins=5)
    a1.set_ylabel(f"{KD_PAR} {PER_M}")
    a1.set_title(f"{KD_PAR} vs path factor"); a1.legend(loc="upper left")

    a2.errorbar(d.sza, d.Kd, yerr=d.Kd_sd, fmt="o", ms=10, capsize=4,
                color="#c0392b", lw=0, elinewidth=1.5,
                label=f"raw  (CV {cv_raw:.1f}%)")
    a2.errorbar(d.sza, d.Kd_norm, yerr=d.Kd_sd*d.Kd_norm/d.Kd, fmt="s", ms=10,
                capsize=4, color="#2e6f95", lw=0, elinewidth=1.5,
                label=f"× {COS_W}  (CV {cv_norm:.1f}%)")
    a2.set_xlabel("Solar zenith angle (deg)"); a2.set_ylabel(f"{KD_PAR} {PER_M}")
    a2.set_title("Before and after normalisation"); a2.legend(loc="upper left")

    a3.scatter(d.sza, d.Kd_norm, s=240, color="#2e6f95", zorder=3)
    a3.axhline(d.Kd_norm.mean(), color="#888780", ls="--", lw=2,
               label=f"mean {d.Kd_norm.mean():.3f}")
    for r in d.itertuples():
        a3.annotate(r.station, (r.sza, r.Kd_norm), fontsize=17, fontweight="bold",
                    xytext=(8, 8), textcoords="offset points")
    a3.set_xlabel("Solar zenith angle (deg)")
    a3.set_ylabel(f"{KD_COS_W} {PER_M}")
    a3.set_title("Residual after normalisation"); a3.legend(loc="lower right")

    fig.tight_layout()

    res = dict(cv_raw=cv_raw, cv_norm=cv_norm, n_used=len(used), which=which,
               pf_span=float(d.path_factor.max() - d.path_factor.min()),
               predicted_pct=100*(d.path_factor.max()/d.path_factor.min() - 1),
               observed_pct=100*(d.Kd.max()/d.Kd.min() - 1),
               residual_pct=100*(d.Kd_norm.max()/d.Kd_norm.min() - 1))
    if fit is not None:
        res.update(slope=fit.slope, intercept=fit.intercept,
                   r2=fit.rvalue**2, p=fit.pvalue,
                   Kd_at_zenith=fit.intercept + fit.slope)
        # is the residual still trending with sun angle?
        rf = stats.linregress(d.sza, d.Kd_norm)
        res.update(resid_slope=rf.slope, resid_p=rf.pvalue, resid_r2=rf.rvalue**2)
    _finish(fig)
    return fig, res
