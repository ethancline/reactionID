"""Render the ReactionID review deck: what was checked, what was fixed, and the results.

Every number in the deck is read from a file produced by a run, never typed in:

  report.json               make_report.py      both shipped models scored on valid.csv
  before_fix/report.json    make_report.py      the same, for the models before the review
  ablation.json             ablate.py --json    controls and leave-one-group-out
  audit.json                audit_data.py       overlap, single-input leakage screen, stale rows
  results/parity_*.json     verify_inference.py --json   cooker vs Python, event by event
  logs/export.log           export_onnx.py --verify      ONNX vs native model
  logs/plots_valid_file2.log, logs/plots_calib_file1.log  the cooker's own truth comparison
  model/output.pth          the MLP checkpoint (architecture, best epoch)

and every physics figure is drawn by the cooker (ReactionID_Plots_MC.xml) and
embedded as a PNG, so the deck is one portable file.

    python make_presentation.py      # -> ../muse/doc/slides/reaction_id_review.html
"""

import argparse
import base64
import json
import os
import re

import numpy as np
import torch

import make_slides as ms
import reactionData as rd
from make_slides import esc

C_MLP, C_GBDT, C_FAINT, C_GRID = ms.C_MLP, ms.C_GBDT, ms.C_FAINT, ms.C_GRID
MODELS = (("gbdt", "Boosted trees", C_GBDT), ("mlp", "MLP", C_MLP))


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def load_json(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing - run the step that writes it (see this script's docstring)")
    with open(path) as f:
        return json.load(f)


COUNTS = r"TP=(\d+) FN=(\d+) FP=(\d+) TN=(\d+)"


def parse_cooker_log(path):
    """The summary lines ReactionID::draw_truth_plots prints at finalize."""
    with open(path, errors="replace") as f:
        text = f.read()
    out = {}
    for key, pat in (("truth", r"ReactionID truth comparison: " + COUNTS),
                     ("ml_full", r"ReactionID beta, full reconstruction only: " + COUNTS),
                     ("cut_full", r"ReactionID cut-based, full reconstruction only: " + COUNTS)):
        m = re.search(pat, text)
        if m:
            tp, fn, fp, tn = map(int, m.groups())
            out[key] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
    for arm in ("left", "right"):
        m = re.search(rf"beta calibration, {arm} arm: positron \(beta=1\) peak ([-\d.]+) ns \[sigma ([\d.naN]+), n=(\d+)\]; "
                      rf"muon \(beta=p/E\) peak ([-\d.]+) ns \[sigma ([\d.naN]+), n=(\d+)\]", text)
        if m:
            out[f"cal_{arm}"] = {"pos": float(m.group(1)), "npos": int(m.group(3)),
                                 "mu": float(m.group(4)), "nmu": int(m.group(6))}
    # The vertex target cut the cooker loaded from its cuts database.
    m = re.search(r'"Vertex:target_cut:LH2:radius",\s*([\d.]+)', text)
    if m:
        out["lh2_radius"] = float(m.group(1))
    return out


def rates(c):
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    return {"eff": tp / (tp + fn) if tp + fn else float("nan"),
            "pur": tp / (tp + fp) if tp + fp else float("nan"),
            "spec": tn / (tn + fp) if tn + fp else float("nan"),
            "agr": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan"),
            "n": tp + tn + fp + fn}


def parse_export_log(path):
    """The `_compare` lines export_onnx.py --verify prints, one per model."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m = re.match(r"\s+(mlp|gbdt)\s+n=([\d,]+)\s+mean\|dP\|=(\S+)\s+max\|dP\|=(\S+)\s+\|dP\|>\S+: (\d+) .*label disagreements=(\d+).*(PASS|FAIL)", line)
            if m:
                out[m.group(1)] = {"n": int(m.group(2).replace(",", "")), "mean": float(m.group(3)), "max": float(m.group(4)),
                                   "over": int(m.group(5)), "flips": int(m.group(6)), "ok": m.group(7) == "PASS"}
    return out


def mlp_architecture(path):
    sd = torch.load(path, weights_only=False, map_location="cpu")["model_state_dict"]
    widths = [tuple(v.shape) for k, v in sd.items() if k.endswith("weight") and v.dim() == 2]
    return [w[1] for w in widths] + [widths[-1][0]]


def png(path, alt, caption=None, maxh="58vh"):
    if not os.path.exists(path):
        return f'<p class="caption">[missing figure: {esc(path)}]</p>'
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    cap = f'<p class="caption">{caption}</p>' if caption else ""
    return (f'<div class="figure"><img src="data:image/png;base64,{b64}" alt="{esc(alt)}" '
            f'style="max-height:{maxh}">{cap}</div>')


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------
def pct(v, d=1):
    return "&ndash;" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{d}f}%"


def num(v):
    return f"{v:,}"


def sci(v, d=1):
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    return f"{v / 10 ** e:.{d}f}&times;10<sup>{e}</sup>"


def swatch(col):
    return f'<span style="display:inline-block;width:.8em;height:.8em;border-radius:2px;background:{col};margin-right:.45em;vertical-align:-.05em"></span>'


# --------------------------------------------------------------------------
# charts
#
# Each SVG's viewBox is sized to roughly the width it renders at in its column,
# so one user unit is about one CSS pixel and the font sizes below are what the
# viewer actually sees - 15-17px beside ~28px body text. Text wears text tokens
# (the .cx classes injected into the page), never a series colour; identity is
# carried by the swatch or line beside it.
# --------------------------------------------------------------------------
CHART_CSS = """
  .cx{ font-family: var(--font-mono); font-size: 17px; fill: var(--text-dim); }
  .cx.lab{ font-size: 18px; fill: var(--text); }
  .cx.val{ font-size: 17px; fill: var(--text); font-variant-numeric: tabular-nums; }
  .cx.ttl{ font-size: 16px; fill: var(--text-faint); }
  .cx.box{ font-family: var(--font-display); font-weight: 700; font-size: 19px; fill: var(--text); }
  table.data td.nowrap{ white-space: nowrap; }
  .cx.on{ fill: #1a1206; }
  table.data.compact{ font-size: clamp(13px,1.7vmin,20px); }
  table.data.compact td, table.data.compact th{ padding-right: 18px; }
  .legend-row{ display:flex; gap: 28px; flex-wrap: wrap; margin-top: 6px; font-family: var(--font-mono);
               font-size: clamp(12px,1.5vmin,17px); color: var(--text-dim); }
"""
C_NEUTRAL = "#7f93a8"   # single-series bars that belong to neither model
SURFACE = "#121a22"


def nice_ticks(lo, hi, n=5):
    """Round tick positions covering [lo, hi]: steps of 1, 2 or 5 x 10^k."""
    span = hi - lo if hi > lo else abs(hi) or 1.0
    raw = span / max(n - 1, 1)
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = np.floor(lo / step) * step
    ticks = np.arange(first, hi + step * 0.999, step)
    return [float(round(t, 10)) for t in ticks]


def svg_bars(rows, color, W=700, label_w=200, fmt=str, tick_fmt=None, ref=None, xlabel="", row_h=36, lo=None, hi=None):
    """One series of horizontal bars with the value at the bar's end.

    rows: [(label, value)]. With `ref`, bars grow from ref (e.g. 1.0 for a ratio)."""
    tick_fmt = tick_fmt or fmt
    vals = [v for _, v in rows]
    base = 0.0 if ref is None else ref
    lo = min([base] + vals) if lo is None else lo
    hi = max([base] + vals) if hi is None else hi
    ticks = nice_ticks(lo, hi)
    lo, hi = min(ticks[0], lo), max(ticks[-1], hi)
    T, B, R = 8, 56, 70
    n = len(rows)
    H = T + n * row_h + B
    pw = W - label_w - R

    def X(v):
        return label_w + pw * (v - lo) / (hi - lo)

    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(xlabel)}">']
    for t in ticks:
        s.append(f'<line x1="{X(t):.1f}" y1="{T}" x2="{X(t):.1f}" y2="{T+n*row_h}" stroke="{C_GRID}"/>')
        s.append(f'<text x="{X(t):.1f}" y="{T+n*row_h+20}" text-anchor="middle" class="cx">{esc(tick_fmt(t))}</text>')
    s.append(f'<line x1="{X(base):.1f}" y1="{T}" x2="{X(base):.1f}" y2="{T+n*row_h}" stroke="{C_FAINT}" stroke-width="1.5"/>')
    bh = min(20, row_h - 12)
    for i, (label, v) in enumerate(rows):
        yc = T + i * row_h + row_h / 2
        s.append(f'<text x="{label_w-12}" y="{yc+5:.1f}" text-anchor="end" class="cx lab">{esc(label)}</text>')
        xa, xb = sorted((X(base), X(v)))
        s.append(f'<rect x="{xa:.1f}" y="{yc-bh/2:.1f}" width="{max(xb-xa,1.5):.1f}" height="{bh}" rx="3" fill="{color}">'
                 f'<title>{esc(label)}: {esc(fmt(v))}</title></rect>')
        right = v >= base
        s.append(f'<text x="{(xb+8) if right else (xa-8):.1f}" y="{yc+5:.1f}" text-anchor="{"start" if right else "end"}" class="cx val">{esc(fmt(v))}</text>')
    s.append(f'<text x="{label_w+pw/2:.0f}" y="{H-8}" text-anchor="middle" class="cx ttl">{esc(xlabel)}</text>')
    s.append("</svg>")
    return "\n".join(s)


def svg_eff_fpr(rep):
    """Signal efficiency vs false-positive rate on true non-decays, log scale.

    With AUCs above 0.99 a linear ROC is a line in the corner; the log axis is
    where the two models - and the operating points - actually differ.
    """
    W, H = 720, 500
    L, R, T, B = 104, 24, 16, 70
    pw, ph = W - L - R, H - T - B
    x0, x1 = 0.5, 1.0
    y0, y1 = -5.0, 0.0

    def X(e):
        return L + pw * (e - x0) / (x1 - x0)

    def Y(f):
        return T + ph * (1 - (np.log10(max(f, 10 ** y0)) - y0) / (y1 - y0))

    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Signal efficiency against false-positive rate on true non-decays, both models">']
    for e in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s.append(f'<line x1="{X(e):.1f}" y1="{T}" x2="{X(e):.1f}" y2="{T+ph}" stroke="{C_GRID}"/>')
        s.append(f'<text x="{X(e):.1f}" y="{T+ph+24}" text-anchor="middle" class="cx">{e:.1f}</text>')
    for k in range(int(y0), int(y1) + 1):
        yy = Y(10 ** k)
        s.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{L+pw}" y2="{yy:.1f}" stroke="{C_GRID}"/>')
        lab = {0: "100%", -1: "10%", -2: "1%", -3: "0.1%", -4: "0.01%", -5: "0.001%"}[k]
        s.append(f'<text x="{L-10}" y="{yy+5:.1f}" text-anchor="end" class="cx">{lab}</text>')
    s.append(f'<text x="{L+pw/2:.0f}" y="{H-12}" text-anchor="middle" class="cx ttl">signal efficiency (share of relevant decays found)</text>')
    s.append(f'<text transform="translate(16,{T+ph/2:.0f}) rotate(-90)" text-anchor="middle" class="cx ttl">false-positive rate on true non-decays</text>')
    for key, lab, col in MODELS:
        pts = [(f, t) for f, t in rep[key]["roc_nondecay"] if t >= x0 and f > 0]
        d = " ".join(("M" if i == 0 else "L") + f"{X(t):.1f},{Y(f):.1f}" for i, (f, t) in enumerate(pts))
        s.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
    for key, lab, col in MODELS:
        a = rep[key]["at_threshold"]
        s.append(f'<circle cx="{X(a["efficiency"]):.1f}" cy="{Y(a["fpr_nondecay"]):.1f}" r="7" fill="{col}" stroke="{SURFACE}" stroke-width="2.5">'
                 f'<title>{lab}: shipped threshold {rep[key]["threshold"]:.4f} - efficiency {100*a["efficiency"]:.2f}%, '
                 f'false-positive rate {100*a["fpr_nondecay"]:.3f}%</title></circle>')
    lx, ly = L + 18, T + 26
    for i, (key, lab, col) in enumerate(MODELS):
        yy = ly + 28 * i
        s.append(f'<line x1="{lx}" y1="{yy-5}" x2="{lx+22}" y2="{yy-5}" stroke="{col}" stroke-width="3"/>')
        s.append(f'<text x="{lx+32}" y="{yy}" class="cx lab">{esc(lab)} &#183; AUC {rep[key]["auc_vs_nondecay"]:.4f}</text>')
    s.append(f'<circle cx="{lx+11}" cy="{ly+51}" r="6" fill="{C_FAINT}" stroke="{SURFACE}" stroke-width="2"/>'
             f'<text x="{lx+32}" y="{ly+56}" class="cx">shipped operating point</text>')
    s.append("</svg>")
    return "\n".join(s)


def svg_regions(rep):
    """Schematic, not to scale: where the label is positive along the beam line."""
    rc = rep["dataset"]["region_counts"]
    lo = rep["prior"]["window_mm"][0]
    W, H = 1400, 190
    boxes = [(20, 420, "upstream", f"Z {lo:+,.0f} to -200 mm", rc["upstream"], True),
             (450, 330, "target region", "|Z|, R < 200 mm", rc["target"], True),
             (810, 570, "downstream", "Z > +200 mm (label 0)", rc["downstream"], False)]
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Decay regions along the beam axis">']
    s.append(f'<path d="M20,34 L20,24 L780,24 L780,34" fill="none" stroke="#ffb454" stroke-width="2"/>')
    s.append('<text x="400" y="16" text-anchor="middle" class="cx lab" style="fill:#ffb454">label = 1 &#183; decay_relevant</text>')
    for x, w, name, zr, n, on in boxes:
        s.append(f'<rect x="{x}" y="44" width="{w}" height="96" rx="4" class="dbox{" active" if on else ""}"/>')
        s.append(f'<text x="{x+w/2}" y="80" text-anchor="middle" class="cx box{" on" if on else ""}">{name}</text>')
        s.append(f'<text x="{x+w/2}" y="106" text-anchor="middle" class="cx{" on" if on else " lab"}">{zr}</text>')
        s.append(f'<text x="{x+w/2}" y="129" text-anchor="middle" class="cx{" on" if on else ""}">{n:,} events</text>')
    s.append(f'<text x="20" y="176" class="cx ttl">beam &#8594; &#160; schematic, not to scale &#183; validation events after cuts &#183; '
             f'plus {rc["no decay"]:,} events with no decay (label 0)</text>')
    s.append("</svg>")
    return "\n".join(s)


def svg_pipeline(stages):
    boxes = [("g4PSI MC", "2 productions"), ("cooker chain", f"{len(stages)} stages"),
             ("feature Builder", "one vector"), ("muonDecay_out", "CSV per event"),
             ("Python", "train, export"), ("ReactionID", "ONNX in cooker"),
             ("Plots_MC", "vs MC truth")]
    W, H = 1440, 200
    bw, gap = 188, 20
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Processing pipeline">',
         '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{C_FAINT}"/></marker></defs>']
    for i, (a, b) in enumerate(boxes):
        x = 4 + i * (bw + gap)
        on = i == 2
        s.append(f'<rect x="{x}" y="20" width="{bw}" height="84" rx="4" class="dbox{" active" if on else ""}"/>')
        s.append(f'<text x="{x+bw/2}" y="56" text-anchor="middle" class="cx box{" on" if on else ""}" >{a}</text>')
        s.append(f'<text x="{x+bw/2}" y="82" text-anchor="middle" class="cx{" on" if on else ""}">{b}</text>')
        if i < len(boxes) - 1:
            s.append(f'<path d="M{x+bw+2},62 L{x+bw+gap-2},62" class="darrow" marker-end="url(#arr)"/>')
    xb = 4 + 2 * (bw + gap) + bw / 2
    xr = 4 + 5 * (bw + gap) + bw / 2
    s.append(f'<path d="M{xb},106 L{xb},148 L{xr},148 L{xr},110" class="darrow" marker-end="url(#arr)" stroke-dasharray="5 5"/>')
    s.append(f'<text x="{(xb+xr)/2}" y="178" text-anchor="middle" class="cx">the same Builder feeds the model inside the cooker</text>')
    s.append("</svg>")
    return "\n".join(s)


def svg_split(split, valid_n):
    fit, hold = int(split["fit_events"]), int(split["holdout_events"])
    tot = fit + hold + valid_n
    W, H = 1420, 44
    x = 0.0
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Data split">']
    for name, n, col in (("fit", fit, C_GBDT), ("holdout", hold, C_MLP), ("validation", valid_n, C_NEUTRAL)):
        w = W * n / tot
        s.append(f'<rect x="{x:.1f}" y="4" width="{max(w-3,1):.1f}" height="36" rx="3" fill="{col}"><title>{name}: {n:,} events</title></rect>')
        x += w
    s.append("</svg>")
    rows = "".join(f'<tr><td class="nowrap">{swatch(c)}{nm}</td><td class="num">{n:,}</td><td>{use}</td></tr>'
                   for nm, n, c, use in (("fit", fit, C_GBDT, "model weights and tree splits"),
                                         ("holdout", hold, C_MLP, "MLP early stopping and both thresholds"),
                                         ("validation", valid_n, C_NEUTRAL, "every reported number, scored once")))
    return "\n".join(s) + f'<table class="data compact" style="margin-top:6px"><tr><th>part</th><th>events</th><th>used for</th></tr>{rows}</table>'


# --------------------------------------------------------------------------
# slides
# --------------------------------------------------------------------------
def slide(eyebrow, title, body, notes="", cls=""):
    return (f'<section class="slide {cls}" data-notes="{esc(notes)}"><div class="slide-inner">'
            f'<div class="eyebrow">{eyebrow}</div><h1 class="title" style="font-size:clamp(24px,4vmin,52px)">{title}</h1>'
            f'<hr class="rule"/>{body}</div></section>')


def build(R, R0, A, AU, PAR, EXP, LOG, CAL, arch, plots, calplots):
    S = []
    g, m = R["gbdt"], R["mlp"]
    ds, pr, sp = R["dataset"], R["prior"], R["split"]
    mans = R["manifests"]
    stages = [st["name"] for st in (mans[0]["stages"] if mans else [])]
    tr = rates(LOG["truth"]) if "truth" in LOG else None

    # 1 -------------------------------------------------------------------
    S.append(f'''<section class="slide title-slide active" data-notes="{esc("Every number in this deck is read from a result file at build time; see the docstring of make_presentation.py for the list.")}"><div class="slide-inner">
<div class="eyebrow">MUSE &#183; reactionID &#183; review</div>
<h1 class="title">Tagging muon <span class="accent">decays in flight</span><br>with a classifier inside the cooker</h1>
<hr class="rule"/>
<p class="lede">A review of the ML model, the cooker inputs that feed it, and the cooker code that judges it. Bugs were fixed and the models retrained, and the results are below.</p>
<div class="meta-row">
<div><b>Training MC</b><span>{esc(", ".join(mm["tag"] for mm in mans))}</span></div>
<div><b>Events</b><span>{num(ds["train_events"])} train &#183; {num(ds["valid_events"])} validation</span></div>
<div><b>Models</b><span>boosted trees ({g["rounds"]} rounds) &#183; MLP ({m["n_features"]} inputs)</span></div>
<div><b>Exported</b><span>{esc(sp.get("exported") or "")}</span></div>
</div></div></section>''')

    # 2 -------------------------------------------------------------------
    S.append(slide("01 &#183; the problem", "A decay in flight can fake a scatter",
        f'''<div class="two-col"><div><ul class="points">
<li><span>A muon that decays before or inside the target sends a <b>positron</b> into a scattering arm. A decay after the target cannot contaminate the scattering sample.</span></li>
<li><span>So the label is <code>decay_relevant</code>: a truth decay upstream of or inside the target region (|Z|, R &lt; 200 mm). It is {pct(ds["raw_label_fraction"])} of MC events, {pct(ds["decay_fraction"])} after cuts. {pct(ds["raw_muondecay_fraction"],0)} of the simulated muons decay somewhere.</span></li>
<li><span>In a real beam the rate is tiny. At {pr["momentum_MeV"]:.0f} MeV/c the decay length is <b>{pr["decay_length_m"]:,.0f} m</b>, so the chance of decaying over the labelled {pr["window_mm"][1]-pr["window_mm"][0]:,.0f} mm is <b>{pct(pr["physical"],2)}</b>. Precision is quoted at that prior.</span></li>
</ul></div><div class="callout"><b>Why regions matter.</b> The MC decay vertices run from Z = {pr["window_mm"][0]:+,.0f} mm to beyond +5 m. Target-region decays are the hard case: the positron starts where a real scatter would.</div></div>
<div class="diagram-wrap">{svg_regions(R)}</div>''',
        notes=f"physical prior = 1 - exp(-L/(beta gamma c tau)), L from the label window {pr['window_mm']} mm; reactionData.physical_prior. Previous decks quoted {pr['previous']}."))

    # 3 -------------------------------------------------------------------
    S.append(slide("02 &#183; pipeline", "One feature vector, built once, used twice",
        f'''<div class="diagram-wrap">{svg_pipeline(stages)}</div>
<div class="two-col" style="margin-top:12px"><ul class="points">
<li><span><code>decayfeatures::Builder</code> is the only code that builds the vector. <code>muonDecay_out</code> writes it to CSV, and <code>ReactionID</code> passes it to ONNX Runtime.</span></li>
<li><span>Each ONNX file declares its own <code>input_columns</code>, threshold and label. At startup the plugin maps the names onto the builder, and refuses to run if any column is missing.</span></li>
</ul><ul class="points">
<li><span>NaN means "not measured" and is passed through as NaN. The trees split on it, and the MLP graph adds is-missing indicators.</span></li>
<li><span>The chain: <span class="mono" style="font-size:.8em">{" &#8594; ".join(esc(x) for x in stages)}</span></span></li>
</ul></div>''', notes="Stage list read from the training manifest written by cook_mc_chain.sh."))

    # 4 -------------------------------------------------------------------
    chv = ds["chv_rate_by_region"]
    chv_rows = list(chv.items())
    prod_rows = "".join(f"<tr><td class='mono'>{esc(o['train'])}</td><td class='num'>{num(o['rows'])}</td><td class='num'>{o['shared_with_valid']}</td>"
                        f"<td class='num'>{('all ' + format(o['weight_min'], 'g')) if o['weight_min'] == o['weight_max'] else format(o['weight_min'], 'g') + '&ndash;' + format(o['weight_max'], 'g')}</td></tr>" for o in AU["overlap"])
    S.append(slide("03 &#183; data", "Two productions pooled, one untouched validation file",
        f'''<div class="two-col"><div>
<table class="data"><tr><th>training CSV</th><th>rows</th><th>rows shared with valid</th><th>MC weight</th></tr>{prod_rows}
<tr><td class='mono'>{esc(AU["valid"])} (validation)</td><td class='num'>{num(AU["valid_rows"])}</td><td class='num'>&ndash;</td><td class='num'></td></tr></table>
<p class="caption">Rows are compared on the {AU["input_columns"]} model inputs, rounded to 10<sup>-6</sup>. A single shared row is one near-empty event pattern.</p>
<ul class="points"><li><span>Cuts applied in <code>load_csv</code>: <code>has_truth</code> and <code>chv_veto == 0</code>. {num(ds["valid_events_raw"])} &#8594; {num(ds["valid_events"])} validation events.</span></li></ul>
</div><div>
{svg_bars(chv_rows, C_NEUTRAL, W=660, label_w=150, fmt=lambda v: f"{100*v:.1f}%", tick_fmt=lambda v: f"{100*v:.0f}%", xlabel="share of events removed by the chv veto, by truth region")}
<div class="callout warn"><b>The chv veto does not treat the classes equally.</b> It removes {pct(chv["no decay"],0)} of no-decay events but only {pct(chv["upstream"],0)} of upstream decays. It comes from g4PSI truth leaves, so on data it applies only if the trigger vetoes the same thing.</div>
</div></div>''', notes="audit.json (audit_data.py) and report.json dataset block."))

    # 5 -------------------------------------------------------------------
    cols = rd.EXCLUDED_COLUMNS
    by_reason = {}
    for c, why in cols.items():
        by_reason.setdefault(why, []).append(c)
    excl = "".join(f"<tr><td>{esc(w)}</td><td class='num'>{len(v)}</td></tr>" for w, v in by_reason.items())
    imp = {f["name"] for f in R["importance"]["features"]}
    groups = {}
    for name in [f["name"] for f in R["importance"]["features"]]:
        key = ("vertex" if name.startswith(("vtx_", "n_vertices")) else
               "track-to-scintillator correlation" if "_corr_" in name or name == "sps_side" else
               "time of flight" if name in ("tof_raw", "bh_avg_time", "bh_both_planes") else
               "STT track" if name.startswith("stt_") else "GEM track" if name.startswith("gem_") else
               "beam monitor" if name.startswith("bm") else "SPS walls" if name.startswith("sps") else
               "BH walls" if name.startswith("bh") else "veto" if name.startswith("veto") else "other")
        groups[key] = groups.get(key, 0) + 1
    grp = "".join(f"<tr><td>{esc(k)}</td><td class='num'>{v}</td></tr>" for k, v in sorted(groups.items(), key=lambda kv: -kv[1]))
    S.append(slide("04 &#183; inputs", f"{len(imp)} raw inputs, all detector-level",
        f'''<div class="two-col"><div><table class="data"><tr><th>group</th><th>columns</th></tr>{grp}</table></div>
<div><table class="data"><tr><th>excluded on purpose</th><th>columns</th></tr>{excl}</table>
<div class="callout">Columns with no information on the training set are dropped mechanically. On this pure-&#956;&#8314; single-momentum MC that includes the BH PID and the momentum, which are constant. <b>The strongest single input is <code>{esc(AU["single_feature_auc"][0]["name"])}</code> at AUC {AU["single_feature_auc"][0]["auc"]:.3f}</b>, so no one column leaks the label.</div></div></div>''',
        notes="Groups from the ONNX input_columns; exclusions from reactionData.EXCLUDED_COLUMNS; single-input screen in audit.json."))

    # 6 -------------------------------------------------------------------
    gp = g.get("params", {})
    S.append(slide("05 &#183; models &amp; protocol", "Nothing is chosen on the validation file",
        f'''<div class="two-col"><div>{svg_split(sp, ds["valid_events"])}</div><ul class="points" style="margin-top:0">
<li><span>{swatch(C_GBDT)}<b>Boosted trees</b>: sklearn HistGradientBoosting. Learning rate {gp.get("learning_rate")}, {gp.get("max_leaf_nodes")} leaves, L2 {gp.get("l2_regularization")}, {g["rounds"]} rounds. NaN handled natively.</span></li>
<li><span>{swatch(C_MLP)}<b>MLP</b>: {" &#8594; ".join(str(w) for w in arch)} with ReLU, batch-norm and dropout. One-hot and standardisation are baked into the ONNX graph. Best holdout loss at epoch {m["best_epoch"]}.</span></li>
<li><span>The holdout ({float(sp["holdout_frac"]):.0%} of training, stratified, seeded) is recorded in the checkpoint, so <code>export_onnx.py</code> rebuilds the same one to set both thresholds.</span></li>
</ul></div>''', notes="Before the review, the MLP early-stopped on valid.csv, and both thresholds were the 90% points measured on valid.csv - the same file the results were then quoted on."))

    # 7 -------------------------------------------------------------------
    tiles = "".join(
        f'<div style="flex:1;background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:14px 18px">'
        f'<div class="caption" style="margin:0">{swatch(c)}{esc(lab)} at threshold {R[k]["threshold"]:.3f}</div>'
        f'<div style="font-family:var(--font-display);font-size:clamp(22px,3.4vmin,40px);color:var(--text)">{pct(R[k]["at_threshold"]["efficiency"])}</div>'
        f'<div class="caption" style="margin:0">efficiency &#183; FPR {pct(R[k]["at_threshold"]["fpr_nondecay"],3)} on non-decays &#183; precision {pct(R[k]["at_threshold"]["precision_at_prior"],0)} at the physical prior</div></div>'
        for k, lab, c in MODELS)
    S.append(slide("06 &#183; performance", "Boosted trees lead at every efficiency",
        f'''<div class="two-col"><div>{svg_eff_fpr(R)}</div><div>
<div style="display:flex;flex-direction:column;gap:12px">{tiles}</div>
<div class="callout">Thresholds come from the training holdout. On validation the efficiency lands near 90%, not exactly on it. That is how you can tell the threshold was not tuned on this file.</div></div></div>''',
        notes="Curves: report.json roc_nondecay (validation positives vs region-0 events). Dots: the thresholds carried in the ONNX metadata."))

    # 8 -------------------------------------------------------------------
    def auc_cell(r):
        return "&ndash;" if r["auc"] is None else f"{r['auc']:.4f}"

    wp = {k: {round(w["target_eff"], 2): w for w in R[k]["working_points"]} for k, _, _ in MODELS}
    wrows = "".join(
        f"<tr><td class='num'>{pct(t,0)}</td>"
        + "".join(f"<td class='num'>{pct(wp[k][t]['fpr'],3)}</td><td class='num'>{pct(wp[k][t]['precision_at_prior'],1)}</td>" for k, _, _ in MODELS)
        + "</tr>" for t in sorted(wp["gbdt"]))
    S.append(slide("07 &#183; working points", "Precision at the physical prior is the real-data number",
        f'''<table class="data"><tr><th rowspan="2">signal efficiency</th>{"".join(f'<th colspan="2">{swatch(c)}{lab}</th>' for _, lab, c in MODELS)}</tr>
<tr>{("<th>false positives (no decay)</th><th>precision at " + format(pr["physical"]*100, ".2f") + "% prior</th>") * 2}</tr>{wrows}</table>
<div class="callout warn">Measured on validation, so these rows are for reading, not for choosing a threshold. The shipped thresholds are the ones on the previous slide. False positives count <b>true non-decays only</b>. Before the review they also counted downstream decays, which make up {ds["region_counts"]["downstream"]/(ds["region_counts"]["downstream"]+ds["region_counts"]["no decay"]):.0%} of the label-0 events in this MC.</div>''',
        notes=f"Physical prior derived: {pr['physical']:.5f}. The earlier decks used {pr['previous']}, which is {pr['previous']/pr['physical']:.1f}x too high and inflates precision."))

    # 9 -------------------------------------------------------------------
    reg = "".join(
        f"<tr><td>{esc(a['name'])}</td><td class='num'>{num(a['n'])}</td><td>{esc(a['metric'])}</td>"
        f"<td class='num'>{pct(a['value'],2)}</td><td class='num'>{pct(b['value'],2)}</td>"
        f"<td class='num'>{auc_cell(a)}</td><td class='num'>{auc_cell(b)}</td></tr>"
        for a, b in zip(g["regions"], m["regions"]))
    tgt = {r["name"]: r for r in g["regions"]}["target"]
    S.append(slide("08 &#183; by decay region", "Target-region decays are the hard case",
        f'''<table class="data"><tr><th>region</th><th>events</th><th>rate shown</th><th>{swatch(C_GBDT)}trees</th><th>{swatch(C_MLP)}MLP</th><th>AUC vs no decay (trees)</th><th>(MLP)</th></tr>{reg}</table>
<div class="callout">At the shipped threshold the trees find {pct(tgt["value"],1)} of target-region decays, against {pct({r["name"]: r for r in g["regions"]}["upstream"]["value"],1)} of upstream ones. A positron that starts inside the target looks most like a real scatter. <b>Downstream decays are label 0</b>, so their row is a false-positive rate. Before the review the report called it "efficiency".</div>''',
        notes="report.json regions block, at each model's own shipped threshold."))

    # 10 ------------------------------------------------------------------
    xcheck = ""
    if tr:
        c, t = LOG["truth"], R["truth"]
        same = all(c[k] == t[k] for k in ("tp", "fn", "fp", "tn"))
        xcheck = (f'<table class="data"><tr><th></th><th>TP</th><th>FN</th><th>FP</th><th>TN</th></tr>'
                  f'<tr><td>cooker (ReactionID_Plots_MC)</td><td class="num">{num(c["tp"])}</td><td class="num">{num(c["fn"])}</td><td class="num">{num(c["fp"])}</td><td class="num">{num(c["tn"])}</td></tr>'
                  f'<tr><td>Python (make_report.py)</td><td class="num">{num(t["tp"])}</td><td class="num">{num(t["fn"])}</td><td class="num">{num(t["fp"])}</td><td class="num">{num(t["tn"])}</td></tr></table>'
                  f'<div class="callout {"good" if same else "warn"}"><b>{"Identical" if same else "Different"}.</b> Two independent code paths score the same events: the cooker reads the <code>ReactionID</code> branch and MC truth, and Python loads the joblib model on the CSV. '
                  f'Efficiency {pct(tr["eff"],2)}, purity {pct(tr["pur"],2)} (at the enriched MC rate), specificity {pct(tr["spec"],3)}.</div>')
    S.append(slide("09 &#183; judged in the cooker", "The cooker's truth comparison agrees with Python event for event",
        f'''<div class="two-col"><div>{png(plots["truth_confusion"], "confusion matrix", "cooker: ReactionID_Plots_MC.xml on validation file 2", maxh="46vh")}</div>
<div>{xcheck}{png(plots["truth_agreement_by_region"], "agreement by region", None, maxh="28vh")}</div></div>''',
        notes="Cooker counts parsed from logs/plots_valid_file2.log; Python counts from report.json['truth']."))

    # 11 ------------------------------------------------------------------
    S.append(slide("10 &#183; failure modes", "Where the missed decays are",
        f'''{png(plots["truth_decay_position"], "decay position found vs missed", "truth decay Z and R of relevant decays, found vs missed, each normalised to 1", maxh="50vh")}
<div class="callout">Misses concentrate inside the target region and pile up at its <b>edges</b> (Z &#8776; +200 mm, R &#8776; 200 mm). That region is a 200 mm box, while the vertex cut the cooker applies to LH2 is R &lt; {LOG.get("lh2_radius", float("nan")):g} mm. Some of these "misses" are decays the label calls relevant but that sit well outside the liquid.</div>''',
        notes="Drawn by ReactionID::draw_truth_plots."))

    S.append(slide("10 &#183; failure modes", "How the scores separate",
        f'''<div class="two-col narrow-right"><div>{png(plots["truth_score"], "score by truth", "classifier score by truth, log scale; dashed line = shipped threshold", maxh="60vh")}</div>
<ul class="points"><li><span>Almost every event sits at one end: background near 0, relevant decays near 1.</span></li>
<li><span>The shipped threshold ({R["gbdt"]["threshold"]:.3f}) sits where the decay curve begins to climb, so moving it trades efficiency against false positives quickly.</span></li></ul></div>''',
        notes="Boosted trees, the model the cooker runs by default."))

    # 12 ------------------------------------------------------------------
    body = png(plots["beta_full_reco"], "beta by verdict and by truth", "vertices with a full reconstruction: DOCA and fiducial cuts, BH correlation, SPS hit", maxh="36vh")
    if "ml_full" in LOG and "cut_full" in LOG:
        a, b = rates(LOG["ml_full"]), rates(LOG["cut_full"])
        body += (f'<table class="data"><tr><th>same {num(a["n"])} vertices</th><th>efficiency</th><th>purity</th><th>agreement with truth</th></tr>'
                 f'<tr><td>{swatch(C_GBDT)}classifier</td><td class="num">{pct(a["eff"],1)}</td><td class="num">{pct(a["pur"],1)}</td><td class="num">{pct(a["agr"],1)}</td></tr>'
                 f'<tr><td>plugin cuts (&#946;<sub>out</sub> &#8805; cut, &#952; for &#960;)</td><td class="num">{pct(b["eff"],1)}</td><td class="num">{pct(b["pur"],1)}</td><td class="num">{pct(b["agr"],1)}</td></tr></table>')
    S.append(slide("11 &#183; against the cuts", "Where the cuts can run, compare them on the same vertices",
        body, notes="New in this review: the cooker scores its own cut decision (is_not_decay_event) against MC truth on exactly the vertices the classifier is scored on. Counts are per vertex, not per event. The beta alignment comes from training file 1."))

    # 13 ------------------------------------------------------------------
    cal = ""
    if "cal_left" in CAL:
        cal = "".join(f"<tr><td>{arm}</td><td class='num'>{CAL['cal_'+arm]['pos']:.3f} ns</td><td class='num'>{num(CAL['cal_'+arm]['npos'])}</td>"
                      f"<td class='num'>{CAL['cal_'+arm]['mu']:.3f} ns</td><td class='num'>{num(CAL['cal_'+arm]['nmu'])}</td></tr>" for arm in ("left", "right"))
        cal = f'<table class="data"><tr><th>arm</th><th>positron peak</th><th>n</th><th>muon peak</th><th>n</th></tr>{cal}</table>'
    S.append(slide("12 &#183; beta calibration", "The benchmark is now calibrated on training data",
        f'''{png(calplots["truth_beta_calibration"], "beta shift from truth", "required beta shift from MC truth, training file 1", maxh="40vh")}
<div class="two-col" style="margin-top:10px"><div>{cal}</div><div class="callout" style="margin-top:0">MC truth fixes the timing offset. A target decay puts a positron at &#946; = 1, and a scattered muon travels at &#946; = p/E. <b>Before</b>, the offset was fitted on validation file 2 and then used to score the cuts on file 2. <b>Now</b> it is fitted on training file 1 and applied to file 2.
<br><br>The positron and muon peaks differ, so a single offset per arm is only approximately right.</div></div>''',
        notes="logs/plots_calib_file1.log; the values are recorded in mc17606_beta_alignment.txt."))

    # 14 ------------------------------------------------------------------
    def top_rows(key, n=10):
        return [(f["name"], f["mean"]) for f in R[key]["features"][:n]]
    n_zero = sum(1 for f in R["importance"]["features"] if f["mean"] <= 0)
    S.append(slide("13 &#183; what the models use", "Timing dominates, and much of it is absolute time",
        f'''<div class="two-col" style="grid-template-columns:1fr 1fr">
<div><p class="caption">{swatch(C_GBDT)}boosted trees, top 10</p>{svg_bars(top_rows("importance"), C_GBDT, W=700, label_w=190, fmt=lambda v: f"{v:.4f}", tick_fmt=lambda v: f"{v:.3f}", xlabel="drop in ROC AUC when shuffled")}</div>
<div><p class="caption">{swatch(C_MLP)}MLP, top 10 (note the different scale)</p>{svg_bars(top_rows("importance_mlp"), C_MLP, W=700, label_w=190, fmt=lambda v: f"{v:.3f}", tick_fmt=lambda v: f"{v:.2f}", xlabel="drop in ROC AUC when shuffled")}</div></div>
<p class="caption" style="text-align:left">Permutation importance on {num(R["importance"]["n_events"])} validation events, measured on the exported ONNX files. {n_zero} of {len(R["importance"]["features"])} columns do nothing for the trees. The per-wall times are absolute times, so on data they move with every calibration. The MLP leans far harder on single columns: shuffling <code>tof_raw</code> alone costs it {R["importance_mlp"]["features"][0]["mean"]:.2f} in AUC.</p>''',
        notes="report.json importance / importance_mlp."))

    # 15 ------------------------------------------------------------------
    base_err = 1 - A["all_inputs"]["auc"]
    arows = [(gr["label"], (1 - gr["auc"]) / base_err) for gr in sorted(A["groups"], key=lambda x: x["auc"])]
    S.append(slide("14 &#183; ablation", "Most of the signal is redundant, and some of it is how reconstruction fails",
        f'''<div class="two-col"><div>{svg_bars(arows, C_GBDT, W=820, label_w=420, fmt=lambda v: f"{v:.2f}\u00d7", tick_fmt=lambda v: f"{v:.1f}\u00d7", ref=1.0, lo=0.9, xlabel="errors without the group / errors with all inputs")}</div>
<div><table class="data"><tr><th>control (boosted trees)</th><th>AUC</th></tr>
<tr><td>all {A["n_inputs"]} inputs</td><td class="num">{A["all_inputs"]["auc"]:.4f}</td></tr>
<tr><td>only <b>which columns are NaN</b></td><td class="num">{A["nan_only"]["auc"]:.4f}</td></tr>
<tr><td>old label (decay anywhere)</td><td class="num">{A["old_label"]["auc"]:.4f}</td></tr>
<tr><td>RF phase only (excluded)</td><td class="num">{A["rf"]["only"]:.4f}</td></tr></table>
<div class="callout warn">Knowing only <b>which reconstruction steps failed</b> gives AUC {A["nan_only"]["auc"]:.2f}. That signal is real in MC, but it depends on how reconstruction fails, which may be different on data.</div></div></div>''',
        notes="ablation.json (ablate.py). Diagnostic refits scored on valid.csv; not used for any model choice. The stt group no longer swallows the kink columns."))

    # 16 ------------------------------------------------------------------
    prow = "".join(f"<tr><td>{swatch(C_GBDT if k=='gbdt' else C_MLP)}{k}</td><td class='num'>{num(p['compared'])}</td><td class='num'>{p['mean_abs']:.1e}</td>"
                   f"<td class='num'>{p['max_abs']:.1e}</td><td class='num'>{p['over_tol']}</td><td>{'PASS' if p['pass'] else 'FAIL'}</td></tr>"
                   for k, p in PAR.items())
    erow = "".join(f"<tr><td>{k}</td><td class='num'>{num(e['n'])}</td><td class='num'>{e['mean']:.1e}</td><td class='num'>{e['max']:.1e}</td>"
                   f"<td class='num'>{e['over']}</td><td class='num'>{e['flips']}</td><td>{'PASS' if e['ok'] else 'FAIL'}</td></tr>" for k, e in EXP.items())
    st = AU.get("stale", {})
    S.append(slide("15 &#183; train/serve parity", "The cooker gives the model what it was trained on",
        f'''<p class="caption">cooker <code>ReactionID</code> branch vs Python ONNX on the CSV of the same file, joined on tree entry</p>
<table class="data"><tr><th>model</th><th>events</th><th>mean |&#916;p|</th><th>max |&#916;p|</th><th>|&#916;p| &gt; 10<sup>-4</sup></th><th></th></tr>{prow}</table>
<p class="caption" style="margin-top:14px">ONNX export vs the native model (<code>export_onnx.py --verify</code>)</p>
<table class="data"><tr><th>model</th><th>events</th><th>mean |&#916;p|</th><th>max |&#916;p|</th><th>&gt; 10<sup>-4</sup></th><th>decision flips</th><th></th></tr>{erow}</table>
<div class="callout">{num(st.get("stale",0))} of {num(st.get("tree_entries",0))} tree entries ({pct(st.get("stale",0)/max(st.get("tree_entries",1),1),1)}) are skipped upstream by cryptor blinding. They never reach the CSV and must be dropped from the branch by checking <code>entry</code>. Nobody has yet checked that the skipped events are an unbiased sample.</div>''',
        notes="results/parity_*.json, logs/export.log, audit.json."))

    # 17 ------------------------------------------------------------------
    findings = [
        ("reactionID.py, reactionModel.py", "MLP early-stopped on valid.csv, the file its results were quoted on", "optimistic loss and metrics", "early stopping on a 15% training holdout"),
        ("export_onnx.py", "thresholds were the 90% points measured on valid.csv", "efficiency at threshold was 90% by construction", "thresholds set on the holdout"),
        ("make_report.py", "refitted its own trees instead of loading the shipped ones", "report and shipped model could diverge", "loads model/decay_gbdt.joblib"),
        ("make_report.py", "FPR counted downstream decays as background, and the prior was 0.003", "precision at the prior overstated", "FPR on true non-decays; derived prior"),
        ("ReactionID.cpp", "setDecayThreshold silently overwritten by the model's metadata", "a user threshold had no effect", "explicit setting wins, and is logged"),
        ("Plotting.cpp", "track, arm, bar and pid indices used without range checks", "out-of-bounds reads on flagged vertices", "same guards as ReactionID::process"),
        ("Plotting.cpp", "stale-row marker updated after the PbG early return", "stale rows counted after a PbG veto", "marker updated first"),
        ("Plotting.cpp", "cut-based decision never scored against truth", "no like-for-like benchmark", "cut TP/FN/FP/TN on the same vertices"),
        ("beta alignment", "fitted on validation file 2 and scored on file 2", "circular benchmark", "fitted on training file 1"),
        ("ablate.py, reactionPlots.py", "the stt group swallowed kink; the ROC star said PathLength", "misleading diagnostics", "fixed"),
        ("README.md", "57% decay, Z from &#8722;2000, 0.807 threshold, SPS PID of 9.5e8", "stale documentation", "rechecked against the data"),
        ("cook_mc_chain.sh (open)", f"reactionID stage uses real-data TOF offsets on MC: in-time for {AU['rid_flags']['rid_is_intime_tof']['true']} of "
         f"{AU['rid_flags']['rid_is_intime_tof']['defined']:,} vertices, so <code>rid_is_decay</code> = 1 for all",
         "CSV benchmark column meaningless (not a model input)", "<b>not fixed</b> - needs an MC time alignment"),
    ]
    frow = "".join(f"<tr><td class='mono'>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in findings)
    S.append(slide("16 &#183; review findings", "What was wrong, and what changed",
        f'<table class="data" style="font-size:clamp(11px,1.35vmin,16px)"><tr><th>where</th><th>issue</th><th>impact</th><th>fix</th></tr>{frow}</table>',
        notes="Each finding was confirmed against the code or the data, not taken from comments."))

    # 18 ------------------------------------------------------------------
    def old_at_thr(k):
        old = R0[k]
        thr = R0["truth"]["threshold"] if k == "gbdt" else [w for w in old["working_points"] if abs(w["target_eff"] - 0.9) < 1e-9][0]["threshold"]
        return thr, [w for w in old["working_points"] if abs(w["target_eff"] - 0.9) < 1e-9][0]
    brow = ""
    for k, lab, c in MODELS:
        thr0, w0 = old_at_thr(k)
        new = R[k]
        brow += (f"<tr><td>{swatch(c)}{lab}</td><td class='num'>{R0[k]['auc']:.4f} &#8594; {new['auc']:.4f}</td>"
                 f"<td class='num'>{thr0:.3f} &#8594; {new['threshold']:.3f}</td>"
                 f"<td class='num'>{pct(w0['eff'],2)} &#8594; {pct(new['at_threshold']['efficiency'],2)}</td>"
                 f"<td class='num'>{pct(w0['precision_at_prior'],1)} &#8594; {pct(new['at_threshold']['precision_at_prior'],1)}</td></tr>")
    S.append(slide("17 &#183; before / after", "Honest numbers are slightly lower, and that is the point",
        f'''<table class="data"><tr><th>model</th><th>AUC (valid, all label-0)</th><th>threshold</th><th>efficiency at threshold</th><th>precision at prior</th></tr>{brow}</table>
<div class="callout">Before, the efficiency at threshold was 90% <b>by construction</b>, because the threshold was chosen on the same file. Precision was quoted at a prior of {R0["prior"]:g}, with downstream decays counted as background. After, the prior is {pr["physical"]:.4f} and the FPR counts true non-decays only.</div>''',
        notes="before_fix/report.json vs report.json. AUC in the first column is the like-for-like metric (all label-0 events)."))

    # 19 ------------------------------------------------------------------
    S.append(slide("18 &#183; before real data", "What this MC result does not yet show",
        f'''<ul class="checklist">
<li><span><b>Single setting.</b> One momentum ({pr["momentum_MeV"]:.0f} MeV/c), pure &#956;&#8314;, one target. The BH PID and momentum inputs are constant and were dropped. A real beam mixes e, &#956; and &#960;.</span></li>
<li><span><b>Absolute timing.</b> The top inputs are raw scintillator times. Offsets that are right in MC will be wrong on data unless they are recalibrated or replaced with relative times.</span></li>
<li><span><b>Reconstruction-failure signal.</b> AUC {A["nan_only"]["auc"]:.2f} comes from the missing-value pattern alone, which depends on how reconstruction fails in MC.</span></li>
<li><span><b>Selection.</b> The MC-only chv veto removes classes at different rates, and 7% of events are skipped by blinding upstream.</span></li>
<li><span><b>Label geometry.</b> The "target" part of the label is a 200 mm box around the LH2 cell (vertex cut R &lt; {LOG.get("lh2_radius", float("nan")):g} mm), and the misses pile up at its edges. Tighten it to the cell before trusting per-region efficiencies.</span></li>
<li><span><b>Not re-measured on data.</b> Real-data score distributions for these models were not part of this review.</span></li>
</ul>''', notes="Risks that need physics decisions rather than code fixes."))

    # 20 ------------------------------------------------------------------
    S.append(slide("19 &#183; next steps", "Suggested order",
        '''<ul class="points">
<li><span>Score run 24444 (and a pion-rich run) with the new models. Compare score distributions to MC, per BH PID.</span></li>
<li><span>Replace absolute wall times with times relative to the BH/RF reference, and retrain.</span></li>
<li><span>Generate mixed-species, multi-momentum MC, and put momentum and PID back as inputs.</span></li>
<li><span>Decide whether the chv veto belongs in the selection. If it does, emulate it on data. If not, train without it.</span></li>
<li><span>Check the blinded (skipped) events against the unblinded ones for bias.</span></li>
<li><span>A two-segment STT fit (a kink hypothesis) as a physics input, instead of relying on the missing-value pattern.</span></li>
</ul>'''))
    return S


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="../muse/doc/slides/reaction_id_review.html")
    p.add_argument("--plots-prefix", default="cooker_plots/valid_file2")
    p.add_argument("--calib-prefix", default="cooker_plots/calib_file1")
    args = p.parse_args()

    R = load_json("report.json")
    R0 = load_json("before_fix/report.json")
    A = load_json("ablation.json")
    AU = load_json("audit.json")
    PAR = {k: load_json(f"results/parity_{k}.json") for k in ("gbdt", "mlp")}
    EXP = parse_export_log("logs/export.log")
    LOG = parse_cooker_log("logs/plots_valid_file2.log")
    CAL = parse_cooker_log("logs/plots_calib_file1.log")
    arch = mlp_architecture("model/output.pth")
    names = ("truth_confusion", "truth_agreement_by_region", "truth_decay_position", "truth_score", "beta_full_reco", "truth_beta_calibration")
    plots = {n: f"{args.plots_prefix}_{n}.png" for n in names}
    calplots = {n: f"{args.calib_prefix}_{n}.png" for n in names}

    slides = build(R, R0, A, AU, PAR, EXP, LOG, CAL, arch, plots, calplots)
    html = (ms.TEMPLATE.replace("</style>", CHART_CSS + "</style>", 1)
            .replace("__SLIDES__", "\n".join(slides)).replace("__TOTAL__", str(len(slides)))
            .replace("<title>reactionID</title>", "<title>ReactionID review</title>")
            .replace("muon decay-in-flight from the cooker", "review &#183; fixes &#183; results"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}  ({len(slides)} slides, {os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
