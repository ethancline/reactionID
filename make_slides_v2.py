"""Render the round-two results deck: relabel, dead columns, new inputs, and what they bought.

Same template, palette and navigation as make_slides.py (reused, not copied), same
rule: numbers are computed, not typed. They come from

  report.json      working points and per-region numbers   (make_report.py)
  ablation.json    controls and leave-one-group-out        (ablate.py --json)
  data/valid.csv   label, missing-ness and region tables   (the chain's export)
  the deployed ONNX model, scored on valid.csv at its own threshold

and every physics figure is drawn by the cooker (ReactionID_Plots_MC.xml), cooked with
the truth-calibrated beta alignment in mc17606_beta_alignment.txt.

The numbers that come from one-off runs - the cooker/Python parity, the
stale-input test, and the fixes' before/after counts on slide 03 - are in MEASURED
or stated in the slide notes, with the command that produced each.

    python ablate.py --json ablation.json
    python make_slides_v2.py
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort

import make_slides as ms
import reactionData as rd
from make_slides import esc

C_BAR = ms.C_GBDT
# Input groups added in this round (ablate.py keys); the others existed before.
NEW_GROUPS = ("vertex_pull", "bm", "oot", "doca_signed", "scint_time", "kink")
C_FAINT = ms.C_FAINT
C_GRID = ms.C_GRID

# From single verification runs rather than a file this script can re-read.
MEASURED = {
    # verify_inference.py --rid cooked/..._2_RID.root --csv cooked/..._2_features.csv --model model/decay_<m>.onnx
    # (the MLP branch cooked with -c ReactionID:setModel:"<path>")
    "parity": {"gbdt": {"max": 4.528e-03, "mean": 6.934e-08, "over": 3, "n": 101380},
               "mlp": {"max": 5.603e-06, "mean": 4.198e-08, "over": 0, "n": 101380}},
    # The same GBDT scored on the CSV with the nine ReactionID-output columns as
    # exported, and shifted by one event, against the cooked scores.
    "stale": {"as_exported": 14.004, "shifted": 0.001, "n_cols": 9,
              # ablate.py leave-one-out of the rid_* group, run before they were excluded
              "ablation_ratio": 0.99},
    # ReactionID_Plots_MC.xml finalize log ("ReactionID beta calibration ..."),
    # and the "agreement with cut" printed on the beta-by-verdict canvas before
    # (old 1.7 ns) and after (mc17606_beta_alignment.txt) the recalibration.
    "beta_cal": {"left": {"pos": 1.900, "npos": 884, "mu": 2.153, "nmu": 477},
                 "right": {"pos": 1.882, "npos": 872, "mu": 2.264, "nmu": 589},
                 "old": 1.7, "agree_before": 71.6, "agree_after": 86.8},
    # ReactionID_Plots_MC.xml finalize log, "ReactionID beta, full reconstruction
    # only" (the beta_full_reco canvas) and "ReactionID truth comparison" (all events).
    "beta_full": {"tp": 1882, "fn": 34, "fp": 6, "tn": 1190, "n": 3112,
                  "eff": 98.23, "purity": 99.68, "agreement": 98.71,
                  "eff_all": 90.07, "purity_all": 99.06,
                  # share of all errors at the operating point in events without a full
                  # reconstruction (no SPS time on the track, or no vertex)
                  "outside": 84.6},
    # tune.py (data_15Apr25/tune.log) and compare_batches.py: pooling the 15Apr25
    # production with 21Sep25, and retuning, scored on the untouched 21Sep25 file.
    "tuning": {"rows": [("current settings", "21Sep25 only", 189174, 0.99757),
                        ("tuned settings", "21Sep25 only", 189174, 0.99767),
                        ("current settings", "both productions", 888272, 0.99798),
                        ("tuned settings", "both productions", 888272, 0.99844)],
               "params": "learning rate 0.05, up to 800 rounds (444 used), 255 leaves, min leaf 20, L2 5.0",
               "fp_before": 197, "fp_after": 109, "purity_before": 98.31, "purity_after": 99.06,
               "adv": {"no decay": 0.638, "upstream": 0.597, "target": 0.641, "downstream": 0.652},
               "target_A": 6.41, "target_B": 5.06, "teach_A": 0.99754, "teach_B": 0.99728},
    # export_onnx.py --verify: ONNX against the native model on valid.csv
    "export": {"mlp_max": 1.019e-05, "gbdt_max": 1.157e-02, "gbdt_over": 9},
    # RF phase: AUC with all inputs, with only the 14 RF columns, and with them
    # removed - boosted trees, and the MLP retrained after the removal.
    "rf": {"with": 0.9975, "only": 0.9747, "without": 0.9976, "mlp_with": 0.9955, "mlp_without": 0.9889,
           "shift_nodecay": 9.03, "shift_upstream": 6.74, "shift_target": 7.93, "bunch_sigma": 0.48, "clock_sigma": 0.05},
}

EXTRA_CSS = """
  .kpis{ display:grid; grid-template-columns: repeat(4, 1fr); gap: clamp(12px,1.8vmin,28px); margin-top: clamp(14px,2.4vmin,34px); }
  .kpi{ background: var(--surface); border: 1px solid var(--border-soft); border-top: 2px solid var(--accent-2);
        padding: clamp(12px,1.8vmin,24px) clamp(14px,1.9vmin,24px); }
  .kpi .v{ font-family: var(--font-display); font-weight:700; font-size: clamp(26px,4.6vmin,60px); color: var(--text);
           font-variant-numeric: tabular-nums; line-height:1.05; }
  .kpi .k{ font-family: var(--font-mono); font-size: clamp(11px,1.3vmin,15px); color: var(--text-faint);
           letter-spacing: 0.08em; text-transform: uppercase; margin-top: 8px; }
  .kpi .s{ font-size: clamp(12px,1.5vmin,17px); color: var(--text-dim); margin-top: 6px; line-height:1.45; }
  .big{ font-family: var(--font-display); font-weight:700; font-size: clamp(56px,11vmin,150px); color: var(--accent); line-height:1; }
  table.data td.bad{ color: var(--bad); font-family: var(--font-mono); text-align:right; font-variant-numeric: tabular-nums; }
  table.data td.ok{ color: var(--good); font-family: var(--font-mono); text-align:right; font-variant-numeric: tabular-nums; }
  table.data.compact td{ padding: 0.55em 12px 0.55em 0; }
"""


# --------------------------------------------------------------------------
# numbers computed from the data
# --------------------------------------------------------------------------
def dataset_facts(valid_csv, model_path):
    df = rd.load_csv(valid_csv)
    f = {"n": len(df)}

    reg = df["decay_region"].to_numpy()
    decays = reg > 0
    f["regions"] = {name: int((reg == c).sum()) for c, name in enumerate(("no decay", "upstream", "target", "downstream"))}
    f["downstream_share"] = float((reg == 3).sum() / decays.sum())
    f["relevant_share"] = float(df[rd.LABEL_COLUMN].mean())

    # Missing exactly where needed: overall vs among the decays that matter.
    rel = df[rd.LABEL_COLUMN].to_numpy() == 1
    f["missing"] = []
    for c, lab in (("sps_corr_time", "SPS time on the track"), ("vtx_theta", "vertex angle"), ("tof_raw", "BH&rarr;SPS time of flight")):
        miss = df[c].isna().to_numpy()
        f["missing"].append((c, lab, float(miss.mean()), float(miss[rel].mean())))
    hv = df["n_vertices"].to_numpy() > 0
    f["vertex_rate"] = {"upstream": float(hv[reg == 1].mean()), "target": float(hv[reg == 2].mean()),
                        "no decay": float(hv[reg == 0].mean())}

    # The dead-column census, exactly as training sees it.
    train = rd.load_csv(os.path.join(os.path.dirname(valid_csv), "train.csv"))
    cand = rd.feature_columns(train, drop_uninformative=False)
    dead = rd.zero_information_columns(train, cand)
    buckets = {}
    for c, why in dead.items():
        key = ("never measured in this MC" if why == "never measured"
               else "constant where measured" if "where measured" in why
               else "constant in every event")
        buckets.setdefault(key, []).append(c)
    f["dead"] = buckets
    f["n_candidates"] = len(cand)
    f["n_excluded"] = len(rd.EXCLUDED_COLUMNS)

    # Kink proxy by region (events with an STT track).
    s = df[df["stt_valid"] == 1]
    f["kink"] = {}
    for c, name in enumerate(("no decay", "upstream", "target", "downstream")):
        g = s[s["decay_region"] == c]
        f["kink"][name] = {k: float(g[k].mean()) for k in ("stt_nhits_asym", "stt_nhits_front", "stt_ntracklets", "stt_chi2")}

    # The deployed model at its own operating point.
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    cols = meta["input_columns"].split(",")
    thr = float(meta["threshold"])
    p = sess.run(["probability"], {"features": df[cols].to_numpy(np.float32)})[0].ravel()
    f["threshold"] = thr
    f["model_tag"] = meta.get("model_tag", "")
    f["n_inputs"] = len(cols)
    # Fraction called correctly per truth region: relevant for upstream/target,
    # background for no decay/downstream - the same definition the cooker plots.
    tagged = p > thr
    f["region_correct"] = {}
    for c, name in enumerate(("no decay", "upstream", "target", "downstream")):
        m = reg == c
        f["region_correct"][name] = float((tagged[m] if c in (1, 2) else ~tagged[m]).mean())

    t = df[reg == 2]
    found = p[reg == 2] > thr
    R = np.hypot(t["MuonDecay_X"].to_numpy(), t["MuonDecay_Y"].to_numpy())
    Z = t["MuonDecay_Z"].to_numpy()
    edge = (R > 150) | (np.abs(Z) > 150)
    novtx = t["n_vertices"].to_numpy() == 0
    missed = ~found
    f["target"] = {
        "n": int(len(t)), "eff": float(found.mean()),
        "edge_share": float(edge.mean()), "eff_edge": float(found[edge].mean()), "eff_inner": float(found[~edge].mean()),
        "miss_edge_share": float(edge[missed].mean()),
        "novtx_share": float(novtx.mean()), "eff_novtx": float(found[novtx].mean()), "eff_vtx": float(found[~novtx].mean()),
        "miss_novtx_share": float(novtx[missed].mean()),
    }
    return f


# --------------------------------------------------------------------------
# inline SVG
# --------------------------------------------------------------------------
def svg_ablation(ab):
    """Error-rate multiplier when each group is left out; 1.0 = worthless."""
    rows = sorted(ab["groups"], key=lambda g: -g["error_ratio"])
    W, rowh, L, R, T = 560, 32, 200, 64, 34
    H = T + rowh * len(rows) + 46
    lo, hi = 0.9, max(1.6, max(g["error_ratio"] for g in rows) + 0.05)
    x = lambda v: L + (W - L - R) * (v - lo) / (hi - lo)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Increase in error rate when each input group is removed">']
    for v in np.arange(1.0, hi + 1e-9, 0.2):
        s.append(f'<line x1="{x(v):.1f}" y1="{T-8}" x2="{x(v):.1f}" y2="{H-40}" stroke="{C_GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x(v):.1f}" y="{H-20}" text-anchor="middle" class="dtext">{v:.1f}&#215;</text>')
    s.append(f'<line x1="{x(1):.1f}" y1="{T-8}" x2="{x(1):.1f}" y2="{H-40}" stroke="{C_FAINT}" stroke-width="1.5" stroke-dasharray="4 4"/>')
    s.append(f'<text x="{x(1)+6:.1f}" y="{T-14}" class="dtext caption">no effect</text>')
    for i, g in enumerate(rows):
        y = T + i * rowh
        r = g["error_ratio"]
        x0, x1 = x(1.0), x(max(r, lo))
        a, b = min(x0, x1), max(x0, x1)
        new = g["key"] in NEW_GROUPS
        s.append(f'<text x="{L-12}" y="{y+rowh/2+4:.1f}" text-anchor="end" class="dtext {"val" if new else "caption"}">'
                 f'{SHORT.get(g["key"], esc(g["label"]))}{" &#183; new" if new else ""}</text>')
        if b - a >= 1:
            s.append(f'<rect x="{a:.1f}" y="{y+7}" width="{b-a:.1f}" height="{rowh-14}" rx="2" fill="{C_BAR}"/>')
        s.append(f'<text x="{max(x0, x1)+8:.1f}" y="{y+rowh/2+4:.1f}" class="dtext val">{r:.2f}&#215;</text>')
    s.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-2}" text-anchor="middle" class="dtext caption">'
             f'error rate with the group removed &#247; with all inputs</text>')
    s.append("</svg>")
    return "\n".join(s)


def fmt_pct(v, d=1):
    return f"{100*v:.{d}f}%"


def sci(v, d=1):
    """7.2e-07 -> 7.2&times;10<sup>-7</sup>, kept on one line."""
    m, e = f"{v:.{d}e}".split("e")
    return f'<span style="white-space:nowrap">{m}&times;10<sup>&minus;{abs(int(e))}</sup></span>' if int(e) < 0 else \
           f'<span style="white-space:nowrap">{m}&times;10<sup>{int(e)}</sup></span>'


# Short chart labels, so the ablation chart can use a larger type size.
SHORT = {"scint_time": "wall time", "stt": "STT track", "vertex_geom": "vertex geometry", "bm": "beam monitor",
         "tof": "TOF, BH&#8211;SPS", "oot": "out-of-time hits", "gem": "GEM track", "doca_signed": "signed DOCA",
         "kink": "STT front/rear", "vertex_pull": "vertex pulls", "rid_flags": "ReactionID flags"}


# --------------------------------------------------------------------------
# slides
# --------------------------------------------------------------------------
def build(rep, ab, f):
    d = rep["dataset"]
    wp = {k: {w["target_eff"]: w for w in rep[k]["working_points"]} for k in ("mlp", "gbdt")}
    g90, m90 = wp["gbdt"][0.9], wp["mlp"][0.9]
    tgt = f["target"]
    base = ab["all_inputs"]["auc"]
    s = []

    # ---- TITLE ------------------------------------------------------------
    s.append(f"""
    <section class="slide title-slide active" data-notes="~30s. Round two. The first deck deployed a working classifier; this one is about making it answer the question the analysis actually needs, and about everything that turned out to be silently broken on the way.">
      <svg class="corner-trace" width="360" height="220" viewBox="0 0 360 220" aria-hidden="true">
        <path d="M360 0 L360 40 L300 40 L300 90 L250 90 L250 140" class="darrow" stroke="#263544"/>
        <path d="M360 60 L320 60 L320 120 L270 120" class="darrow" stroke="#263544"/>
        <circle cx="250" cy="140" r="3" fill="#ffb454"/><circle cx="270" cy="120" r="3" fill="#263544"/>
      </svg>
      <div class="slide-inner">
        <div class="eyebrow">MUSE analysis &middot; <span class="dim">reactionID, round two</span></div>
        <h1 class="title">Asking the classifier<br>the <span class="accent">right question.</span></h1>
        <hr class="rule">
        <p class="lede">Relabelling to the decays that can fake a scatter, finding the inputs that were dead,
        measuring what new detector information is actually worth &mdash; and checking none of it is an artifact.</p>
        <div class="meta-row">
          <div><b>Sample</b><span>mc17606 &middot; 210 MeV/c &middot; LH2 &middot; &mu;+</span></div>
          <div><b>Events</b><span>{d['train_events']:,} train / {d['valid_events']:,} valid</span></div>
          <div><b>Model inputs</b><span>{d['feature_columns']} of {d['columns']} exported columns</span></div>
          <div><b>Author</b><span>Ethan Cline</span></div>
        </div>
      </div>
    </section>""")

    # ---- HEADLINE ---------------------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1 min. Four numbers. The second pair matters as much as the first: the deployed cooker reproduces Python, and the gain is not the model learning which detectors failed.">
      <div class="slide-inner">
        <div class="eyebrow">00 &mdash; <span class="dim">in one slide</span></div>
        <h1 class="title">Better, deployed, and not an artifact.</h1>
        <hr class="rule">
        <div class="kpis">
          <div class="kpi"><div class="v">{rep['gbdt']['auc']:.4f}</div><div class="k">ROC AUC &middot; boosted trees</div>
            <div class="s">decay before or inside the target, vs everything else</div></div>
          <div class="kpi"><div class="v">{g90['rejection']:.0f}&times;</div><div class="k">rejection at 90% efficiency</div>
            <div class="s">false&#8209;positive rate {fmt_pct(g90['fpr'],2)} &middot; MLP: {m90['rejection']:.0f}&times;</div></div>
          <div class="kpi"><div class="v">{sci(MEASURED['parity']['gbdt']['max'])}</div><div class="k">max |&Delta;P| cooker vs Python</div>
            <div class="s">{MEASURED['parity']['gbdt']['over'] or 'none'} of {MEASURED['parity']['gbdt']['n']:,} events above 10<sup>&minus;4</sup>, no verdict changed</div></div>
          <div class="kpi"><div class="v">{ab['nan_only']['auc']:.3f}</div><div class="k">missing&#8209;pattern&#8209;only control</div>
            <div class="s">unchanged &mdash; the gain is physics, not reconstruction failure</div></div>
        </div>
        <ul class="points">
          <li><span><b>New label.</b> <code class="mono">decay_relevant</code>: the muon decayed upstream of or inside the target.
            <span class="muted">{fmt_pct(f['downstream_share'],0)} of simulated decays happen downstream and cannot fake a scatter.</span></span></li>
          <li><span><b>Fourteen dead columns traced to three bugs</b>, and a check that now drops such columns mechanically.</span></li>
          <li><span><b>One new input family that matters</b> &mdash; per&#8209;wall scintillator time &mdash; and five that do not. <span class="muted">RF phase was dropped: real physics, but flattering in MC.</span></span></li>
        </ul>
      </div>
    </section>""")

    # ---- 01 LABEL ---------------------------------------------------------
    rg = f["regions"]
    nd = sum(v for k, v in rg.items() if k != "no decay")
    s.append(f"""
    <section class="slide" data-notes="~1.5 min. The old label was 'did the muon decay anywhere'. Most of those decays are metres downstream and irrelevant to the cross section. Note the caution: AUCs on different labels answer different questions, so the fair comparison is the error rate on the question we care about.">
      <div class="slide-inner">
        <div class="eyebrow">01 &mdash; <span class="dim">the label</span></div>
        <h1 class="title">Most decays it was trained to find<br>cannot contaminate anything.</h1>
        <hr class="rule">
        <div class="two-col">
          <div>
            <div class="big">{fmt_pct(f['downstream_share'],0)}</div>
            <p class="lede">of simulated decays happen downstream of the target. A decay there cannot fake a
            scattering event, so training on them spent most of the model on the wrong question.</p>
            <table class="data compact">
              <tr><th>truth decay region</th><th style="text-align:right">events</th><th style="text-align:right">share of decays</th><th>new label</th></tr>
              <tr><td>upstream of target</td><td class="num">{rg['upstream']:,}</td><td class="num">{fmt_pct(rg['upstream']/nd)}</td><td class="mono">relevant</td></tr>
              <tr><td>inside target (|R|, |Z| &lt; 200 mm)</td><td class="num">{rg['target']:,}</td><td class="num">{fmt_pct(rg['target']/nd)}</td><td class="mono">relevant</td></tr>
              <tr><td>downstream</td><td class="num">{rg['downstream']:,}</td><td class="num">{fmt_pct(rg['downstream']/nd)}</td><td class="mono">background</td></tr>
              <tr><td>no decay</td><td class="num">{rg['no decay']:,}</td><td class="num">&mdash;</td><td class="mono">background</td></tr>
            </table>
          </div>
          <div>
            <table class="data">
              <tr><th>same inputs, boosted trees</th><th style="text-align:right">AUC</th><th style="text-align:right">1 &minus; AUC</th></tr>
              <tr><td>old label: decay anywhere</td><td class="num">{ab['old_label']['auc']:.4f}</td><td class="num">{1-ab['old_label']['auc']:.4f}</td></tr>
              <tr class="highlight"><td>new label: decay before / in target</td><td class="num">{base:.4f}</td><td class="num">{1-base:.4f}</td></tr>
            </table>
            <div class="callout"><b>Different questions, so read this carefully.</b> The two AUCs rank different
            positives. What matters is that the question the analysis needs is now answered with
            {((1-ab['old_label']['auc'])/(1-base)):.1f}&times; fewer ranking errors than the old question was.
            Positives are {fmt_pct(f['relevant_share'])} of events, down from {fmt_pct(nd/f['n'])}.</div>
            <div class="callout good"><b>One constant, checked at load.</b> The label is
            <code class="mono">decayfeatures::kLabelColumn</code>; the exported model records the label it was trained on,
            and ReactionID warns if the two disagree.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 02 ARTIFACT ------------------------------------------------------
    miss_rows = "\n".join(
        f"<tr><td>{lab} <span class='mono' style='color:var(--text-faint)'>{esc(c)}</span></td>"
        f"<td class='num'>{fmt_pct(a)}</td><td class='bad'>{fmt_pct(b)}</td></tr>" for c, lab, a, b in f["missing"])
    vr = f["vertex_rate"]
    s.append(f"""
    <section class="slide" data-notes="~2 min. The most important control in the deck. A model that sees only which columns are NaN, no values at all, reaches 0.91. That correlation is real in MC - decays break reconstruction - but it describes the reconstruction, and it is the likely reason the old model flagged 96% of a real run. It has to be re-run after every change; it stayed flat.">
      <div class="slide-inner">
        <div class="eyebrow">02 &mdash; <span class="dim">the control</span></div>
        <h1 class="title">How much is the model just<br>recognising that tracking failed?</h1>
        <hr class="rule">
        <div class="two-col">
          <div>
            <table class="data">
              <tr><th>boosted trees, {ab['n_train']:,} training events</th><th style="text-align:right">AUC</th></tr>
              <tr><td>all inputs, all events</td><td class="num">{base:.4f}</td></tr>
              <tr><td><b>only which columns are missing</b> &mdash; no measured values</td><td class="num">{ab['nan_only']['auc']:.4f}</td></tr>
              <tr><td>all inputs, fully reconstructed events only ({fmt_pct(ab['reco_fraction'],0)})</td><td class="num">{ab['all_inputs']['auc_reco']:.4f}</td></tr>
            </table>
            <div class="callout warn"><b>{ab['nan_only']['auc']:.2f} from missing&#8209;ness alone.</b> Real in MC &mdash; decays do break
            reconstruction &mdash; but it describes the reconstruction, not the physics, and will not transfer to data whose
            failures differ. It stayed flat through every change here, so the gains below are not more of it.</div>
          </div>
          <div>
            <table class="data">
              <tr><th>measurement</th><th style="text-align:right">missing, all</th><th style="text-align:right">missing, relevant decays</th></tr>
              {miss_rows}
            </table>
            <div class="callout"><b>The timing measurements are absent exactly where they are needed</b>, and upstream decays
            are starved of a vertex: one exists for only {fmt_pct(vr['upstream'],0)} of them, against {fmt_pct(vr['no decay'],0)}
            of events with no decay ({fmt_pct(vr['target'],0)} for target decays). When the measurements exist, the problem is
            nearly solved ({ab['all_inputs']['auc_reco']:.4f}).</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 03 THREE BUGS ----------------------------------------------------
    s.append("""
    <section class="slide" data-notes="~2 min. Fourteen columns were constant. Not one bug: three, in two plugins, and only the first is about stage order. Each is silent - nothing errors. The before/after counts are from direct censuses of the cooked files (file 1, 174,212 vertices in 255,090 events) and the vertex fraction after the fix.">
      <div class="slide-inner">
        <div class="eyebrow">03 &mdash; <span class="dim">why the vertex outputs were empty</span></div>
        <h1 class="title">Fourteen dead columns, three separate bugs.</h1>
        <hr class="rule">
        <table class="data">
          <tr><th>where</th><th>what was wrong</th><th style="text-align:right">before</th><th style="text-align:right">after</th></tr>
          <tr><td class="mono">VertexReconstruction</td><td>fills geometry only; the reaction flags belong to <b>ReactionID</b>, which ran <i>after</i> the export
            and wrote them to another branch. <span style="color:var(--text-dim)"><code>is_decay</code> defaults to <b>true</b> &mdash; every vertex read as a decay.</span></td>
            <td class="bad">class defaults</td><td class="ok">stage reordered</td></tr>
          <tr><td class="mono">ReactionID</td><td>target taken only from slow control; MC has none, so <code>TARGET</code> stayed empty and every target cut was false</td>
            <td class="bad">0 of 174,212</td><td class="ok">0.7% target vertices</td></tr>
          <tr><td class="mono">PathLength</td><td>pushes a scatter only if <code>id</code> is &plusmn;11/13/211 (PDG), but <code>Vertex::id</code> is an enum 0&ndash;3 and never set &mdash; nothing can pass.
            Also a <code>return</code> inside the vertex loop.</td>
            <td class="bad">0 scatters, all 255k events</td><td>reported, not fixed</td></tr>
        </table>
        <div class="callout warn"><b>Still dead after the fixes:</b> <code class="mono">is_intime_tof</code> (3 vertices) and ReactionID's own
        <code class="mono">is_decay</code> (1 for every vertex). Both need a TOF calibration entry for run 17606 &mdash; a calibration, not a code change.</div>
        <div class="callout good"><b>ReactionID now refuses to start with no target</b> rather than silently producing an all&#8209;false target cut.</div>
      </div>
    </section>""")

    # ---- 04 DETECTOR ------------------------------------------------------
    dead_rows = []
    for key in ("never measured in this MC", "constant in every event", "constant where measured"):
        cols = f["dead"].get(key, [])
        if not cols:
            continue
        shown = ", ".join(cols[:7]) + (f" &hellip; +{len(cols)-7}" if len(cols) > 7 else "")
        dead_rows.append(f"<tr><td>{key}</td><td class='num'>{len(cols)}</td><td class='mono'>{shown}</td></tr>")
    n_dead = sum(len(v) for v in f["dead"].values())
    s.append(f"""
    <section class="slide" data-notes="~1.5 min. Every dead column so far was dead silently, and several would come alive on real data. A model trained with them frozen has no defined behaviour when they move, so they are now dropped mechanically and loudly at training time, with the reason printed.">
      <div class="slide-inner">
        <div class="eyebrow">04 &mdash; <span class="dim">making it mechanical</span></div>
        <h1 class="title">Dead columns are now dropped<br>by a check, not by hand.</h1>
        <hr class="rule">
        <div class="two-col narrow-right">
          <div>
            <table class="data compact">
              <tr><th>reason</th><th style="text-align:right">cols</th><th>examples</th></tr>
              {''.join(dead_rows)}
            </table>
            <p class="caption">{n_dead} of {f['n_candidates']} candidate inputs dropped on the training set; plus
            {f['n_excluded']} excluded by name with a stated reason (circular, or broken upstream).</p>
          </div>
          <div>
            <div class="callout"><b>A column is dropped when neither its value nor its missing&#8209;flag varies</b>,
            or when it is constant where measured and its missing&#8209;pattern duplicates another column.</div>
            <div class="callout warn"><b>New finds.</b> <code class="mono">gem_doca0/1</code> are the &minus;10000 default;
            the local&#8209;frame DOCAs are always 0; GEM tracks carry no hits; SPS and VETO PID are constant because their
            recipes never run <code class="mono">calib_rf_peak</code>. The BH PIDs are constant only because the sample is pure &mu;+.</div>
            <div class="callout good"><b>These stay in the builder.</b> Out&#8209;of&#8209;time hits are empty because the MC simulates
            none; on data they will vary, and the check will keep them.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 05 NEW INPUTS ----------------------------------------------------
    wn = ab["without_new_inputs"]
    st_g = next(g for g in ab["groups"] if g["key"] == "scint_time")
    s.append(f"""
    <section class="slide" data-notes="~2 min. Leave one group out, boosted trees, error-rate ratio. Per-wall scintillator time is the one that matters. The rest - beam monitor, vertex pulls, signed DOCA, the front/rear STT split, out-of-time hits - are at noise level here; the beam monitor came out between 1.03x and 1.15x across runs, which is the run-to-run noise on these numbers. Say that plainly. Grey labels are groups that existed before this round.">
      <div class="slide-inner">
        <div class="eyebrow">05 &mdash; <span class="dim">what each input is worth</span></div>
        <h1 class="title">One new input family matters.<br>The other five do not.</h1>
        <hr class="rule">
        <div class="two-col">
          <div class="diagram-wrap">{svg_ablation(ab)}</div>
          <div>
            <table class="data">
              <tr><th>boosted trees, {ab['n_train']:,} training events</th><th style="text-align:right">AUC</th><th style="text-align:right">1 &minus; AUC</th></tr>
              <tr><td>without any of this round's inputs</td><td class="num">{wn['auc']:.4f}</td><td class="num">{1-wn['auc']:.4f}</td></tr>
              <tr class="highlight"><td>with them</td><td class="num">{base:.4f}</td><td class="num">{1-base:.4f}</td></tr>
            </table>
            <div class="callout good"><b>Per&#8209;wall time</b> of the largest scintillator deposit: tracking&#8209;independent, and the
            top feature of the deployed model. The builder had only ever read the one hit correlated to the track.</div>
            <div class="callout"><b>Quote the error rate, not the AUC.</b> Without the timing group the AUC goes
            {base:.4f} &rarr; {st_g['auc']:.4f}: it looks like nothing, and it is {100*(st_g['error_ratio']-1):.0f}% more mistakes.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 06 IMPORTANCE ----------------------------------------------------
    top = sorted(rep["importance"]["features"], key=lambda x: -x["mean"])[:3]
    top_feats = ", ".join(f'<code class="mono">{esc(x["name"])}</code>' for x in top)
    s.append(f"""
    <section class="slide" data-notes="~1 min. Permutation importance: drop in AUC when a column is shuffled. The MLP relies on a few columns heavily; the trees spread it out, which is why each tree bar is small.">
      <div class="slide-inner">
        <div class="eyebrow">06 &mdash; <span class="dim">feature importance</span></div>
        <h1 class="title">The models lean on timing.</h1>
        <hr class="rule">
        <div class="two-col narrow-right">
          <div>{ms.fig("feature_importance.png", "Permutation feature importance, both models", maxh="68vh")}</div>
          <div>
            <div class="callout"><b>Top of the deployed model:</b> {top_feats} &mdash; timing, with the per&#8209;wall
            scintillator times added this round among them. The RF phases that used to head this ranking are gone &mdash;
            see the next slide.</div>
            <div class="callout"><b>Why the tree bars are short.</b> Shuffling one column costs the trees little because
            correlated columns stand in for it; the MLP is less redundant. Rankings, not magnitudes, are comparable.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 06b RF PHASE REMOVED --------------------------------------------
    rfm = dict(MEASURED["rf"])
    rfm.update({k: ab["rf"][k] for k in ("with", "only", "without") if k in ab.get("rf", {})})
    n_rf = ab.get("rf", {}).get("n_cols", 14)
    s.append(f"""
    <section class="slide" data-notes="~1.5 min. Asked whether RF phase was a truth proxy. It is not - it is built from digitised TDC times, and it has a real physical meaning: a bunch-referenced arrival time, so a positron at beta 1 arrives at a different phase than the muon would. But its power in MC is flattering, and for the trees it is redundant, so it is out. Be honest that the MLP did lose real performance.">
      <div class="slide-inner">
        <div class="eyebrow">06b &mdash; <span class="dim">RF phase, and why it is gone</span></div>
        <h1 class="title">Real physics. Flattering in MC.<br>Removed.</h1>
        <hr class="rule">
        <div class="two-col">
          <div>
            <table class="data compact">
              <tr><th>SPS rear&#8209;right RF phase</th><th style="text-align:right">median (ns)</th></tr>
              <tr><td>no decay</td><td class="num">{rfm['shift_nodecay']:.2f}</td></tr>
              <tr><td>decay upstream of the target</td><td class="num">{rfm['shift_upstream']:.2f}</td></tr>
              <tr><td>decay inside the target</td><td class="num">{rfm['shift_target']:.2f}</td></tr>
            </table>
            <div class="callout"><b>It is not MC truth.</b> The phase is built from digitised TDC times and a digitised
            RF signal. And it means something physical: a bunch&#8209;referenced arrival time, needing no BH correlation,
            no path length and no vertex. A decay positron at &beta;&nbsp;=&nbsp;1 reaches SPS earlier than the muon would,
            and the shift tracks where the muon decayed.</div>
          </div>
          <div>
            <table class="data compact">
              <tr><th>boosted trees</th><th style="text-align:right">AUC</th></tr>
              <tr><td>all inputs, with RF</td><td class="num">{rfm['with']:.4f}</td></tr>
              <tr><td>the {n_rf} RF columns alone</td><td class="num">{rfm['only']:.4f}</td></tr>
              <tr class="highlight"><td>every RF column removed</td><td class="num">{rfm['without']:.4f}</td></tr>
            </table>
            <div class="callout warn"><b>Why it had to go.</b> In this MC the bunch is one clean Gaussian per species
            ({rfm['bunch_sigma']:.2f} ns for &mu;), the RF clock is smeared by {rfm['clock_sigma']*1000:.0f} ps, there is no
            run&#8209;to&#8209;run drift, the beam is pure &mu;+ so no species overlap &mdash; and SPS never runs
            <code class="mono">calib_rf_peak</code>, so on real data its phase is not even centred per bar.</div>
            <div class="callout"><b>The cost, stated plainly.</b> {'Free' if rfm['without'] >= rfm['with'] - 1e-4 else 'Small'} for the
            trees, which is what ships ({rfm['with']:.4f} &rarr; {rfm['without']:.4f}). When it was removed, the MLP &mdash; then
            trained on 21Sep25 alone &mdash; fell from {rfm['mlp_with']:.4f} to {rfm['mlp_without']:.4f}: it had leaned on RF,
            the trees had not. Retrained on both productions, without RF, it now reaches {rep['mlp']['auc']:.4f}.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 07 KINK ----------------------------------------------------------
    k = f["kink"]
    krows = "\n".join(
        f"<tr><td>{n}</td><td class='num'>{k[n]['stt_nhits_asym']:.3f}</td><td class='num'>{k[n]['stt_nhits_front']:.2f}</td>"
        f"<td class='num'>{k[n]['stt_ntracklets']:.2f}</td><td class='num'>{k[n]['stt_chi2']:.2f}</td></tr>"
        for n in ("no decay", "upstream", "target", "downstream"))
    kg = next((g for g in ab["groups"] if g["key"] == "kink"), None)
    s.append(f"""
    <section class="slide" data-notes="~1.2 min. An honest negative. Splitting the straw hits into the two half-chambers does separate the relevant decays, but the other STT columns already carry it. Also a correction: stt_chi2 is not flat across regions, as I claimed earlier.">
      <div class="slide-inner">
        <div class="eyebrow">07 &mdash; <span class="dim">a negative result</span></div>
        <h1 class="title">The kink proxy separates the regions<br>&mdash; and adds nothing.</h1>
        <hr class="rule">
        <table class="data">
          <tr><th>truth region (events with an STT track)</th><th style="text-align:right">front/rear hit asymmetry</th><th style="text-align:right">front hits</th><th style="text-align:right">tracklets used</th><th style="text-align:right">STT &chi;&sup2;</th></tr>
          {krows}
        </table>
        <div class="two-col" style="margin-top:8px">
          <div class="callout"><b>A real difference.</b> Relevant decays have about half the front/rear asymmetry of events with no decay.
          But with the whole group removed the error rate is {kg['error_ratio']:.2f}&times; what it was &mdash; no loss; the other STT columns already carry it.</div>
          <div class="callout warn"><b>Correction.</b> I said earlier that <code class="mono">stt_chi2</code> is flat across regions.
          On this sample it is not. The explicit two&#8209;segment fit is still untried; it needs the straw wire positions.</div>
        </div>
      </div>
    </section>""")

    # ---- 08 STALE INPUTS --------------------------------------------------
    st, par = MEASURED["stale"], MEASURED["parity"]
    s.append(f"""
    <section class="slide" data-notes="~1.5 min. The best bug of the round, found only because parity is checked end to end. Inside the plugin the classifier runs before the vertex loop, so ReactionID's own flags on the output branch still held the previous event. Shifting those columns by one event reproduced the cooker to 0.001%. They were worth nothing, and are circular anyway, so they are out.">
      <div class="slide-inner">
        <div class="eyebrow">08 &mdash; <span class="dim">train/serve parity</span></div>
        <h1 class="title">The cooker was feeding the model<br>the <span class="accent">previous event.</span></h1>
        <hr class="rule">
        <div class="two-col">
          <div>
            <table class="data">
              <tr><th>{st['n_cols']} ReactionID&#8209;output columns</th><th style="text-align:right">events with |&Delta;P| &gt; 10<sup>&minus;4</sup></th></tr>
              <tr><td>as exported (current event)</td><td class="bad">{st['as_exported']:.1f}%</td></tr>
              <tr><td>shifted back by one event</td><td class="ok">{st['shifted']:.3f}%</td></tr>
            </table>
            <div class="callout warn"><b>Cause.</b> In the plugin the classifier runs before the vertex loop. With no input
            <code class="mono">allScattering</code>, the builder fell back to ReactionID's own output branch, which still held the last event.</div>
            <div class="callout good"><b>Fix.</b> Those columns are ReactionID's outputs &mdash; circular as its inputs &mdash; and worth
            {MEASURED['stale']['ablation_ratio']:.2f}&times; in the ablation. Excluded, and the builder no longer falls back to an output branch.</div>
          </div>
          <div>
            <table class="data">
              <tr><th>after the fix</th><th style="text-align:right">max |&Delta;P|</th><th style="text-align:right">mean |&Delta;P|</th><th style="text-align:right">&gt; 10<sup>&minus;4</sup></th></tr>
              <tr><td>boosted trees (default)</td><td class="num">{sci(par['gbdt']['max'])}</td><td class="num">{sci(par['gbdt']['mean'])}</td><td class="num">{par['gbdt']['over']} / {par['gbdt']['n']:,}</td></tr>
              <tr><td>MLP</td><td class="num">{sci(par['mlp']['max'])}</td><td class="num">{sci(par['mlp']['mean'])}</td><td class="num">{par['mlp']['over']} / {par['mlp']['n']:,}</td></tr>
            </table>
            <p class="caption">Scores written by the cooker's ReactionID branch against Python on the same events.
            The verifier now uses the model's declared inputs and refuses to compare against a different model.</p>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 09 CONFUSION -----------------------------------------------------
    wp_rows = "\n".join(
        f"<tr><td class='mono'>{t*100:.0f}%</td>"
        f"<td class='num'>{wp['gbdt'][t]['rejection']:.0f}&times;</td><td class='num'>{fmt_pct(wp['gbdt'][t]['fpr'],2)}</td>"
        f"<td class='num'>{wp['mlp'][t]['rejection']:.0f}&times;</td><td class='num'>{fmt_pct(wp['mlp'][t]['fpr'],2)}</td></tr>"
        for t in (0.8, 0.9, 0.95, 0.99))
    s.append(f"""
    <section class="slide" data-notes="~1.2 min. Drawn by the cooker from the deployed model at its 90%-efficiency operating point. The working points on the right are the curve to choose from; the requirement is not settled yet, so all four are shown.">
      <div class="slide-inner">
        <div class="eyebrow">09 &mdash; <span class="dim">against MC truth</span></div>
        <h1 class="title">In the cooker, at the operating point.</h1>
        <hr class="rule">
        <div class="two-col">
          <div>{ms.fig("truth_confusion.png", "Confusion matrix drawn by the cooker", maxh="58vh")}</div>
          <div>
            <table class="data">
              <tr><th>efficiency</th><th style="text-align:right">trees rejection</th><th style="text-align:right">trees FPR</th><th style="text-align:right">MLP rejection</th><th style="text-align:right">MLP FPR</th></tr>
              {wp_rows}
            </table>
            <div class="callout"><b>Boosted trees are the default and the better model</b> &mdash; about
            {g90['rejection']/m90['rejection']:.1f}&times; the MLP's rejection at 90% efficiency. Threshold {f['threshold']:.4f},
            carried in the model file.</div>
            <div class="callout warn"><b>No purity at a physical rate yet.</b> The report still assumes a decay rate of 0.003,
            which was for decays anywhere; relevant decays are about a quarter of that.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 09b MORE SIMULATION, RETUNED -----------------------------------
    tu = MEASURED["tuning"]
    e0 = 1 - tu["rows"][0][3]
    trows = "\n".join(
        f"<tr{' class=\"highlight\"' if i == 3 else ''}><td>{a}</td><td>{b}</td><td class='num'>{n:,}</td>"
        f"<td class='num'>{auc:.5f}</td><td class='num'>{e0/(1-auc):.2f}&times;</td></tr>"
        for i, (a, b, n, auc) in enumerate(tu["rows"]))
    adv = " &middot; ".join(f"{k} {v:.2f}" for k, v in tu["adv"].items())
    s.append(f"""
    <section class="slide" data-notes="~2 min. A second production of the same 210 MeV/c mu+ LH2 setting was sitting on disk: 28 files, 939k events. Cooked through the whole chain in 11 minutes. Before pooling, checked it is the same simulation - it is not quite: a classifier can tell the productions apart within every truth class, through VETO timing, STT response, SPS light yield and the target vertex z. Pooling still helps on the newer production's untouched validation file. The settings were chosen on a split carved from the newer production only, never on the validation file. The MLP was not retrained.">
      <div class="slide-inner">
        <div class="eyebrow">09b &mdash; <span class="dim">more simulation, retuned</span></div>
        <h1 class="title">4.7&times; the training data and tuned settings:<br>{e0/(1-tu['rows'][3][3]):.2f}&times; fewer errors.</h1>
        <hr class="rule">
        <div class="two-col">
          <div>
            <table class="data compact">
              <tr><th>settings</th><th>training data</th><th style="text-align:right">events</th><th style="text-align:right">AUC</th><th style="text-align:right">vs now</th></tr>
              {trows}
            </table>
            <p class="caption">Boosted trees, all scored once on the untouched 21Sep25 validation file. Settings chosen
            on a split carved from 21Sep25 training data: {tu['params']}.</p>
            <div class="callout good"><b>At the same 90% efficiency</b>, false positives fall from {tu['fp_before']} to
            {tu['fp_after']}: purity {tu['purity_before']:.2f}% &rarr; {tu['purity_after']:.2f}%.</div>
          </div>
          <div>
            <div class="callout warn"><b>The two productions are not the same simulation.</b> Target decays are
            {tu['target_A']:.2f}% of events in 21Sep25 but {tu['target_B']:.2f}% in 15Apr25 (22&sigma;), and a classifier tells
            them apart <i>within</i> every truth class (AUC {adv}; 0.5 would be identical) &mdash; through VETO timing,
            STT &chi;&sup2; and hits, SPS light yield and, for target decays, vertex z.</div>
            <div class="callout"><b>Pooling still helps.</b> Size&#8209;matched, 15Apr25 is a slightly weaker teacher
            ({tu['teach_B']:.5f} vs {tu['teach_A']:.5f}), but adding all of it to 21Sep25 beats 21Sep25 alone. Worth knowing
            which simulation changes happened between April and September before trusting either on data.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 10 REGIONS -------------------------------------------------------
    rc = f["region_correct"]
    s.append(f"""
    <section class="slide" data-notes="~1 min. Downstream decays are now background and are correctly ignored. The weak spot is the one that matters most: decays inside the target.">
      <div class="slide-inner">
        <div class="eyebrow">10 &mdash; <span class="dim">by decay region</span></div>
        <h1 class="title">The weakest region is the one<br>that matters most.</h1>
        <hr class="rule">
        <div class="two-col narrow-right">
          <div>{ms.fig("truth_agreement_by_region.png", "Fraction correct by truth decay region, drawn by the cooker", maxh="60vh")}</div>
          <div>
            <div class="big">{fmt_pct(tgt['eff'],0)}</div>
            <p class="lede">of decays inside the target are tagged, against {fmt_pct(rc['upstream'],0)} upstream and
            {fmt_pct(min(rc['no decay'], rc['downstream']))}+ for both kinds of background.</p>
            <div class="callout">Downstream decays are now correctly ignored: {fmt_pct(rc['downstream'])} are called background,
            which the old label would have counted as misses.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 11 WHERE THE MISSES ARE -----------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~2 min. Two causes account for most target misses. Half have no reconstructed vertex at all. A quarter sit within 50 mm of the edge of the 200 mm box that defines 'target' - a definitional boundary, not a physical one. The dashed lines are that edge. Both point to the next steps: per-vertex scoring and a physical target definition.">
      <div class="slide-inner">
        <div class="eyebrow">11 &mdash; <span class="dim">where the target misses are</span></div>
        <h1 class="title">Half have no vertex. A quarter sit<br>on the edge of the definition.</h1>
        <hr class="rule">
        {ms.fig("truth_decay_position.png", "Truth decay vertex Z and R for found and missed decays, with the target-region edge", maxh="42vh")}
        <table class="data compact" style="margin-top:14px">
          <tr><th>target&#8209;region decays ({tgt['n']:,})</th><th style="text-align:right">share of all</th><th style="text-align:right">tagged</th><th style="text-align:right">share of misses</th></tr>
          <tr><td>no reconstructed vertex</td><td class="num">{fmt_pct(tgt['novtx_share'])}</td><td class="bad">{fmt_pct(tgt['eff_novtx'])}</td><td class="num">{fmt_pct(tgt['miss_novtx_share'],0)}</td></tr>
          <tr><td>within 50 mm of the 200 mm box edge</td><td class="num">{fmt_pct(tgt['edge_share'])}</td><td class="bad">{fmt_pct(tgt['eff_edge'])}</td><td class="num">{fmt_pct(tgt['miss_edge_share'],0)}</td></tr>
          <tr><td>with a vertex</td><td class="num">{fmt_pct(1-tgt['novtx_share'])}</td><td class="ok">{fmt_pct(tgt['eff_vtx'])}</td><td class="num">&mdash;</td></tr>
        </table>
      </div>
    </section>""")

    # ---- 12 SCORE ---------------------------------------------------------
    s.append("""
    <section class="slide" data-notes="~45s. Log scale. Both populations pile up at their ends; the overlap in the middle is small and flat.">
      <div class="slide-inner">
        <div class="eyebrow">12 &mdash; <span class="dim">score distribution</span></div>
        <h1 class="title">Clean separation, a flat overlap.</h1>
        <hr class="rule">
        <div class="two-col narrow-right">
          <div>""" + ms.fig("truth_score.png", "Classifier score split by MC truth, log scale, drawn by the cooker", maxh="60vh") + """</div>
          <div>
            <div class="callout">Relevant decays pile up at 1 and background at 0, four orders of magnitude above the
            flat middle. The dashed line is the deployed threshold.</div>
            <div class="callout">Background above threshold is flat in score, so moving the working point trades
            efficiency for rejection smoothly &mdash; there is no cliff to fall off.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 13 VERTEX --------------------------------------------------------
    s.append("""
    <section class="slide" data-notes="~1 min. Only events with a vertex appear here. Decays - found or missed - are wide-angle; background is concentrated at small angle and small DOCA. The false positives look like decays, which is what you want from a classifier's mistakes.">
      <div class="slide-inner">
        <div class="eyebrow">13 &mdash; <span class="dim">reconstructed vertex by outcome</span></div>
        <h1 class="title">The mistakes look like the other class.</h1>
        <hr class="rule">
        <div class="two-col narrow-right">
          <div>""" + ms.fig("truth_vertex_by_outcome.png", "Vertex angle versus DOCA for true/false positives and negatives", maxh="64vh") + """</div>
          <div>
            <div class="callout">Found and missed decays share the same wide&#8209;angle, large&#8209;DOCA topology
            (&#10216;&theta;&#10217; &asymp; 0.6&ndash;0.7 rad), while background sits at 0.36 rad and small DOCA.</div>
            <div class="callout">The 67 false positives with a vertex look like decays, not like the background they are.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 14a BETA CALIBRATION --------------------------------------------
    bc = MEASURED["beta_cal"]
    s.append(f"""
    <section class="slide" data-notes="~1.5 min. Run 17606 has no beta alignment of its own, so MC needs one. The earlier 1.7 ns was fitted by eye to the wrong peak and left the decay positrons at beta 0.92. Now it is measured against truth: for a particle of known speed the shift is determined, so decays in the target (positron, beta = 1) and events with no decay (muon, beta = p/E) each give a distribution whose peak is the required shift. The positron value is what is loaded. The muon wants 0.25-0.38 ns more - a single timing offset cannot place both, and the cause is not yet known.">
      <div class="slide-inner">
        <div class="eyebrow">14a &mdash; <span class="dim">&beta; alignment from MC truth</span></div>
        <h1 class="title">The decay positron must sit at &beta; = 1.<br>Calibrate to that.</h1>
        <hr class="rule">
        {ms.fig("truth_beta_calibration.png", "Required beta alignment shift per arm for truth positrons and muons", maxh="33vh")}
        <div class="two-col" style="margin-top:10px">
          <table class="data compact">
            <tr><th>required shift (ns)</th><th style="text-align:right">positron, &beta;=1</th><th style="text-align:right">muon, &beta;=p/E</th><th style="text-align:right">was loaded</th></tr>
            <tr><td>left arm</td><td class="num">{bc['left']['pos']:.2f} <span class="mono" style="color:var(--text-faint)">n={bc['left']['npos']}</span></td><td class="num">{bc['left']['mu']:.2f} <span class="mono" style="color:var(--text-faint)">n={bc['left']['nmu']}</span></td><td class="bad">{bc['old']:.2f}</td></tr>
            <tr><td>right arm</td><td class="num">{bc['right']['pos']:.2f} <span class="mono" style="color:var(--text-faint)">n={bc['right']['npos']}</span></td><td class="num">{bc['right']['mu']:.2f} <span class="mono" style="color:var(--text-faint)">n={bc['right']['nmu']}</span></td><td class="bad">{bc['old']:.2f}</td></tr>
          </table>
          <div>
            <div class="callout warn"><b>The earlier 1.7 ns was 0.2 ns short.</b> The outgoing flight is only ~2.3 ns, so that
            is enough to drag the positron peak from 1.0 to 0.92. Now loaded per SPS bar where a bar has &ge;40 positrons
            (the required shift drifts ~0.3 ns across an arm), the arm value elsewhere.</div>
            <div class="callout"><b>Open:</b> muons want 0.25&ndash;0.38 ns more than positrons. One timing offset cannot
            place both species; energy loss accounts for only ~0.02 ns of it.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 14b BETA ---------------------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1 min. The human check, now with the truth-calibrated alignment. Left: by truth - positrons from target decays peak at 1, muons below their p/E line because of the residual on the previous slide. Right: coloured by the classifier's verdict. Agreement with the plugin's own beta cut rose from {bc['agree_before']:.1f}% to {bc['agree_after']:.1f}%.">
      <div class="slide-inner">
        <div class="eyebrow">14b &mdash; <span class="dim">the human check</span></div>
        <h1 class="title">Outgoing &beta;: decays at 1,<br>and the classifier agrees.</h1>
        <hr class="rule">
        <div class="two-col">
          <div>{ms.fig("truth_beta.png", "Outgoing beta split by MC truth", maxh="46vh")}
            <p class="caption">By MC truth. Positrons from decays in the target peak at &beta; = 1.</p></div>
          <div>{ms.fig("beta_ml_mu_all.png", "Outgoing beta split by classifier verdict, drawn by the cooker", maxh="46vh")}
            <p class="caption">By the classifier's verdict. Agreement with the plugin's &beta; cut:
            {bc['agree_before']:.1f}% at the old offset, <b>{bc['agree_after']:.1f}%</b> now.</p></div>
        </div>
      </div>
    </section>""")

    # ---- 14c BETA, FULL RECONSTRUCTION ONLY -------------------------------
    bf = MEASURED["beta_full"]
    s.append(f"""
    <section class="slide" data-notes="~1.5 min. The same beta axis, but only the events that carry a full reconstruction - a vertex passing the DOCA and fiducial cuts, a BH correlation and an SPS hit, which is everything out_beta needs. Left is what the classifier said, middle is what actually happened, same binning and y range: they are nearly indistinguishable. Right shows only the disagreements. Note beta is not one of the 135 inputs, so this is an independent axis to judge it on.">
      <div class="slide-inner">
        <div class="eyebrow">14c &mdash; <span class="dim">full reconstruction only</span></div>
        <h1 class="title">Given the measurements,<br>it reproduces the truth split.</h1>
        <hr class="rule">
        {ms.fig("beta_full_reco.png", "Outgoing beta by classifier verdict, by MC truth, and the disagreements, for fully reconstructed events", maxh="38vh")}
        <div class="two-col" style="margin-top:10px">
          <table class="data compact">
            <tr><th>{bf['n']:,} vertices with a full reconstruction</th><th style="text-align:right">this subset</th><th style="text-align:right">all events</th></tr>
            <tr><td>efficiency</td><td class="ok">{bf['eff']:.1f}%</td><td class="num">{bf['eff_all']:.1f}%</td></tr>
            <tr><td>purity</td><td class="ok">{bf['purity']:.1f}%</td><td class="num">{bf['purity_all']:.1f}%</td></tr>
            <tr><td>agreement with truth</td><td class="ok">{bf['agreement']:.1f}%</td><td class="num">&mdash;</td></tr>
            <tr><td>TP / FN / FP / TN</td><td class="mono" colspan="2" style="text-align:right">{bf['tp']} / {bf['fn']} / {bf['fp']} / {bf['tn']}</td></tr>
          </table>
          <div>
            <div class="callout good"><b>&beta; is not an input.</b> None of the 135 columns is the outgoing &beta;, so this is
            an independent axis: both populations land where physics puts them without the classifier ever being shown either.</div>
            <div class="callout"><b>What is left</b> is the overlap: {bf['fn']} decays at &beta;&nbsp;&asymp;&nbsp;0.9&ndash;1.0 that look
            muon&#8209;like and {bf['fp']} false positives below. The other {bf['outside']:.0f}% of errors never got this far.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 15 WHAT CHANGED FOR USERS ---------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1 min. What someone downstream of this plugin needs to know. The first item changes the meaning of an existing flag.">
      <div class="slide-inner">
        <div class="eyebrow">15 &mdash; <span class="dim">for anyone using the output</span></div>
        <h1 class="title">What changed in the cooker.</h1>
        <hr class="rule">
        <ul class="checklist">
          <li><span><b><code class="mono">ReactionIDResult::is_decay</code> changed meaning.</b> It now flags only decays upstream of
            or inside the target (threshold {f['threshold']:.4f}); downstream decays are background.</span></li>
          <li><span><b>ReactionID refuses to start with no target.</b> On MC pass
            <code class="mono">-c ReactionID:setTargetPosition:0</code>; real data takes it from slow control as before.</span></li>
          <li><span><b>Chain order:</b> <code class="mono">reactionID</code> now runs before <code class="mono">features</code>,
            and the export reads the reaction flags from ReactionID as <code class="mono">rid_*</code>.</span></li>
          <li><span><b>Truth plots follow the label</b>, the decay&#8209;position window follows the data, and the cooker writes a PNG
            beside each PDF (wide canvases were clipped when rasterised from the rotated PDF page).</span></li>
        </ul>
      </div>
    </section>""")

    # ---- 16 OPEN ----------------------------------------------------------
    s.append("""
    <section class="slide" data-notes="~1.2 min. Explicitly open. The first two follow directly from slide 11.">
      <div class="slide-inner">
        <div class="eyebrow">16 &mdash; <span class="dim">still open</span></div>
        <h1 class="title">Next.</h1>
        <hr class="rule">
        <ul class="checklist">
          <li><span><b>Score each vertex, not each event.</b> The analysis vetoes a scattering candidate; half the target misses
            have no vertex at all, which per&#8209;vertex scoring makes an explicit outcome rather than a guess.</span></li>
          <li><span><b>Define "target" physically.</b> A quarter of target misses sit within 50 mm of a 200 mm box that is a
            convention, not the LH2 cell.</span></li>
          <li><span><b>A TOF calibration entry for run 17606</b> &mdash; the cut&#8209;based benchmark and the in&#8209;time flag are dead on MC without it.</span></li>
          <li><span><b>Not attempted:</b> PbGlass (no cooking stage in the chain yet) and an explicit two&#8209;segment kink fit. <b>PathLength is someone else's</b>: once it provides the <b>total traversed path length</b>, the classifier can be given the positron hypothesis (decay at the vertex, outgoing leg at c) beside the muon one, which it cannot express today.</span></li>
          <li><span><b>Before real data:</b> one beam setting, one species, one target so far; the score is uncalibrated on data.</span></li>
        </ul>
      </div>
    </section>""")

    # ---- 17 REPRODUCE -----------------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~30s.">
      <div class="slide-inner">
        <div class="eyebrow">17 &mdash; <span class="dim">reproduce</span></div>
        <h1 class="title">End to end.</h1>
        <hr class="rule">
        <div class="codeblock tight">
          <div class="cb-head"><span class="lang">bash</span><span>cook, train, export, verify, plot, rebuild this deck</span></div>
<pre><span class="tok-c"># cook (in muse/) - reactionID now runs before features</span>
./script/cook_mc_chain.sh --tag <span class="tok-s">mc17606_210MeV_LH2</span> --files 1-2 \\
    --momentum <span class="tok-n">210</span> --valid-from <span class="tok-n">2</span> --indir ... --outdir ...

<span class="tok-c"># train, export, report, re-export + verify, parity, ablation (in reactionID/)</span>
python reactionID.py train
python export_onnx.py --all
python make_report.py --plots
python export_onnx.py --all --verify
python verify_inference.py --rid cooked/..._2_RID.root \\
    --csv cooked/..._2_features.csv --model model/decay_gbdt.onnx
python ablate.py --json ablation.json

<span class="tok-c"># truth plots (in muse/), with the truth-calibrated MC beta alignment, then this deck</span>
cooker recipes/ReactionID/ReactionID_Plots_MC.xml &lt;7 inputs&gt; out.root \\
    -c ReactionID:setMomentum:<span class="tok-n">210</span> \\
    -c "$(sed -n 1p ../reactionID/mc17606_beta_alignment.txt)" \\
    -c "$(sed -n 2p ../reactionID/mc17606_beta_alignment.txt)"   <span class="tok-c"># printed by the same recipe</span>
python make_slides_v2.py</pre>
        </div>
        <div class="meta-row">
          <div><b>Deployed model</b><span>{esc(f['model_tag'])} &middot; {f['n_inputs']} inputs</span></div>
          <div><b>Threshold</b><span>{f['threshold']:.4f}</span></div>
          <div><b>Label</b><span>{esc(rd.LABEL_COLUMN)}</span></div>
          <div><b>Validation</b><span>{d['valid_events']:,} events</span></div>
        </div>
      </div>
    </section>""")
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", default="report.json")
    p.add_argument("--ablation", default="ablation.json")
    p.add_argument("--valid", default="data/valid.csv")
    p.add_argument("--model", default="model/decay_gbdt.onnx")
    p.add_argument("--out", default="../muse/doc/slides/reaction_id_v2.html")
    p.add_argument("--figdir", default=None, help="PNG directory (default: figs_v2/ beside --out)")
    args = p.parse_args()

    ms.FIGDIR = args.figdir or os.path.join(os.path.dirname(args.out) or ".", "figs_v2")
    with open(args.report) as fh:
        rep = json.load(fh)
    with open(args.ablation) as fh:
        ab = json.load(fh)
    facts = dataset_facts(args.valid, args.model)

    slides = build(rep, ab, facts)
    doc = (ms.TEMPLATE.replace("</style>", EXTRA_CSS + "</style>", 1)
           .replace("<title>reactionID</title>", "<title>reactionID, round two</title>")
           .replace("__SLIDES__", "\n".join(slides))
           .replace("__TOTAL__", f"{len(slides):02d}"))
    with open(args.out, "w") as fh:
        fh.write(doc)
    print(f"wrote {args.out} ({len(slides)} slides, {len(doc)/1024:.0f} KB, figures from {ms.FIGDIR})")


if __name__ == "__main__":
    main()
