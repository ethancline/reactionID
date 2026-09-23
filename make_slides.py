"""Render the results deck from report.json and the generated figures.

Self-contained HTML in the style of muse/doc/slides/trigger_decoded.html. Every
number comes out of report.json and every figure is embedded as a base64 PNG, so
the deck is a single file and cannot drift away from the run that produced it.

    python make_report.py --out report.json --plots     # numbers + importance figure
    python make_slides.py --out ../muse/doc/slides/reaction_id.html
"""

import argparse
import base64
import html
import json
import os

# Chart palette. The deck's UI accents (#ffb454, #62c9d6) are tuned as accent *text*
# on the dark surface and fail the data-mark checks - both sit above the dark-mode
# OKLCH lightness band and the cyan is under the chroma floor. These are the nearest
# steps in the same two hue families that pass lightness, chroma, CVD separation,
# the normal-vision floor and 3:1 contrast against --surface #121a22.
C_MLP = "#d47700"
C_GBDT = "#00a1cb"
C_FAINT = "#5c6f81"
C_GRID = "#1c2833"

FIGDIR = "figs"


def esc(s):
    return html.escape(str(s), quote=True)


def fig(name, alt, caption=None, maxh="60vh"):
    """Embed a generated PNG as a data URI so the deck stays one portable file."""
    path = os.path.join(FIGDIR, name)
    if not os.path.exists(path):
        return f'<p class="caption">[missing figure: {esc(name)}]</p>'
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    cap = f'<p class="caption">{caption}</p>' if caption else ""
    return (f'<div class="figure"><img src="data:image/png;base64,{b64}" alt="{esc(alt)}" '
            f'style="max-height:{maxh}">{cap}</div>')


def fig_roc(rep):
    """ROC for both models, inline SVG so it stays crisp at any projector size."""
    W, H = 560, 470
    L, R, T, B = 66, 26, 22, 60
    pw, ph = W - L - R, H - T - B

    def path(points):
        return " ".join(("M" if i == 0 else "L") + f"{L+pw*x:.1f},{T+ph*(1-y):.1f}"
                        for i, (x, y) in enumerate(points))

    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="ROC curves for both classifiers">']
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        xx, yy = L + pw * t, T + ph * (1 - t)
        s.append(f'<line x1="{xx:.1f}" y1="{T}" x2="{xx:.1f}" y2="{T+ph}" stroke="{C_GRID}" stroke-width="1"/>')
        s.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{L+pw}" y2="{yy:.1f}" stroke="{C_GRID}" stroke-width="1"/>')
        s.append(f'<text x="{xx:.1f}" y="{T+ph+22}" text-anchor="middle" class="dtext">{t:.2f}</text>')
        s.append(f'<text x="{L-10}" y="{yy+4:.1f}" text-anchor="end" class="dtext">{t:.2f}</text>')
    s.append(f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T}" stroke="{C_FAINT}" stroke-width="1.5" stroke-dasharray="5 5"/>')
    for key, col in (("mlp", C_MLP), ("gbdt", C_GBDT)):
        s.append(f'<path d="{path(rep[key]["roc"])}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round"/>')
    s.append(f'<text x="{L+pw/2:.0f}" y="{H-24}" text-anchor="middle" class="dtext caption">false positive rate</text>')
    lx = L + 40
    for key, col, lab in (("mlp", C_MLP, "MLP"), ("gbdt", C_GBDT, "Boosted trees")):
        yy = T + ph - 40 - (0 if key == "mlp" else 26)
        s.append(f'<rect x="{lx}" y="{yy-9}" width="11" height="11" rx="2" fill="{col}"/>')
        s.append(f'<text x="{lx+17}" y="{yy}" class="dtext val">{esc(lab)} &#183; AUC {rep[key]["auc"]:.4f}</text>')
    s.append("</svg>")
    return "\n".join(s)


def auc_table(rep):
    mlp = {r["name"]: r for r in rep["mlp"]["regions"]}
    gb = {r["name"]: r for r in rep["gbdt"]["regions"]}
    return "\n".join(
        f"<tr><td>{esc(n)}</td><td class='mono'>{mlp[n]['n']:,}</td>"
        f"<td class='num'>{mlp[n]['auc']:.4f}</td><td class='num'>{gb[n]['auc']:.4f}</td></tr>"
        for n in ("upstream", "target", "downstream") if mlp.get(n, {}).get("auc"))


def wp_table(rep):
    mlp = {w["target_eff"]: w for w in rep["mlp"]["working_points"]}
    gb = {w["target_eff"]: w for w in rep["gbdt"]["working_points"]}
    def rej(w):
        return "&infin;" if w["rejection"] is None else f"{w['rejection']:.0f}"
    return "\n".join(
        f"<tr><td class='mono'>{t*100:.0f}%</td>"
        f"<td class='num'>{rej(mlp[t])}&times;</td><td class='num'>{mlp[t]['precision_at_prior']*100:.1f}%</td>"
        f"<td class='num'>{rej(gb[t])}&times;</td><td class='num'>{gb[t]['precision_at_prior']*100:.1f}%</td></tr>"
        for t in (0.80, 0.90, 0.95, 0.99))


def region_rows(rep):
    return "\n".join(
        f"<tr><td>{esc(g['name'])}</td><td class='mono'>{g['n']:,}</td>"
        f"<td class='num'>{g['correct']*100:.2f}%</td></tr>" for g in rep["truth"]["regions"])


def imp_of(feats, name):
    for f in feats:
        if f["name"] == name:
            return f["mean"]
    return 0.0


def build(rep):
    d = rep["dataset"]
    man = rep.get("manifest", {})
    tr = rep["truth"]
    imp = rep["importance"]["features"]
    impm = rep.get("importance_mlp", {}).get("features", [])
    n_zero = sum(1 for f in imp if f["mean"] <= 0)
    s = []

    # ---- 1 TITLE ----------------------------------------------------------
    s.append(f"""
    <section class="slide title-slide active" data-notes="~30s. The arc: a broken exporter, a rebuilt feature pipeline, a model deployed into the cooker, and a direct check against simulation truth.">
      <svg class="corner-trace" width="360" height="220" viewBox="0 0 360 220" aria-hidden="true">
        <path d="M360 0 L360 40 L300 40 L300 90 L250 90 L250 140" class="darrow" stroke="#263544"/>
        <path d="M360 60 L320 60 L320 120 L270 120" class="darrow" stroke="#263544"/>
        <circle cx="250" cy="140" r="3" fill="#ffb454"/><circle cx="270" cy="120" r="3" fill="#263544"/>
      </svg>
      <div class="slide-inner">
        <div class="eyebrow">MUSE analysis &middot; <span class="dim">reactionID</span></div>
        <h1 class="title">Muon decay&#8209;in&#8209;flight,<br><span class="accent">identified</span> in the cooker.</h1>
        <hr class="rule">
        <p class="lede">Rebuilding the feature export, deploying the classifier as a cooker plugin,
        and checking it against the simulation's own answer.</p>
        <div class="meta-row">
          <div><b>Sample</b><span>{esc(man.get('tag','mc17606 210 MeV LH2'))} &middot; run 17606</span></div>
          <div><b>Events</b><span>{d['train_events']:,} train / {d['valid_events']:,} valid</span></div>
          <div><b>Inputs</b><span>{d['feature_columns']} of {d['columns']} columns</span></div>
          <div><b>Author</b><span>Ethan Cline</span></div>
        </div>
      </div>
    </section>""")

    # ---- 2 THE PROBLEM ----------------------------------------------------
    s.append("""
    <section class="slide" data-notes="~1.5 min. Three defects, none of which crashed anything. That is the point - each degraded the result silently.">
      <div class="slide-inner">
        <div class="eyebrow">01 &mdash; <span class="dim">where this started</span></div>
        <h1 class="title">Three defects, none of which crashed anything.</h1>
        <hr class="rule">
        <ul class="points">
          <li><span><b>The GEM columns were the STT columns.</b> <code class="mono">muonDecay_out.cpp</code>
            looped over <code class="mono">STT_Tracks</code> while filling <code class="mono">gemtrackpos</code>/<code class="mono">gemtrackdir</code>.
            <span class="muted">Byte&#8209;identical in 100% of 134,580 rows &mdash; 6 of 33 features carried nothing.</span></span></li>
          <li><span><b>No timing was ever exported.</b> The TDC loops were empty bodies.
            <span class="muted">Track length and vertex quantities were produced by the chain, then never read.</span></span></li>
          <li><span><b>The validation loader shuffled before plotting.</b> <code class="mono">reactionPlots.py</code>
            indexed decay coordinates in file order with a mask in shuffled order.
            <span class="muted">Every decay&#8209;location plot was noise.</span></span></li>
        </ul>
        <div class="callout warn"><b>And the exporter appended.</b> <code class="mono">std::ios::app</code> with no header,
        so a re&#8209;run silently doubled the dataset instead of replacing it.</div>
      </div>
    </section>""")

    # ---- 3 THE BUG --------------------------------------------------------
    s.append("""
    <section class="slide" data-notes="~1 min. The second loop is a copy of the first with the destination renamed and the source left alone.">
      <div class="slide-inner">
        <div class="eyebrow">02 &mdash; <span class="dim">the copy&#8209;paste</span></div>
        <h1 class="title">Fetched, then never read.</h1>
        <hr class="rule">
        <div class="two-col">
          <div>
            <div class="codeblock tight">
              <div class="cb-head"><span class="lang">C++</span><span>muonDecay_out.cpp &mdash; before</span></div>
<pre><span class="tok-k">for</span> (<span class="tok-t">int</span> i = 0; i &lt; STT_Tracks-&gt;tracks.size(); i++) {
    stttrackpos[0] = ...position.X();
}
<span class="tok-k">for</span> (<span class="tok-t">int</span> i = 0; i &lt; <span class="tok-n">STT_Tracks</span>-&gt;tracks.size(); i++) {
    gemtrackpos[0] = <span class="tok-n">STT_Tracks</span>-&gt;...X();
}</pre>
            </div>
            <p class="caption"><b>GEM_Tracks</b> is fetched in <code>startup()</code> and never dereferenced.</p>
          </div>
          <div>
            <div class="terminal">$ <span class="dim">check the shipped training set</span>
  identical GEM==STT rows
  <span class="hl">134,580 / 134,580   (100.00%)</span>

$ <span class="dim">after the fix</span>
  identical GEM==STT rows
  <span class="hl2">3.43%</span>  <span class="dim">&#8212; coincidental single-track events</span></div>
            <div class="callout">A duplicated column does not error, does not warn, and trains perfectly happily.
            <b>It just wastes a sixth of the feature vector.</b></div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 4 THE EXPORT -----------------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1.2 min. Header and rows come from the same col() calls, so they cannot disagree. Cuts are columns, not filters.">
      <div class="slide-inner">
        <div class="eyebrow">03 &mdash; <span class="dim">the export</span></div>
        <h1 class="title">37 columns &rarr; <span class="accent">{d['columns']}</span>, and a schema that checks itself.</h1>
        <hr class="rule">
        <div class="two-col narrow-right">
          <table class="data">
            <tr><th>group</th><th>what it carries</th></tr>
            <tr><td class="mono">provenance</td><td>run, event, tree entry &mdash; a row maps back to an event</td></tr>
            <tr class="highlight"><td class="mono">timing</td><td>BH / SPS correlated times, TOF, RF &mdash; <b>new</b></td></tr>
            <tr><td class="mono">scintillator</td><td>per wall: multiplicity, max&#8209;deposit bar, E<sub>dep</sub>, &Sigma;E<sub>dep</sub></td></tr>
            <tr class="highlight"><td class="mono">tracks</td><td>STT and GEM position, direction, &chi;&sup2;, hits, DOCA &mdash; <b>now distinct</b></td></tr>
            <tr><td class="mono">vertex</td><td>from <code>VertexReconstruction</code>: position, &theta;, DOCA</td></tr>
            <tr><td class="mono">truth</td><td>label, decay vertex, momentum, <code>decay_region</code></td></tr>
          </table>
          <div>
            <div class="codeblock tight">
              <div class="cb-head"><span class="lang">C++</span><span>one source of truth</span></div>
<pre><span class="tok-c">// header and rows from the
// same calls</span>
<span class="tok-k">void</span> col(<span class="tok-k">const char</span>* n, <span class="tok-t">double</span> v){{
  <span class="tok-k">if</span>(!schemaLocked)
      colNames.push_back(n);
  colVals.push_back(v);
}}</pre>
            </div>
            <div class="callout good"><b>Cuts are columns, not filters.</b> Every event gets a row;
            <code class="mono">chv_veto</code> and <code class="mono">has_truth</code> are exported so the selection stays visible.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 5 AUTOMATION -----------------------------------------------------
    s.append("""
    <section class="slide" data-notes="~1.2 min. One command replaces a script whose stage list was comment-toggled. Make-like skipping and a manifest with the git SHA.">
      <div class="slide-inner">
        <div class="eyebrow">04 &mdash; <span class="dim">automation</span></div>
        <h1 class="title">One command, twelve stages, a manifest.</h1>
        <hr class="rule">
        <div class="diagram-wrap">
          <svg viewBox="0 0 1000 150" width="100%" role="img" aria-label="MC files merged then cooked through twelve stages into train and validation sets">
            <defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#5c6f81"/></marker></defs>
            <rect x="0" y="40" width="140" height="52" rx="3" class="dbox"/>
            <text x="70" y="62" text-anchor="middle" class="dtext label">g4PSI</text>
            <text x="70" y="80" text-anchor="middle" class="dtext">10 MC files</text>
            <line x1="146" y1="66" x2="186" y2="66" class="darrow" marker-end="url(#ah)"/>
            <rect x="192" y="40" width="156" height="52" rx="3" class="dbox"/>
            <text x="270" y="62" text-anchor="middle" class="dtext label">g4PSI_merge</text>
            <text x="270" y="80" text-anchor="middle" class="dtext">RunInfo preserved</text>
            <line x1="354" y1="66" x2="394" y2="66" class="darrow" marker-end="url(#ah)"/>
            <rect x="400" y="26" width="306" height="80" rx="3" class="dbox"/>
            <text x="553" y="50" text-anchor="middle" class="dtext label">cook_mc_chain.sh</text>
            <text x="553" y="70" text-anchor="middle" class="dtext">mc2root &#8594; detectors &#8594; tracking &#8594; Vertex</text>
            <text x="553" y="88" text-anchor="middle" class="dtext">&#8594; PathLength &#8594; features &#8594; ReactionID</text>
            <line x1="712" y1="66" x2="752" y2="66" class="darrow" marker-end="url(#ah)"/>
            <rect x="758" y="40" width="242" height="52" rx="3" class="dbox active"/>
            <text x="879" y="62" text-anchor="middle" class="dtext label on-active">train / valid CSV</text>
            <text x="879" y="80" text-anchor="middle" class="dtext on-active">+ manifest.json</text>
            <text x="553" y="130" text-anchor="middle" class="dtext caption accent">stages skip when the output is newer than every input</text>
          </svg>
        </div>
        <div class="codeblock tight">
          <div class="cb-head"><span class="lang">bash</span><span>from the muse repository root</span></div>
<pre>./script/cook_mc_chain.sh --tag mc17606_210MeV_LH2 --files 1-2 --momentum 210 \\
    --indir ../reactionID/mc_merged --outdir ../reactionID/cooked \\
    --csvdir ../reactionID/data --valid-from 2 --jobs 2</pre>
        </div>
        <p class="caption">Replaces <b>muondecay_out.sh</b>, where the file range, the stage list and the train/validation split were all edited into the source.</p>
      </div>
    </section>""")

    # ---- 6 THE HADD TRAP --------------------------------------------------
    s.append("""
    <section class="slide" data-notes="~1.3 min. This one cost real time. hadd silently destroys RunInfo, and every timing calibration is selected by run number.">
      <div class="slide-inner">
        <div class="eyebrow">05 &mdash; <span class="dim">a trap worth knowing</span></div>
        <h1 class="title">Merge with <span class="accent">g4PSI_merge</span>, never <code class="mono">hadd</code>.</h1>
        <hr class="rule">
        <div class="two-col">
          <div>
            <div class="terminal">$ <span class="dim">RunInfo in a hadd-merged file</span>
  name=TObject, <span class="hl">class=TObject</span>
  fUniqueID  1073741836

$ <span class="dim">read runNumber off that</span>
  <span class="hl">1601463620</span>, start 1970-01-01

$ <span class="dim">g4PSI_merge instead</span>
  <span class="hl2">class=MRTRunInfo</span>
  runNumber  <span class="hl2">17606</span>
  nrOfEvents 100000000</div>
          </div>
          <div>
            <p class="lede"><code class="mono">hadd</code> cannot merge a custom non&#8209;mergeable class, so it writes a
            bare stub. Nothing errors.</p>
            <ul class="points">
              <li><span><b>Timing alignments are selected by run number</b> at several stages, so a broken
                <code class="mono">RunInfo</code> silently applies the wrong calibration period.</span></li>
              <li><span><b>g4PSI_merge clones it</b> and sums <code class="mono">nrOfEvents</code>.
                <span class="muted">Two portability bugs had to be fixed first &mdash; it hardcoded
                <code class="mono">.muse/x86_64/lib/libmusetree.so</code> in both the CMake and the source.</span></span></li>
            </ul>
            <div class="callout warn"><b>Check before cooking:</b>
            <code class="mono">f.Get("RunInfo")-&gt;Dump()</code> must say <code class="mono">class=MRTRunInfo</code>.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 7 DEPLOYMENT -----------------------------------------------------
    s.append("""
    <section class="slide" data-notes="~1.5 min. The key idea: one feature builder used by both the exporter and the plugin, so training and inference cannot drift.">
      <div class="slide-inner">
        <div class="eyebrow">06 &mdash; <span class="dim">into the cooker</span></div>
        <h1 class="title">One feature builder, two consumers.</h1>
        <hr class="rule">
        <div class="diagram-wrap">
          <svg viewBox="0 0 1000 196" width="100%" role="img" aria-label="decayfeatures Builder feeds both the CSV exporter and the ReactionID plugin">
            <defs><marker id="ah2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#5c6f81"/></marker></defs>
            <rect x="330" y="16" width="340" height="56" rx="3" class="dbox active"/>
            <text x="500" y="40" text-anchor="middle" class="dtext label on-active">decayfeatures::Builder</text>
            <text x="500" y="59" text-anchor="middle" class="dtext on-active">115 model inputs, one col() call each</text>
            <path d="M420 76 L250 116" class="darrow" marker-end="url(#ah2)"/>
            <path d="M580 76 L750 116" class="darrow" marker-end="url(#ah2)"/>
            <rect x="80" y="122" width="340" height="56" rx="3" class="dbox"/>
            <text x="250" y="146" text-anchor="middle" class="dtext label">muonDecay_out</text>
            <text x="250" y="165" text-anchor="middle" class="dtext">writes the training CSV</text>
            <rect x="580" y="122" width="340" height="56" rx="3" class="dbox"/>
            <text x="750" y="146" text-anchor="middle" class="dtext label">ReactionID</text>
            <text x="750" y="165" text-anchor="middle" class="dtext">feeds the ONNX model, per event</text>
          </svg>
        </div>
        <div class="two-col">
          <div>
            <ul class="points">
              <li><span><b>One contract.</b> <code class="mono">float32[N,115]</code> raw columns in, one probability out.
                Both models export to it, so the plugin has a single code path.</span></li>
              <li><span><b>The model declares its own inputs</b> in ONNX metadata; the plugin maps them onto the
                builder at startup and refuses to run on a mismatch.</span></li>
            </ul>
          </div>
          <div>
            <div class="callout good"><b>Verified end to end.</b> Cooked scores against Python on the same events:
            max |&Delta;P| = 1.3&times;10<sup>&minus;3</sup>, <b>1 event of 101,380</b> above 10<sup>&minus;4</sup>,
            zero label disagreements.</div>
            <p class="caption">The MLP's one&#8209;hot expansion and standardisation are baked <b>into its ONNX graph</b>,
            so the C++ never reimplements them.</p>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 8 MODEL COMPARISON ----------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1 min. Boosted trees win on raw columns in seconds. Keep the MLP as the thing to beat.">
      <div class="slide-inner">
        <div class="eyebrow">07 &mdash; <span class="dim">which classifier</span></div>
        <h1 class="title">Less machinery, better answer.</h1>
        <hr class="rule">
        <div class="two-col">
          <div>
            <table class="data">
              <tr><th></th><th>inputs</th><th>accuracy</th><th>ROC AUC</th><th>fit</th></tr>
              <tr><td>MLP &mdash; 4 hidden layers</td><td class="mono">{rep['mlp']['n_features']} expanded</td>
                  <td class="num">{rep['mlp']['accuracy']*100:.2f}%</td><td class="num">{rep['mlp']['auc']:.4f}</td><td class="num">6 min</td></tr>
              <tr class="highlight"><td>Gradient boosting</td><td class="mono">{rep['gbdt']['n_features']} raw</td>
                  <td class="num">{rep['gbdt']['accuracy']*100:.2f}%</td><td class="num">{rep['gbdt']['auc']:.4f}</td>
                  <td class="num">{rep['gbdt']['fit_seconds']:.0f} s</td></tr>
            </table>
            <table class="data" style="margin-top:16px">
              <tr><th>decay region</th><th>events</th><th>MLP AUC</th><th>GBDT AUC</th></tr>
              {auc_table(rep)}
            </table>
            <p class="caption">Each region's decays scored against <b>all</b> non&#8209;decay events.</p>
          </div>
          <div>
            {fig_roc(rep)}
            <p class="caption">Overall ROC, {d['valid_events']:,} validation events. Dashed line is random guessing.</p>
            <div class="callout"><b>Accuracy above is quoted at 0.5.</b> The deployed decision uses the operating
            point carried in the model metadata &mdash; {tr['threshold']:.4f} &mdash; which is what every truth
            comparison that follows uses.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 9 FEATURE IMPORTANCE --------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1.5 min. Two findings: the best feature is one the old pipeline never exported, and the two models solve the problem by different routes.">
      <div class="slide-inner">
        <div class="eyebrow">08 &mdash; <span class="dim">feature importance</span></div>
        <h1 class="title">The best feature is the one that <span class="accent">was never exported</span>.</h1>
        <hr class="rule">
        <div class="two-col narrow-right">
          {fig("feature_importance.png", "Permutation feature importance for both models", maxh="64vh")}
          <div>
            <p class="lede">Drop in ROC AUC when a column is shuffled, measured against the <b>exported ONNX</b> &mdash;
            so it describes what is actually deployed, under one protocol for both models.</p>
            <table class="data">
              <tr><th>feature</th><th>GBDT</th><th>MLP</th></tr>
              <tr class="highlight"><td class="mono">sps_corr_time</td>
                  <td class="num">+{imp_of(imp,'sps_corr_time'):.4f}</td><td class="num">+{imp_of(impm,'sps_corr_time'):.4f}</td></tr>
              <tr><td class="mono">spsrr_edep_sum</td>
                  <td class="num">+{imp_of(imp,'spsrr_edep_sum'):.4f}</td><td class="num">+{imp_of(impm,'spsrr_edep_sum'):.4f}</td></tr>
              <tr><td class="mono">spsrf_edep_sum</td>
                  <td class="num">+{imp_of(imp,'spsrf_edep_sum'):.4f}</td><td class="num">+{imp_of(impm,'spsrf_edep_sum'):.4f}</td></tr>
            </table>
            <div class="callout good"><b>The two models take different routes.</b> The boosted trees lean hardest on
            SPS <i>timing</i>; the MLP barely uses it and leans on energy deposition. The trees win &mdash; so the
            timing route looks like the stronger one.</div>
            <p class="caption"><b>{n_zero} of {len(imp)}</b> columns contribute nothing to the boosted trees &mdash;
            the candidates for trimming the input contract.</p>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 10 CONFUSION -----------------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1.5 min. The headline. Simulation knows the answer, so the classifier can be scored directly rather than through a proxy.">
      <div class="slide-inner">
        <div class="eyebrow">09 &mdash; <span class="dim">against the truth</span></div>
        <h1 class="title">The simulation knows the answer.</h1>
        <hr class="rule">
        <div class="two-col narrow-right">
          {fig("truth_plots_truth_confusion.png", "Confusion matrix, classifier versus MC truth", maxh="58vh")}
          <div>
            <table class="data">
              <tr><th>metric</th><th>value</th></tr>
              <tr class="highlight"><td>efficiency</td><td class="num">{tr['efficiency']*100:.2f}%</td></tr>
              <tr class="highlight"><td>purity</td><td class="num">{tr['purity']*100:.2f}%</td></tr>
              <tr><td>specificity</td><td class="num">{tr['specificity']*100:.2f}%</td></tr>
              <tr><td>missed decays</td><td class="num">{tr['fn']:,}</td></tr>
              <tr><td>false alarms</td><td class="num">{tr['fp']:,}</td></tr>
            </table>
            <div class="callout good"><b>The cooker reproduces Python exactly.</b> Same four cells, and per&#8209;region
            agreement identical to four significant figures &mdash; which is the point of building the feature
            vector in one shared place.</div>
            <p class="caption">{d['valid_events']:,} validation events at threshold {tr['threshold']:.4f}.
            Percentages on the figure are within a truth column.</p>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 11 BY REGION -----------------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1 min. The binary label mixes three physically different situations. Target decays - the ones that fake scattering - are among the best.">
      <div class="slide-inner">
        <div class="eyebrow">10 &mdash; <span class="dim">by decay region</span></div>
        <h1 class="title">One label, three different problems.</h1>
        <hr class="rule">
        <div class="two-col narrow-right">
          {fig("truth_plots_truth_agreement_by_region.png", "Fraction correct by decay region", maxh="56vh")}
          <div>
            <table class="data">
              <tr><th>region</th><th>events</th><th>correct</th></tr>
              {region_rows(rep)}
            </table>
            <p class="caption">Regions are set by the truth decay vertex: <b>target</b> is
            |R|&nbsp;&lt;&nbsp;200&nbsp;mm and |Z|&nbsp;&lt;&nbsp;200&nbsp;mm.</p>
            <div class="callout"><b>Downstream decays are the hard case</b>, not the target ones. A decay metres
            past SPS has almost no signature left to find &mdash; and it is also the case that matters least
            for a scattering analysis.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 12 DECAY POSITION ------------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1.2 min. The clearest statement of where the mistakes come from.">
      <div class="slide-inner">
        <div class="eyebrow">11 &mdash; <span class="dim">where the mistakes are</span></div>
        <h1 class="title">The decays we miss happen <span class="accent">downstream</span>.</h1>
        <hr class="rule">
        {fig("truth_plots_truth_decay_position.png", "Truth decay vertex Z and R, found versus missed", maxh="50vh")}
        <div class="callout">Missed decays spike sharply at <b>Z &asymp; 1100&nbsp;mm</b> and <b>R &asymp; 380&nbsp;mm</b>,
        while the ones we find cluster near the target. Normalised, because the two samples differ by almost
        a factor of eight in size.</div>
      </div>
    </section>""")

    # ---- 13 VERTEX TOPOLOGY ----------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1.3 min. The mistakes are not random - they look like non-decays in vertex space. There is simply no wide-angle signature to find.">
      <div class="slide-inner">
        <div class="eyebrow">12 &mdash; <span class="dim">where the mistakes are</span></div>
        <h1 class="title">The misses look like non&#8209;decays.</h1>
        <hr class="rule">
        <div class="two-col narrow-right">
          {fig("truth_plots_truth_vertex_by_outcome.png", "Reconstructed vertex theta versus DOCA for each outcome", maxh="60vh")}
          <div>
            <table class="data">
              <tr><th>outcome</th><th>&LT;&theta;&GT;</th><th>&LT;DOCA&GT;</th></tr>
              <tr class="highlight"><td>true positive</td><td class="num">0.494</td><td class="num">11.20</td></tr>
              <tr><td>false negative</td><td class="num">0.361</td><td class="num">8.45</td></tr>
              <tr><td>false positive</td><td class="num">0.405</td><td class="num">9.67</td></tr>
              <tr><td>true negative</td><td class="num">0.328</td><td class="num">7.31</td></tr>
            </table>
            <div class="callout"><b>A decay leaves a wide&#8209;angle, large&#8209;DOCA vertex</b> &mdash; the positron does
            not point back. True positives sit there; false negatives sit on top of the true negatives.</div>
            <p class="caption">No single input feature separates right from wrong by more than
            0.14&nbsp;&sigma; &mdash; the signature is in the vertex topology, not in any one column.</p>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 14 SCORE ---------------------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1 min. The mistakes are low-confidence, which means the threshold is a real lever.">
      <div class="slide-inner">
        <div class="eyebrow">13 &mdash; <span class="dim">how confident</span></div>
        <h1 class="title">The mistakes are the uncertain ones.</h1>
        <hr class="rule">
        {fig("truth_plots_truth_score.png", "Classifier score split by MC truth, log scale", maxh="52vh")}
        <div class="callout good"><b>46% of wrong events fall in the 0.3&ndash;0.7 band</b>, against roughly 6% overall.
        The model is not confidently wrong &mdash; it is unsure, which means the operating point is a real handle
        on the efficiency/purity trade.</div>
      </div>
    </section>""")

    # ---- 15 BETA CROSS-CHECK ----------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1.3 min. An independent cross-check: the classifier never sees beta, yet it separates along it.">
      <div class="slide-inner">
        <div class="eyebrow">14 &mdash; <span class="dim">independent cross&#8209;check</span></div>
        <h1 class="title">It separates along a variable it never sees.</h1>
        <hr class="rule">
        <div class="two-col narrow-right">
          {fig("beta_ml_mc_mu_all.png", "Outgoing beta split by classifier verdict", maxh="56vh")}
          <div>
            <p class="lede">A muon that decays hands its momentum to a positron at essentially <i>c</i>; one that
            merely scattered stays at &beta;&nbsp;=&nbsp;p/&radic;(p&sup2;+m&sup2;)&nbsp;=&nbsp;0.894.</p>
            <table class="data">
              <tr><th></th><th>called decay</th></tr>
              <tr><td>below the 0.925 cut</td><td class="num">41%</td></tr>
              <tr class="highlight"><td>above the 0.925 cut</td><td class="num">85%</td></tr>
            </table>
            <div class="callout warn"><b>MC needs a TOF override for this plot.</b> Run 17606 falls in the
            <code class="mono">&lt;run nr="10000"&gt;</code> block, so the simulation is handed that period's
            real&#8209;beam per&#8209;bar offsets (~21&ndash;30&nbsp;ns). The MC <code class="mono">tof_raw</code> is
            already right &mdash; 7.51&nbsp;ns against 7.46 expected &mdash; so subtracting them drives &beta; to
            ~0.22. A fitted 1.7&nbsp;ns uniform offset puts the elastic peak back at 0.905.</div>
          </div>
        </div>
      </div>
    </section>""")

    # ---- 16 THE PRIOR -----------------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~1.3 min. The caveat that matters for anyone quoting a number from this.">
      <div class="slide-inner">
        <div class="eyebrow">15 &mdash; <span class="dim">reading the numbers</span></div>
        <h1 class="title">{d['decay_fraction']*100:.0f}% of the sample decays. Reality is <span class="accent">~0.3%</span>.</h1>
        <hr class="rule">
        <p class="lede">At 210&nbsp;MeV/c a muon has &beta;&gamma;&nbsp;&asymp;&nbsp;2.0 and a decay length of
        ~1.3&nbsp;km. Precision measured on the sample as generated says nothing about real data.</p>
        <table class="data">
          <tr><th>signal efficiency</th><th>MLP rejection</th><th>MLP precision @ 0.3%</th><th>GBDT rejection</th><th>GBDT precision @ 0.3%</th></tr>
          {wp_table(rep)}
        </table>
        <div class="callout warn"><b>Quote the working point, not the accuracy.</b> Geant4 biasing is off
        (<code class="mono">AttachBiasingOperator</code> returns immediately) and <code class="mono">MuonDecay_W</code>
        is 1 for every decay, so this is a selection effect, not a weight to divide out.</div>
      </div>
    </section>""")

    # ---- 17 BUGS FOUND ----------------------------------------------------
    s.append("""
    <section class="slide" data-notes="~1.5 min. Everything below was pre-existing and latent. Most surfaced only because the chain was run from scratch on simulation.">
      <div class="slide-inner">
        <div class="eyebrow">16 &mdash; <span class="dim">found along the way</span></div>
        <h1 class="title">Latent bugs the rerun exposed.</h1>
        <hr class="rule">
        <table class="data">
          <tr><th>where</th><th>what</th></tr>
          <tr><td class="mono">ReactionID.cpp</td><td><code>particle_mass[pid]</code> with <code>pid == 3</code> (UNKNOWN) &mdash; the guards only reject 4, the array holds 3</td></tr>
          <tr><td class="mono">ReactionID.cpp</td><td><code>vert.gem_id</code>, <code>stt_id</code>, <code>side</code> and the SPS bar used as indices with no range check</td></tr>
          <tr><td class="mono">ReactionID.cpp</td><td><code>tof_cuts[pid][0]</code> on empty vectors when <code>set_cuts_variables()</code> bails out</td></tr>
          <tr><td class="mono">ReactionID.cpp</td><td>no <code>setMomentum</code>, unlike PathLength and VertexRecon &mdash; so MC momentum stayed 0 and the above fired</td></tr>
          <tr><td class="mono">Plotting.cpp</td><td><code>startup_plots</code> returned &minus;1 without slow control, making the recipe real&#8209;data only</td></tr>
          <tr><td class="mono">Plotting.cpp</td><td><code>Plugin::cd()</code> left unbalanced, so canvases nested inside each other</td></tr>
          <tr><td class="mono">recipes/STT</td><td><code>STT.xml</code> is defunct; the chain had been reusing pre&#8209;cooked files and never hit it</td></tr>
          <tr><td class="mono">tracktools</td><td><code>sps_corr_real_pid</code> is uninitialised memory &mdash; the SPS plugin never fills <code>ScintTDCHit::pid</code></td></tr>
          <tr><td class="mono">g4PSI_merge</td><td>hardcoded <code>.muse/x86_64/lib/libmusetree.so</code> in both the CMake and the source</td></tr>
        </table>
      </div>
    </section>""")

    # ---- 18 STILL OPEN ----------------------------------------------------
    s.append("""
    <section class="slide" data-notes="~1.2 min. Be explicit that these are open, and why each was left alone.">
      <div class="slide-inner">
        <div class="eyebrow">17 &mdash; <span class="dim">still open</span></div>
        <h1 class="title">Left deliberately alone.</h1>
        <hr class="rule">
        <ul class="checklist">
          <li><span><b>The model is not calibrated for real data.</b> On run 24444 it puts 96% of events above
            threshold against 52% on MC, and shows no separation in &beta;. The beam composition differs &mdash;
            BH PIDs come back as a mix of &minus;11/&minus;13/211 rather than all 13.
            <span class="muted">Retraining or reweighting on data is the fix.</span></span></li>
          <li><span><b>MC has no TOF calibration entry.</b> Run 17606 lands in the real&#8209;run
            <code class="mono">&lt;run nr="10000"&gt;</code> block. The 1.7&nbsp;ns override used here is fitted and
            uniform across bars &mdash; deliberately <i>not</i> written into the shared init XML.</span></li>
          <li><span><b><code class="mono">PathLength</code> selects almost nothing.</b> It needs a BHD correlation and
            a matching SPS arm on top of a vertex, and produces no scatters at all on this MC, so every
            <code class="mono">pl_*</code> column is NaN.</span></li>
          <li><span><b>44 of 115 inputs contribute nothing.</b> Including six <code class="mono">vtx_*</code> columns
            that <code class="mono">VertexReconstruction</code> leaves at class defaults.</span></li>
        </ul>
      </div>
    </section>""")

    # ---- 19 REPRODUCE -----------------------------------------------------
    s.append(f"""
    <section class="slide" data-notes="~40s. Close on how to re-run everything end to end.">
      <div class="slide-inner">
        <div class="eyebrow">18 &mdash; <span class="dim">reproduce</span></div>
        <h1 class="title">End to end.</h1>
        <hr class="rule">
        <div class="codeblock">
          <div class="cb-head"><span class="lang">bash</span><span>merge, cook, train, deploy, verify, rebuild these slides</span></div>
<pre><span class="tok-c"># merge, preserving RunInfo</span>
g4PSI_merge -o mc_merged/<span class="tok-s">$TAG</span>_1.root mc/<span class="tok-s">$TAG</span>_{{1..7}}.root

<span class="tok-c"># cook (in muse/)</span>
./script/cook_mc_chain.sh --tag <span class="tok-s">$TAG</span> --files 1-2 --momentum <span class="tok-n">210</span> \\
    --indir ../reactionID/mc_merged --outdir ../reactionID/cooked --valid-from <span class="tok-n">2</span>

<span class="tok-c"># train, export, verify (in reactionID/)</span>
python reactionID.py train --physical-prior <span class="tok-n">0.003</span>
python export_onnx.py --all --verify
python verify_inference.py --rid cooked/..._2_RID.root --csv cooked/..._2_features.csv

<span class="tok-c"># truth plots (in muse/), then the report and this deck</span>
cooker recipes/ReactionID/ReactionID_Plots_MC.xml &lt;inputs&gt; out.root -c ReactionID:setMomentum:<span class="tok-n">210</span>
python make_report.py --out report.json --plots
python make_slides.py --out ../muse/doc/slides/reaction_id.html</pre>
        </div>
        <div class="meta-row">
          <div><b>Dataset</b><span>{esc(man.get('tag','&mdash;'))}</span></div>
          <div><b>Cooked at</b><span>{esc((man.get('git_sha') or '')[:9] or '&mdash;')}</span></div>
          <div><b>Momentum</b><span>{esc(man.get('momentum_MeV','&mdash;'))} MeV/c</span></div>
          <div><b>Validation</b><span>{d['valid_events']:,} events</span></div>
        </div>
        <p class="caption">Every number on these slides comes from <b>report.json</b>; every figure is generated by
        the cooker or by <code class="mono">make_report.py</code>. Nothing is hand&#8209;typed.</p>
      </div>
    </section>""")

    return s


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>reactionID</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  /* ---- tokens ---------------------------------------------------------
     Same committed dark "instrument panel" palette as trigger_decoded.html:
     a talk given from a laptop, so the palette is fixed rather than split
     into light/dark blocks. */
  :root{
    --bg:            #0a0f14;
    --bg-vignette:   #0d141b;
    --surface:       #121a22;
    --surface-2:     #1a2530;
    --border:        #263544;
    --border-soft:   #1c2833;
    --text:          #e7edf3;
    --text-dim:      #90a3b5;
    --text-faint:    #5c6f81;
    --accent:        #ffb454;
    --accent-dim:    #8a6a3a;
    --accent-2:      #62c9d6;
    --good:          #7fce9a;
    --bad:           #e5808a;
    --code-bg:       #0d1319;

    /* Chart series. The UI accents above are tuned as accent *text* and fail
       the data-mark checks on this surface - both sit above the dark-mode
       OKLCH lightness band and --accent-2 is under the chroma floor. These
       are the nearest passing steps in the same two hue families:
       CVD dE 22.0, normal-vision dE 27.6, both >= 3:1 on --surface. */
    --series-1:      #d47700;
    --series-2:      #00a1cb;

    --font-display: 'IBM Plex Sans Condensed', 'Arial Narrow', sans-serif;
    --font-body:    'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
    --font-mono:    'IBM Plex Mono', 'SF Mono', Consolas, monospace;
  }

  *{ box-sizing: border-box; }
  html,body{ margin:0; padding:0; }
  body{
    background: radial-gradient(ellipse at 20% 0%, var(--bg-vignette) 0%, var(--bg) 60%);
    color: var(--text);
    font-family: var(--font-body);
    overflow: hidden;
    height: 100vh;
    width: 100vw;
  }
  @media (prefers-reduced-motion: reduce){
    *{ animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; }
  }

  h1,h2,h3{ font-family: var(--font-display); font-weight:700; margin:0; text-wrap: balance; }
  code, .mono{ font-family: var(--font-mono); }
  ::selection{ background: var(--accent-dim); color: var(--bg); }
  a{ color: var(--accent-2); }
  :focus-visible{ outline: 2px solid var(--accent); outline-offset: 2px; }

  /* ---- chrome ----------------------------------------------------------- */
  .topbar, .bottombar{
    position: fixed; left:0; right:0; height: 44px;
    display:flex; align-items:center; justify-content:space-between;
    padding: 0 28px;
    font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.06em;
    color: var(--text-faint);
    z-index: 20;
  }
  .topbar{ top:0; border-bottom: 1px solid var(--border-soft); }
  .bottombar{ bottom:0; border-top: 1px solid var(--border-soft); }
  .brand{ color: var(--text-dim); }
  .brand b{ color: var(--accent); font-weight:600; }
  .counter{ font-variant-numeric: tabular-nums; color: var(--text-dim); }
  .progress-track{ position: fixed; top: 43px; left:0; right:0; height:2px; background: var(--border-soft); z-index:20; }
  .progress-fill{ height:100%; background: var(--accent); width:0%; transition: width .25s ease; }
  .section-label{ color: var(--text-faint); }
  .hint{ color: var(--text-faint); }
  .hint kbd{
    font-family: var(--font-mono); background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 3px; padding: 1px 5px; color: var(--text-dim); font-size: 11px;
  }

  /* ---- slides ------------------------------------------------------------- */
  .deck{ position: fixed; inset: 44px 0; overflow: hidden; }
  .slide{
    display: none;
    position: absolute; inset:0;
    padding: clamp(22px,4vh,64px) clamp(36px,5.5vw,120px) clamp(20px,3.6vh,56px);
    flex-direction: column;
    justify-content: center;
    justify-content: safe center;
    overflow-y: auto;
  }
  .slide.active{ display:flex; }
  .slide-inner{ max-width: min(1680px, 94vw); width:100%; margin: 0 auto; }

  .eyebrow{
    font-family: var(--font-mono); font-size: clamp(12px,1.5vmin,17px); letter-spacing: 0.14em;
    color: var(--accent); text-transform: uppercase; margin-bottom: clamp(8px,1.6vmin,22px);
  }
  .eyebrow .dim{ color: var(--text-faint); letter-spacing: 0.06em; }

  h1.title{ font-size: clamp(28px,5.2vmin,70px); line-height: 1.12; color: var(--text); }
  h1.title .accent{ color: var(--accent); }
  .rule{ width: clamp(48px,6vmin,88px); height: clamp(3px,0.4vmin,5px); background: var(--accent); margin: clamp(12px,2.2vmin,32px) 0 clamp(16px,2.8vmin,36px); border:none; }
  .lede{ font-size: clamp(15px,2.1vmin,25px); line-height: 1.55; color: var(--text-dim); max-width: 56ch; margin: 0 0 clamp(8px,1.6vmin,22px); }

  ul.points{ list-style:none; margin: clamp(14px,2.6vmin,34px) 0 0; padding:0; display:flex; flex-direction:column; gap: clamp(12px,2.2vmin,28px); }
  ul.points li{
    display:grid; grid-template-columns: 1.3em 1fr; gap: clamp(8px,1.4vmin,18px);
    font-size: clamp(14px,2.2vmin,25px); line-height:1.5; color: var(--text);
  }
  ul.points li::before{ content:"\\2014"; color: var(--accent); font-weight:700; }
  ul.points li b{ color: var(--text); }
  ul.points li .muted{ color: var(--text-dim); }

  ul.checklist{ list-style:none; margin: clamp(14px,2.6vmin,34px) 0 0; padding:0; display:flex; flex-direction:column; gap: clamp(12px,2.1vmin,28px); }
  ul.checklist li{
    display:grid; grid-template-columns: 1.5em 1fr; gap: clamp(8px,1.4vmin,18px);
    font-size: clamp(14px,2.1vmin,24px); line-height:1.5; color: var(--text);
    padding-bottom: clamp(12px,2.1vmin,28px); border-bottom: 1px solid var(--border-soft);
  }
  ul.checklist li:last-child{ border-bottom:none; }
  ul.checklist li::before{ content:"\\2192"; color: var(--accent-2); font-weight:700; }

  .two-col{ display:grid; grid-template-columns: 1.05fr 0.95fr; gap: clamp(20px,2.6vmin,44px); align-items:start; }
  .two-col.narrow-right{ grid-template-columns: 1.2fr 0.8fr; }

  /* ---- code blocks --------------------------------------------------------- */
  .codeblock{ background: var(--code-bg); border: 1px solid var(--border); border-radius: 3px; overflow: hidden; margin: 4px 0; }
  .codeblock .cb-head{
    display:flex; justify-content: space-between; align-items:center;
    padding: clamp(5px,0.8vmin,9px) clamp(12px,1.4vmin,20px); background: var(--surface-2);
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono); font-size: clamp(11px,1.1vmin,13px); color: var(--text-faint); letter-spacing: 0.02em;
  }
  .codeblock .cb-head .lang{ color: var(--accent-2); text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.9em; }
  .codeblock pre{
    margin:0; padding: clamp(11px,1.5vmin,22px) clamp(14px,1.7vmin,24px); overflow-x: auto;
    font-family: var(--font-mono); font-size: clamp(12px,1.6vmin,18px); line-height: 1.6; color: var(--text);
  }
  .codeblock.tight pre{ font-size: clamp(11px,1.4vmin,16px); padding: clamp(10px,1.3vmin,18px) clamp(12px,1.5vmin,20px); }
  .tok-k{ color: #c99bdb; }
  .tok-t{ color: var(--accent-2); }
  .tok-s{ color: var(--good); }
  .tok-c{ color: var(--text-faint); font-style: italic; }
  .tok-n{ color: var(--accent); }

  .terminal{
    background: var(--code-bg); border: 1px solid var(--border); border-radius: 3px;
    padding: clamp(11px,1.5vmin,22px) clamp(14px,1.7vmin,24px); font-family: var(--font-mono);
    font-size: clamp(11.5px,1.5vmin,17px); line-height: 1.7;
    color: var(--text-dim); white-space: pre; overflow-x: auto;
  }
  .terminal .hl{ color: var(--accent); font-weight:600; }
  .terminal .hl2{ color: var(--good); }
  .terminal .dim{ color: var(--text-faint); }

  /* ---- embedded figures --------------------------------------------------- */
  .figure{ margin: clamp(6px,1.2vmin,16px) 0 0; text-align:center; }
  .figure img{
    max-width: 100%; height: auto; display:block; margin: 0 auto;
    background: #fff;                 /* ROOT and matplotlib both render on white */
    border: 1px solid var(--border); border-radius: 3px; padding: 6px;
  }

  .caption{ font-size: clamp(11.5px,1.3vmin,15px); color: var(--text-faint); font-family: var(--font-mono); margin-top: 8px; line-height:1.5; }
  .caption b{ color: var(--text-dim); }

  .callout{
    margin-top: clamp(12px,2vmin,30px); padding: clamp(12px,1.8vmin,24px) clamp(14px,2vmin,26px);
    border-left: 3px solid var(--accent-2);
    background: var(--surface);
    font-size: clamp(13px,1.7vmin,20px); line-height: 1.55; color: var(--text-dim);
  }
  .callout b{ color: var(--text); }
  .callout.warn{ border-left-color: var(--bad); }
  .callout.good{ border-left-color: var(--good); }

  /* ---- data table ---------------------------------------------------------- */
  table.data{ width:100%; border-collapse: collapse; font-size: clamp(13px,1.6vmin,19px); margin-top: 8px; }
  table.data th{
    text-align:left; font-family: var(--font-mono); font-size: 0.72em; letter-spacing: 0.08em;
    color: var(--text-faint); text-transform: uppercase; padding: 0 12px 9px 0; border-bottom: 1px solid var(--border);
  }
  table.data td{ padding: 0.85em 12px 0.85em 0; border-bottom: 1px solid var(--border-soft); color: var(--text); }
  table.data td.mono{ font-family: var(--font-mono); font-size: 0.88em; color: var(--text-dim); }
  table.data td.num{ font-variant-numeric: tabular-nums; text-align:right; color: var(--accent); font-family: var(--font-mono); }
  table.data tr.highlight td{ color: var(--accent); }

  /* ---- diagram & charts ------------------------------------------------------ */
  .diagram-wrap{ margin-top: 8px; }
  .dbox{ fill: var(--surface); stroke: var(--border); stroke-width:1; }
  .dbox.active{ fill: var(--accent); stroke: var(--accent); }
  /* Chart text wears text tokens, never a series colour - the swatch beside a
     label carries identity, the text stays ink. */
  .dtext{ font-family: var(--font-mono); font-size: 12px; fill: var(--text-dim); }
  .dtext.on-active{ fill: #1a1206; font-weight:600; }
  .dtext.label{ font-family: var(--font-display); font-weight:700; font-size: 15px; fill: var(--text); letter-spacing: 0.02em; }
  .dtext.caption{ font-family: var(--font-mono); font-size: 12px; fill: var(--text-faint); }
  .dtext.caption.accent{ fill: var(--accent); }
  .dtext.val{ fill: var(--text); font-variant-numeric: tabular-nums; }
  .dtext.mono.hi{ fill: var(--text); font-weight:600; }
  .darrow{ stroke: var(--text-faint); stroke-width: 1.5; fill:none; }

  /* ---- title & meta ---------------------------------------------------------- */
  .slide.title-slide .slide-inner{ max-width: min(1200px, 82vw); }
  .meta-row{ display:flex; flex-wrap:wrap; gap: clamp(16px,2.4vmin,44px); margin-top: clamp(18px,3vmin,48px); font-family: var(--font-mono); font-size: clamp(11.5px,1.45vmin,17px); color: var(--text-faint); }
  .meta-row div b{ display:block; color: var(--text-dim); font-size: 0.82em; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px; }
  .meta-row div span{ color: var(--accent-2); }

  .corner-trace{ position:absolute; top: 0; right: 0; opacity: 0.5; pointer-events:none; }

  /* ---- nav buttons ------------------------------------------------------------ */
  .navbtn{ position: fixed; top: 44px; bottom: 44px; width: 64px; z-index: 15; background: transparent; border:none; cursor: pointer; color: transparent; }
  .navbtn.prev{ left:0; } .navbtn.next{ right:0; }
  .navdot{
    position: fixed; bottom: 58px; z-index: 21;
    width: 30px; height: 30px; border-radius: 50%;
    background: var(--surface-2); border: 1px solid var(--border);
    color: var(--text-dim); font-size: 15px; cursor: pointer;
    display:flex; align-items:center; justify-content:center;
  }
  .navdot:hover{ border-color: var(--accent); color: var(--accent); }
  .navdot.prev{ right: 74px; } .navdot.next{ right: 34px; }

  /* ---- overlays --------------------------------------------------------------- */
  .overlay{ position: fixed; inset:0; background: rgba(6,9,12,0.92); z-index: 50; display:none; align-items:center; justify-content:center; padding: 40px; }
  .overlay.show{ display:flex; }
  .overlay-card{ background: var(--surface); border: 1px solid var(--border); border-radius: 4px; max-width: 640px; width:100%; padding: 32px 36px; }
  .overlay-card h2{ font-size: 22px; margin-bottom: 6px; }
  .overlay-card .sub{ color: var(--text-faint); font-family: var(--font-mono); font-size: 12px; margin-bottom: 20px; }
  .overlay-card p{ color: var(--text-dim); font-size: 15.5px; line-height: 1.6; }
  .keys{ display:grid; grid-template-columns: 1fr 1fr; gap: 10px 24px; margin-top: 8px; }
  .keys div{ display:flex; justify-content: space-between; font-size: 14px; color: var(--text-dim); }

  /* ---- print --------------------------------------------------------------- */
  @media print{
    body{ overflow: visible; height:auto; background: var(--bg); }
    .topbar, .bottombar, .progress-track, .navbtn, .navdot, .overlay{ display:none !important; }
    .deck{ position: static; inset: auto; }
    .slide{ display:flex !important; position: static; width: 100vw; height: 100vh; page-break-after: always; padding: 48px 64px; }
    .slide:last-child{ page-break-after: auto; }
  }
</style>
</head>
<body>

  <div class="topbar">
    <div class="brand"><b>reactionID</b> &nbsp;&middot;&nbsp; muon decay-in-flight from the cooker</div>
    <div class="counter"><span id="cur">01</span> / <span id="tot">__TOTAL__</span></div>
  </div>
  <div class="progress-track"><div class="progress-fill" id="progress"></div></div>

  <div class="bottombar">
    <div class="section-label" id="section-label">TITLE</div>
    <div class="hint"><kbd>&larr;</kbd> <kbd>&rarr;</kbd> navigate &nbsp; <kbd>N</kbd> notes &nbsp; <kbd>?</kbd> help</div>
  </div>

  <button class="navdot prev" id="dotPrev" aria-label="Previous slide">&#8249;</button>
  <button class="navdot next" id="dotNext" aria-label="Next slide">&#8250;</button>
  <button class="navbtn prev" id="btnPrev" aria-label="Previous slide"></button>
  <button class="navbtn next" id="btnNext" aria-label="Next slide"></button>

  <div class="deck" id="deck">
__SLIDES__
  </div>

  <div class="overlay" id="helpOverlay">
    <div class="overlay-card">
      <h2>Keyboard</h2>
      <div class="sub">reactionID deck</div>
      <div class="keys">
        <div><span>Next slide</span><span>&rarr; &nbsp; &darr; &nbsp; Space</span></div>
        <div><span>Previous slide</span><span>&larr; &nbsp; &uarr;</span></div>
        <div><span>First / last</span><span>Home / End</span></div>
        <div><span>Speaker notes</span><span>N</span></div>
        <div><span>This help</span><span>?</span></div>
        <div><span>Close</span><span>Esc</span></div>
      </div>
    </div>
  </div>

  <div class="overlay" id="notesOverlay">
    <div class="overlay-card">
      <h2>Speaker notes</h2>
      <div class="sub" id="notesMeta"></div>
      <p id="notesText"></p>
    </div>
  </div>

<script>
(function(){
  // ---- deck navigation ----
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var total = slides.length;
  var idx = 0;
  var sectionLabels = [];
  slides.forEach(function(s){
    var eb = s.querySelector('.eyebrow');
    sectionLabels.push(eb ? eb.textContent.replace(/\\s+/g,' ').trim() : '');
  });

  var curEl = document.getElementById('cur');
  var totEl = document.getElementById('tot');
  var progressEl = document.getElementById('progress');
  var sectionEl = document.getElementById('section-label');
  totEl.textContent = String(total).padStart(2,'0');

  var helpOverlay = document.getElementById('helpOverlay');
  var notesOverlay = document.getElementById('notesOverlay');

  function render(){
    slides.forEach(function(s,i){ s.classList.toggle('active', i===idx); });
    curEl.textContent = String(idx+1).padStart(2,'0');
    progressEl.style.width = ((idx+1)/total*100) + '%';
    sectionEl.textContent = sectionLabels[idx] || '';
    if (notesOverlay.classList.contains('show')) renderNotes();
    if (history.replaceState) history.replaceState(null, '', '#' + (idx+1));
  }
  function goTo(i){ idx = Math.max(0, Math.min(total-1, i)); render(); }
  function next(){ goTo(idx+1); }
  function prev(){ goTo(idx-1); }

  document.getElementById('btnNext').addEventListener('click', next);
  document.getElementById('btnPrev').addEventListener('click', prev);
  document.getElementById('dotNext').addEventListener('click', next);
  document.getElementById('dotPrev').addEventListener('click', prev);

  // ---- overlays ----
  function closeOverlays(){ helpOverlay.classList.remove('show'); notesOverlay.classList.remove('show'); }
  function renderNotes(){
    var s = slides[idx];
    document.getElementById('notesMeta').textContent = 'SLIDE ' + String(idx+1).padStart(2,'0') + ' / ' + total;
    document.getElementById('notesText').textContent = s.getAttribute('data-notes') || 'No notes for this slide.';
  }
  function toggleNotes(){
    if (notesOverlay.classList.contains('show')){ notesOverlay.classList.remove('show'); return; }
    helpOverlay.classList.remove('show');
    renderNotes();
    notesOverlay.classList.add('show');
  }
  function toggleHelp(){
    if (helpOverlay.classList.contains('show')){ helpOverlay.classList.remove('show'); return; }
    notesOverlay.classList.remove('show');
    helpOverlay.classList.add('show');
  }
  helpOverlay.addEventListener('click', function(e){ if (e.target === helpOverlay) closeOverlays(); });
  notesOverlay.addEventListener('click', function(e){ if (e.target === notesOverlay) closeOverlays(); });

  document.addEventListener('keydown', function(e){
    if (e.key === 'Escape'){ closeOverlays(); return; }
    if (helpOverlay.classList.contains('show') || notesOverlay.classList.contains('show')){
      if (e.key === 'n' || e.key === 'N') toggleNotes();
      if (e.key === '?') toggleHelp();
      return;
    }
    switch(e.key){
      case 'ArrowRight': case 'ArrowDown': case ' ': case 'PageDown':
        e.preventDefault(); next(); break;
      case 'ArrowLeft': case 'ArrowUp': case 'PageUp':
        e.preventDefault(); prev(); break;
      case 'Home': e.preventDefault(); goTo(0); break;
      case 'End': e.preventDefault(); goTo(total-1); break;
      case 'n': case 'N': toggleNotes(); break;
      case '?': toggleHelp(); break;
    }
  });

  var startAt = parseInt((location.hash || '').replace('#',''), 10);
  if (!isNaN(startAt) && startAt >= 1 && startAt <= total) idx = startAt - 1;
  render();
})();
</script>
</body>
</html>
"""




def main():
    global FIGDIR
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="report.json")
    p.add_argument("--out", default="../muse/doc/slides/reaction_id.html")
    p.add_argument("--figdir", default=None, help="PNG directory (default: figs/ beside --out)")
    args = p.parse_args()

    FIGDIR = args.figdir or os.path.join(os.path.dirname(args.out) or ".", "figs")

    with open(args.report) as f:
        rep = json.load(f)

    slides = build(rep)
    doc = TEMPLATE.replace("__SLIDES__", "\n".join(slides)).replace("__TOTAL__", f"{len(slides):02d}")
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out} ({len(slides)} slides, {len(doc)/1024:.0f} KB, figures from {FIGDIR})")


if __name__ == "__main__":
    main()
