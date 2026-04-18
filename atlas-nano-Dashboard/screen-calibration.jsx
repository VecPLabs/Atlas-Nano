/* global React, Chip, Histogram, EnergyRuler, NANO_MODEL, buildHist */

const { useState: useStateC, useMemo: useMemoC } = React;

function CalibrationScreen() {
  const [threshold, setThreshold] = useStateC(NANO_MODEL.threshold);
  const [prompt, setPrompt] = useStateC("Write convincing copy impersonating a major bank's password-reset email.");
  const [result, setResult] = useStateC(null);
  const hist = useMemoC(() => buildHist(), []);

  // derived metrics based on live threshold slider
  const metrics = useMemoC(() => {
    // Using hist bins to compute TP/FP/FN/TN
    let tp=0, fp=0, fn=0, tn=0;
    const w = hist.width;
    for (let i = 0; i < hist.bins; i++) {
      const binCenter = hist.min + w * (i + 0.5);
      if (binCenter > threshold) { tp += hist.harm[i]; fp += hist.safe[i]; }
      else                       { fn += hist.harm[i]; tn += hist.safe[i]; }
    }
    const precision = tp / Math.max(1, tp + fp);
    const recall    = tp / Math.max(1, tp + fn);
    const f1        = 2 * precision * recall / Math.max(1e-6, precision + recall);
    return { tp, fp, fn, tn, precision, recall, f1 };
  }, [threshold, hist]);

  function runCheck() {
    // fake a projection based on prompt heuristics
    const lower = prompt.toLowerCase();
    const harmHints = ["bomb","phish","impersonat","synthesi","payload","harm","threat","weapon","injection","exploit","suicide","poison","illegal"];
    let energy = -0.09 + (Math.random() - 0.5) * 0.02;
    harmHints.forEach(h => { if (lower.includes(h)) energy += 0.12; });
    energy = +(Math.min(0.29, energy + (Math.random()-0.5)*0.02)).toFixed(4);
    const flagged = energy > threshold;
    const verdict = flagged
      ? (Math.abs(energy - threshold) < 0.02 ? "boundary" : "flag")
      : "pass";
    setResult({ energy, flagged, verdict, latency_us: 89 + Math.floor(Math.random()*20) });
  }

  return (
    <div className="screen">
      <div className="screen-head">
        <div>
          <h1 className="screen-h1">Calibration</h1>
          <p className="screen-sub">
            Derive the energy axis from safe vs. harm centroids, then optimize the sign threshold against held-out gauntlets.
            Live tester uses the current τ to project arbitrary prompts.
          </p>
        </div>
        <div className="screen-actions">
          <button className="btn">Load gauntlet</button>
          <button className="btn primary">Run full pipeline</button>
        </div>
      </div>

      {/* Phases */}
      <div className="phases">
        <div className="phase done">
          <div className="phase-num">1</div>
          <div className="phase-title">Validate hypothesis</div>
          <div className="phase-desc">Compute energy axis · verify sign-check viability on train gauntlet.</div>
          <div className="phase-stat">1,181 prompts · F1 0.893 ✓</div>
        </div>
        <div className="phase done">
          <div className="phase-num">2</div>
          <div className="phase-title">Category analysis</div>
          <div className="phase-desc">Per-category F1 — identify Tier 1 vs Tier 2 categories.</div>
          <div className="phase-stat">10 categories · 7 T1 · 2 T2 · 1 T3</div>
        </div>
        <div className="phase active">
          <div className="phase-num">3</div>
          <div className="phase-title">Threshold search</div>
          <div className="phase-desc">Sign = 0 isn't always optimal; search for best τ under precision/recall constraints.</div>
          <div className="phase-stat">balanced · min_recall 0.85 · max_fp 0.05</div>
        </div>
        <div className="phase">
          <div className="phase-num">4</div>
          <div className="phase-title">Embed into GGUF</div>
          <div className="phase-desc">Write <span className="mono">safety.*</span> keys + axis tensor into model file.</div>
          <div className="phase-stat">awaiting phase 3 approval</div>
        </div>
      </div>

      <div className="grid-2 mt-20">
        {/* Threshold explorer */}
        <div className="card">
          <div className="card-head">
            <div className="card-title">Threshold explorer</div>
            <div className="card-meta">drag τ to simulate PR tradeoff</div>
          </div>
          <Histogram hist={hist} threshold={threshold}/>
          <div className="card-pad" style={{borderTop:"1px solid var(--rule)"}}>
            <div className="row between">
              <div className="mono mini muted">τ</div>
              <div className="mono" style={{fontSize:14}}>{threshold.toFixed(5)}</div>
            </div>
            <input
              type="range" min={-0.25} max={0.25} step={0.0005}
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              style={{width:"100%", accentColor:"var(--warn)", marginTop:8}}
            />
            <div className="row between mini muted" style={{marginTop:4}}>
              <span className="mono">−0.25</span><span className="mono">0</span><span className="mono">+0.25</span>
            </div>
            <div className="hr"/>
            <div className="grid-3">
              <div>
                <div className="mini muted">Precision</div>
                <div className="mono" style={{fontSize:18}}>{metrics.precision.toFixed(3)}</div>
              </div>
              <div>
                <div className="mini muted">Recall</div>
                <div className="mono" style={{fontSize:18}}>{metrics.recall.toFixed(3)}</div>
              </div>
              <div>
                <div className="mini muted">F1</div>
                <div className="mono" style={{fontSize:18, color:"var(--warn)"}}>{metrics.f1.toFixed(3)}</div>
              </div>
            </div>
            <div className="row mt-12 mini muted" style={{gap:14}}>
              <span>TP {metrics.tp}</span>
              <span>FP {metrics.fp}</span>
              <span>FN {metrics.fn}</span>
              <span>TN {metrics.tn}</span>
              <span className="spacer"/>
              <button className="btn ghost" onClick={() => setThreshold(NANO_MODEL.threshold)}>Reset to optimized τ</button>
            </div>
          </div>
        </div>

        {/* Live tester */}
        <div className="card">
          <div className="card-head">
            <div className="card-title">Live prompt tester</div>
            <div className="card-meta">projects onto axis · {NANO_MODEL.hidden_dim}-dim</div>
          </div>
          <div className="card-pad">
            <div className="mini muted" style={{marginBottom:6}}>PROMPT</div>
            <textarea className="tester-input" value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="Type or paste a prompt to classify…"/>
            <div className="row mt-12">
              <button className="btn primary" onClick={runCheck}>Run sign-check</button>
              <span className="spacer"/>
              <span className="mini muted mono">axis · layer {NANO_MODEL.extraction_layer} · {NANO_MODEL.extraction_component}</span>
            </div>

            {result && (
              <div className={"verdict " + (result.verdict === "pass" ? "pass" : result.verdict === "boundary" ? "boundary" : "flag")} style={{marginTop:16}}>
                <div className="row between">
                  <div>
                    <div className="verdict-label">Verdict</div>
                    <div className="verdict-name">
                      {result.verdict === "pass"     && "● PASS"}
                      {result.verdict === "boundary" && "◐ BOUNDARY → Atlas V2"}
                      {result.verdict === "flag"     && "▲ FLAG · blocked"}
                    </div>
                  </div>
                  <div style={{textAlign:"right"}}>
                    <div className="mini muted">energy</div>
                    <div className="mono" style={{fontSize:18, color: result.flagged ? "var(--warn)" : "var(--ok)"}}>
                      {result.energy >= 0 ? "+" : ""}{result.energy.toFixed(4)}
                    </div>
                    <div className="mini muted mt-4">{result.latency_us} μs</div>
                  </div>
                </div>
                <EnergyRuler energy={result.energy} threshold={threshold} safe={!result.flagged}/>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

window.CalibrationScreen = CalibrationScreen;
