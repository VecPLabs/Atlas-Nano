/* global React, KPI, Chip, Spark, EnergyBar, EnergyRuler, Histogram, fmtTime, sparkData, buildHist, NANO_MODEL, CATEGORIES, seedStream */

const { useState, useEffect, useRef, useMemo } = React;

function MonitoringScreen({ streamOn }) {
  const [events, setEvents] = useState(() => seedStream(220));
  const [selectedId, setSelectedId] = useState(null);
  const [filter, setFilter] = useState("all"); // all | flagged | pass | boundary
  const [catFilter, setCatFilter] = useState("any");
  const counterRef = useRef(0);
  const histRef = useRef(buildHist());

  // Streaming: push a new event every ~1.4s when streamOn
  useEffect(() => {
    if (!streamOn) return;
    const iv = setInterval(() => {
      const pool = window.SAMPLE_PROMPTS;
      const s = pool[Math.floor(Math.random() * pool.length)];
      const jitter = (Math.random() - 0.5) * 0.06;
      const energy = +(s.energy + jitter).toFixed(4);
      const flagged = energy > NANO_MODEL.threshold;
      counterRef.current += 1;
      const ev = {
        id: "req_" + (2097152 + Date.now()).toString(16).slice(-6),
        ts: new Date().toISOString(),
        prompt: s.p, category: s.cat, energy, flagged,
        tier: flagged ? (Math.abs(energy - NANO_MODEL.threshold) < 0.03 ? "T2" : "T1") : "pass",
        verdict: flagged ? (Math.abs(energy - NANO_MODEL.threshold) < 0.03 ? "boundary" : "blocked") : "pass",
        model: "qwen3-4b-safety", tokens: 6 + Math.floor(Math.random()*40), latency_us: 80 + Math.floor(Math.random()*35),
        _new: true
      };
      setEvents(prev => [ev, ...prev].slice(0, 400));
    }, 1400);
    return () => clearInterval(iv);
  }, [streamOn]);

  const visible = useMemo(() => events.filter(e => {
    if (filter === "flagged"  && !e.flagged) return false;
    if (filter === "pass"     &&  e.flagged) return false;
    if (filter === "boundary" && e.verdict !== "boundary") return false;
    if (catFilter !== "any" && e.category !== catFilter) return false;
    return true;
  }), [events, filter, catFilter]);

  const selected = events.find(e => e.id === selectedId) || events[0];

  // stats
  const flagCount = events.filter(e => e.flagged).length;
  const flagRate = ((flagCount / events.length) * 100).toFixed(1);
  const boundary = events.filter(e => e.verdict === "boundary").length;
  const tier1 = events.filter(e => e.tier === "T1").length;
  const tier2 = events.filter(e => e.tier === "T2").length;

  // expose nav badge count
  useEffect(() => {
    const el = document.getElementById("nav-flag-count");
    if (el) el.textContent = String(flagCount);
  }, [flagCount]);

  return (
    <div className="screen">
      <div className="screen-head">
        <div>
          <h1 className="screen-h1">Live Monitoring</h1>
          <p className="screen-sub">
            Energy-axis sign-check at layer {NANO_MODEL.extraction_layer} on <span className="mono">{NANO_MODEL.base}</span>.
            Threshold <span className="mono">τ = {NANO_MODEL.threshold.toFixed(4)}</span>.
            Every token projected onto the 2,560-dim axis; only dot products cross the hot path.
          </p>
        </div>
        <div className="screen-actions">
          <button className="btn ghost">
            <svg viewBox="0 0 16 16" width="12" height="12"><rect x="2" y="6" width="12" height="4" fill="none" stroke="currentColor" strokeWidth="1.3"/><circle cx="6" cy="8" r="1" fill="currentColor"/></svg>
            Export (CSV)
          </button>
          <button className="btn">
            <span className={"dot " + (streamOn ? "ok" : "mute")} style={{marginRight:2}}/>
            {streamOn ? "Streaming" : "Paused"}
          </button>
          <button className="btn primary">Create incident</button>
        </div>
      </div>

      {/* KPI strip */}
      <div className="kpi-grid">
        <KPI label="Requests · 24h"  value="48,612" delta={+4.2} spark={sparkData(24, 1, 0.7)} />
        <KPI label="Flag rate"       value={flagRate} unit="%" delta={-1.1} spark={sparkData(24, 3, 0.4)} tone="accent"/>
        <KPI label="Boundary (→ T2)" value={boundary.toString().padStart(2,"0")} delta={+0.8} spark={sparkData(24, 5, 0.3)} />
        <KPI label="Mean latency"    value="94" unit="μs" delta={-0.4} spark={sparkData(24, 9, 0.2)} tone="ok" />
        <KPI label="Calibrated F1"   value={NANO_MODEL.f1.toFixed(3)} delta={+0} spark={sparkData(24, 11, 0.15)} tone="ok" />
      </div>

      {/* Tier flow */}
      <div className="tier-strip">
        <div className="tier">
          <span className="tier-name">Tier 1 · Nano</span>
          <span className="tier-title">Sign-check (in GGUF)</span>
          <span className="tier-desc">1 dot product + comparison per token, {NANO_MODEL.hidden_dim}-dim energy axis at layer {NANO_MODEL.extraction_layer}.</span>
          <div className="tier-bar"><div style={{width: "100%"}}/></div>
          <div className="tier-meta"><span>48,612 calls</span><span>· 94 μs</span><span>· {tier1} flagged</span></div>
        </div>
        <div className="tier-arrow"><svg viewBox="0 0 20 20" width="16" height="16"><path d="M4 10 L16 10 M11 5 L16 10 L11 15" fill="none" stroke="currentColor" strokeWidth="1.4"/></svg></div>
        <div className="tier">
          <span className="tier-name">Tier 2 · Atlas V2</span>
          <span className="tier-title">Triaxial fallback</span>
          <span className="tier-desc">Harm / Utility / Normality geometry on boundary cases only (|energy − τ| &lt; 0.03).</span>
          <div className="tier-bar"><div style={{width: "8%"}}/></div>
          <div className="tier-meta"><span>3,871 calls (8.0%)</span><span>· 2.1 ms</span><span>· {tier2} flagged</span></div>
        </div>
        <div className="tier-arrow"><svg viewBox="0 0 20 20" width="16" height="16"><path d="M4 10 L16 10 M11 5 L16 10 L11 15" fill="none" stroke="currentColor" strokeWidth="1.4"/></svg></div>
        <div className="tier">
          <span className="tier-name">Tier 3 · Review</span>
          <span className="tier-title">Multi-model consensus</span>
          <span className="tier-desc">Human + ensemble verification for the hardest boundary cases or appeals.</span>
          <div className="tier-bar"><div style={{width: "0.4%"}}/></div>
          <div className="tier-meta"><span>194 calls (0.4%)</span><span>· queue 7</span></div>
        </div>
      </div>

      {/* Two-col: stream + detail */}
      <div className="grid-2">
        {/* Stream */}
        <div className="card">
          <div className="card-head">
            <div className="row" style={{gap:12}}>
              <div className="card-title">Event Stream</div>
              <div className="seg" style={{marginLeft:8}}>
                {[["all","All"],["flagged","Flagged"],["boundary","Boundary"],["pass","Pass"]].map(([k,l]) => (
                  <button key={k} className={filter===k ? "on" : ""} onClick={() => setFilter(k)}>{l}</button>
                ))}
              </div>
            </div>
            <div className="card-meta">showing {visible.length} of {events.length}</div>
          </div>
          <div className="table-wrap" style={{maxHeight: 520, overflow:"auto"}}>
            <table className="data">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>ID</th>
                  <th>Prompt</th>
                  <th style={{textAlign:"right"}}>Energy</th>
                  <th>Tier</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {visible.slice(0, 100).map((e) => (
                  <tr key={e.id} className={(e._new ? "new-row " : "") + (selected && selected.id === e.id ? "selected" : "")}
                      onClick={() => setSelectedId(e.id)}>
                    <td className="col-time mono">{fmtTime(e.ts)}</td>
                    <td className="col-id mono">{e.id}</td>
                    <td className="col-prompt truncate">{e.prompt}</td>
                    <td className="col-energy"><EnergyBar energy={e.energy} threshold={NANO_MODEL.threshold}/></td>
                    <td>
                      {e.tier === "T1" && <Chip tone="warn">T1</Chip>}
                      {e.tier === "T2" && <Chip tone="info">T2</Chip>}
                      {e.tier === "pass" && <Chip tone="ok">pass</Chip>}
                    </td>
                    <td>
                      {e.verdict === "blocked" && <Chip tone="crit" dot>block</Chip>}
                      {e.verdict === "boundary" && <Chip tone="info" dot>boundary</Chip>}
                      {e.verdict === "pass" && <Chip tone="ok" dot>pass</Chip>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Detail */}
        <div className="detail">
          <div className="detail-head">
            <div>
              <div className="detail-id">{selected.id} · {fmtTime(selected.ts)}</div>
              <div style={{fontSize:14, fontWeight:600, marginTop:4}}>
                {selected.verdict === "blocked"  && "Flagged · blocked"}
                {selected.verdict === "boundary" && "Boundary case · routed to Atlas V2"}
                {selected.verdict === "pass"     && "Clean · passed"}
              </div>
            </div>
            <div className="row" style={{gap:6}}>
              <Chip tone={selected.flagged ? "warn" : "ok"} dot>{selected.flagged ? "harm axis" : "safe axis"}</Chip>
              <Chip tone="ghost">{selected.category}</Chip>
            </div>
          </div>
          <div className="detail-body">
            <div className="mini muted" style={{marginBottom:6}}>PROMPT</div>
            <div className="prompt-block">{selected.prompt}</div>

            <div className="mini muted mt-16" style={{marginBottom:4}}>ENERGY PROJECTION</div>
            <EnergyRuler energy={selected.energy} threshold={NANO_MODEL.threshold} safe={!selected.flagged}/>
            <div className="row between mini muted">
              <span>safe ← centroid ← benign</span>
              <span>harm centroid → threat →</span>
            </div>

            <div className="hr"/>

            <dl className="kv">
              <dt>Energy</dt>
              <dd style={{color: selected.flagged ? "var(--warn)" : "var(--ok)"}}>{selected.energy >= 0 ? "+" : ""}{selected.energy.toFixed(4)}</dd>
              <dt>Margin to τ</dt>
              <dd>{(selected.energy - NANO_MODEL.threshold).toFixed(4)}</dd>
              <dt>Extraction</dt>
              <dd>layer {NANO_MODEL.extraction_layer} / {NANO_MODEL.extraction_component}</dd>
              <dt>Model</dt>
              <dd>{NANO_MODEL.id}</dd>
              <dt>Tokens</dt>
              <dd>{selected.tokens}</dd>
              <dt>Latency</dt>
              <dd>{selected.latency_us} μs <span className="muted">· cold</span></dd>
            </dl>

            <div className="row mt-16" style={{gap:6}}>
              <button className="btn">Open in Atlas V2</button>
              <button className="btn">Replay</button>
              <button className="btn ghost">Add to gauntlet</button>
            </div>
          </div>
        </div>
      </div>

      {/* Histogram + categories */}
      <div className="grid-2 mt-20">
        <div className="card">
          <div className="card-head">
            <div className="card-title">Energy Distribution · last 1,500</div>
            <div className="card-meta">safe vs. harm · threshold overlay</div>
          </div>
          <Histogram hist={histRef.current} threshold={NANO_MODEL.threshold}/>
        </div>

        <div className="card">
          <div className="card-head">
            <div className="card-title">By Category · live 24h</div>
            <div className="card-meta">F1 · tier assignment</div>
          </div>
          <div className="card-pad">
            {CATEGORIES.slice(0, 8).map(c => (
              <div key={c.key} className="cat-row">
                <div className="cat-name">{c.name}</div>
                <div className={"cat-bar " + (c.f1 < 0.75 ? "crit" : c.f1 < 0.85 ? "warn" : "")}>
                  <div style={{width: (c.f1 * 100) + "%"}}/>
                </div>
                <div className="cat-val">{c.f1.toFixed(3)}</div>
                <div className="cat-tier">{c.tier} · {c.volume}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

window.MonitoringScreen = MonitoringScreen;
