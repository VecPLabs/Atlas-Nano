/* global React */
// Atlas Nano — shared primitive components

const { useMemo } = React;

// Sparkline
function Spark({ data, tone = "default", height = 28, width = 120 }) {
  const path = useMemo(() => {
    if (!data || !data.length) return { line:"", fill:"" };
    const min = Math.min(...data), max = Math.max(...data);
    const span = max - min || 1;
    const stepX = width / (data.length - 1);
    const pts = data.map((v, i) => [i * stepX, height - ((v - min) / span) * (height - 2) - 1]);
    const line = "M " + pts.map(p => p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" L ");
    const fill = line + ` L ${width} ${height} L 0 ${height} Z`;
    return { line, fill };
  }, [data, height, width]);
  const cls = "spark " + (tone === "accent" ? "accent" : tone === "ok" ? "ok" : "");
  return (
    <svg className={cls} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{width:"100%", height}}>
      <path className="fill" d={path.fill}/>
      <path className="line" d={path.line}/>
    </svg>
  );
}

// KPI cell
function KPI({ label, value, unit, delta, spark, tone }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}{unit && <span style={{fontSize:13, color:"var(--fg-2)", marginLeft:4}}>{unit}</span>}</div>
      <div className="kpi-sub">
        {delta != null && (
          <span className={"kpi-delta " + (delta >= 0 ? "up" : "down")}>
            {delta >= 0 ? "▲" : "▼"} {Math.abs(delta)}%
          </span>
        )}
        <span className="muted">{spark ? "last 24h" : ""}</span>
      </div>
      {spark && <div className="kpi-spark"><Spark data={spark} tone={tone || "default"} height={24}/></div>}
    </div>
  );
}

// Chip
function Chip({ tone = "ghost", children, dot, style }) {
  return (
    <span className={"chip " + tone} style={style}>
      {dot && <span className={"dot " + tone}/>}
      {children}
    </span>
  );
}

// Energy bar (used in table rows)
function EnergyBar({ energy, threshold = -0.0347, min = -0.3, max = 0.3 }) {
  const span = max - min;
  const pct  = Math.max(0, Math.min(1, (energy - min) / span));
  const tpct = Math.max(0, Math.min(1, (threshold - min) / span));
  const flagged = energy > threshold;
  const color = flagged ? "var(--warn)" : "var(--ok)";
  // draw a bar starting at threshold going to pct
  const left  = Math.min(pct, tpct) * 100;
  const width = Math.abs(pct - tpct) * 100;
  return (
    <div className="energy-cell">
      <span className="num muted" style={{color: flagged ? "var(--warn)" : "var(--fg-1)", width:58, textAlign:"right"}}>
        {energy >= 0 ? "+" : ""}{energy.toFixed(3)}
      </span>
      <div className="energy-bar" title={`energy=${energy}`}>
        <div className="fill" style={{left: left + "%", width: width + "%", background: color}}/>
        <div className="mid" style={{left: tpct * 100 + "%"}}/>
      </div>
    </div>
  );
}

// Energy ruler (detail view)
function EnergyRuler({ energy, threshold, min = -0.3, max = 0.3, safe }) {
  const span = max - min;
  const pct  = Math.max(0, Math.min(1, (energy - min) / span));
  const tpct = Math.max(0, Math.min(1, (threshold - min) / span));
  return (
    <div className={"energy-visual " + (safe ? "safe" : "")}>
      <div className="axis"/>
      <div className="zero"/>
      <div className="thresh" style={{ left: tpct * 100 + "%" }} title={`threshold=${threshold}`}/>
      <div className="marker" style={{ left: pct * 100 + "%" }}/>
      <div className="label l">−0.30</div>
      <div className="label r">+0.30</div>
      <div className="label" style={{left: tpct * 100 + "%", transform:"translateX(-50%)", color:"var(--warn)"}}>
        τ {threshold.toFixed(4)}
      </div>
    </div>
  );
}

// Histogram for safe/harm distributions
function Histogram({ hist, threshold }) {
  if (!hist) return null;
  const maxCount = Math.max(...hist.safe, ...hist.harm);
  const thresholdBin = (threshold - hist.min) / (hist.max - hist.min);
  return (
    <div className="hist-wrap">
      <div className="hist" style={{ gridTemplateColumns: `repeat(${hist.bins}, 1fr)` }}>
        <div className="bars">
          {hist.safe.map((_, i) => {
            const sh = (hist.safe[i] / maxCount) * 100;
            const hh = (hist.harm[i] / maxCount) * 100;
            return (
              <div key={i} className="bar">
                <div className="safe" style={{height: sh + "%"}} title={`safe: ${hist.safe[i]}`}/>
                <div className="harm" style={{height: hh + "%"}} title={`harm: ${hist.harm[i]}`}/>
              </div>
            );
          })}
          <div className="thresh-line" style={{ left: thresholdBin * 100 + "%" }}/>
        </div>
        <div className="hist-axis">
          <span>−0.25</span><span>−0.10</span><span className="mono">0</span><span>+0.10</span><span>+0.25</span>
        </div>
      </div>
      <div className="row mt-8 mini muted">
        <Chip tone="ok" dot>safe · 900</Chip>
        <Chip tone="warn" dot>harm · 600</Chip>
        <span className="spacer"/>
        <span className="mono muted">τ = −0.0347 · F1 0.895</span>
      </div>
    </div>
  );
}

// Format helpers
function relTime(iso) {
  const d = new Date(iso);
  const s = Math.max(1, Math.floor((Date.now() - d.getTime())/1000));
  if (s < 60)   return s + "s";
  if (s < 3600) return Math.floor(s/60) + "m";
  if (s < 86400)return Math.floor(s/3600) + "h";
  return Math.floor(s/86400) + "d";
}
function fmtTime(iso) {
  const d = new Date(iso);
  return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}
function fmtDate(iso) {
  return new Date(iso).toISOString().replace("T", " ").slice(0, 19);
}

Object.assign(window, { Spark, KPI, Chip, EnergyBar, EnergyRuler, Histogram, relTime, fmtTime, fmtDate });
