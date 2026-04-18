/* global React, Chip, CATEGORIES, GAUNTLETS, AUDIT */

function CategoriesScreen() {
  return (
    <div className="screen">
      <div className="screen-head">
        <div>
          <h1 className="screen-h1">Categories</h1>
          <p className="screen-sub">
            Per-category F1 determines tier routing: T1 handled by Nano alone, T2 routes to Atlas V2, T3 requires human review.
          </p>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <div className="card-title">HarmBench taxonomy · 10 categories</div>
          <div className="card-meta">gauntlet v3 corrected · n = 1,181</div>
        </div>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Category</th>
                <th style={{textAlign:"right"}}>F1</th>
                <th style={{textAlign:"right"}}>Precision</th>
                <th style={{textAlign:"right"}}>Recall</th>
                <th style={{textAlign:"right"}}>Volume 24h</th>
                <th>Tier</th>
              </tr>
            </thead>
            <tbody>
              {CATEGORIES.map(c => (
                <tr key={c.key}>
                  <td>{c.name}</td>
                  <td className="mono" style={{textAlign:"right"}}>{c.f1.toFixed(3)}</td>
                  <td className="mono" style={{textAlign:"right"}}>{(c.f1 + 0.03).toFixed(3)}</td>
                  <td className="mono" style={{textAlign:"right"}}>{(c.f1 - 0.02).toFixed(3)}</td>
                  <td className="mono" style={{textAlign:"right"}}>{c.volume}</td>
                  <td>
                    {c.tier === "T1" && <Chip tone="ok" dot>T1 · Nano</Chip>}
                    {c.tier === "T2" && <Chip tone="warn" dot>T2 · Atlas V2</Chip>}
                    {c.tier === "T3" && <Chip tone="crit" dot>T3 · review</Chip>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function GauntletScreen() {
  return (
    <div className="screen">
      <div className="screen-head">
        <div>
          <h1 className="screen-h1">Gauntlets</h1>
          <p className="screen-sub">Prompt corpora used for calibration and held-out evaluation.</p>
        </div>
        <div className="screen-actions">
          <button className="btn">Import .txt</button>
          <button className="btn primary">New gauntlet</button>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <div className="card-title">Corpora</div>
          <div className="card-meta">5 registered · ~3,666 prompts total</div>
        </div>
        <table className="data">
          <thead>
            <tr><th>Name</th><th style={{textAlign:"right"}}>Prompts</th><th>Split</th><th>Updated</th><th></th></tr>
          </thead>
          <tbody>
            {GAUNTLETS.map(g => (
              <tr key={g.name}>
                <td className="mono">{g.name}.txt</td>
                <td className="mono" style={{textAlign:"right"}}>{g.prompts.toLocaleString()}</td>
                <td>
                  {g.split === "train"     && <Chip tone="info">train</Chip>}
                  {g.split === "held-out"  && <Chip tone="warn">held-out</Chip>}
                  {g.split === "held-out+" && <Chip tone="warn">held-out+</Chip>}
                  {g.split === "calibrate" && <Chip tone="ok">calibrate</Chip>}
                  {g.split === "external"  && <Chip tone="ghost">external</Chip>}
                </td>
                <td className="mono muted">{g.updated}</td>
                <td style={{textAlign:"right"}}><button className="btn ghost">Inspect →</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AuditScreen() {
  return (
    <div className="screen">
      <div className="screen-head">
        <div>
          <h1 className="screen-h1">Audit Log</h1>
          <p className="screen-sub">Every threshold change, deploy, and tier routing decision, immutable.</p>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <div className="card-title">Today · Apr 17, 2026</div>
          <div className="card-meta">7 events</div>
        </div>
        <table className="data">
          <thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Target</th><th>Detail</th></tr></thead>
          <tbody>
            {AUDIT.map((a,i) => (
              <tr key={i}>
                <td className="mono muted">{a.ts}</td>
                <td>{a.actor === "system" ? <Chip tone="ghost">system</Chip> : a.actor}</td>
                <td>{a.action}</td>
                <td className="mono">{a.target}</td>
                <td className="muted">{a.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

window.CategoriesScreen = CategoriesScreen;
window.GauntletScreen   = GauntletScreen;
window.AuditScreen      = AuditScreen;
