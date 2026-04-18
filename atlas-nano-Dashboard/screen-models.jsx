/* global React, Chip, NANO_MODEL, MODELS, fmtDate */

const { useState: useStateM } = React;

function ModelsScreen() {
  const [activeId, setActiveId] = useStateM(MODELS[0].id);
  const model = MODELS.find(m => m.id === activeId) || MODELS[0];

  const tensorEntries = [
    { k: "safety.version",             v: "1",                          t: "uint32" },
    { k: "safety.type",                v: "sign_check_atlas",           t: "string" },
    { k: "safety.extraction_layer",    v: String(model.extraction_layer), t: "uint32" },
    { k: "safety.extraction_component",v: model.extraction_component,   t: "string" },
    { k: "safety.threshold",           v: model.threshold.toFixed(17),  t: "float32" },
    { k: "safety.hidden_dim",          v: String(model.hidden_dim),     t: "uint32" },
    { k: "safety.calibration_f1",      v: model.f1.toFixed(16),         t: "float32" },
    { k: "safety.calibration_precision",v: model.precision.toFixed(16), t: "float32" },
    { k: "safety.calibration_recall",  v: model.recall.toFixed(16),     t: "float32" },
    { k: "safety.model_name",          v: model.base,                   t: "string" },
    { k: "safety.calibration_date",    v: model.calibrated,             t: "string" },
    { k: "safety.energy_axis",         v: `[${model.hidden_dim}] float32`, t: "tensor" },
    { k: "safety.energy_axis_norm",    v: "1.0",                        t: "float32" },
  ];

  return (
    <div className="screen">
      <div className="screen-head">
        <div>
          <h1 className="screen-h1">Models</h1>
          <p className="screen-sub">
            GGUF files with embedded sign-check safety tensors. Each adds a single <span className="mono">[hidden_dim]</span> energy axis plus calibration metadata — under 11 KB of overhead per model.
          </p>
        </div>
        <div className="screen-actions">
          <button className="btn">Inject into GGUF…</button>
          <button className="btn primary">Calibrate new model</button>
        </div>
      </div>

      <div className="grid-2" style={{gridTemplateColumns:"1fr 1.1fr"}}>
        {/* Model list */}
        <div>
          <div className="model-list">
            {MODELS.map(m => (
              <div key={m.id} className={"model-card " + (m.id === activeId ? "active" : "")} onClick={() => setActiveId(m.id)}>
                <div className="model-icon">{m.hidden_dim}</div>
                <div>
                  <div className="model-name">{m.name}</div>
                  <div className="model-meta">
                    {m.file} · L{m.extraction_layer} · {m.size} ·{" "}
                    <span style={{color: m.status === "deployed" ? "var(--ok)" : m.status === "staging" ? "var(--warn)" : "var(--fg-3)"}}>
                      {m.status}
                    </span>
                  </div>
                </div>
                <div className="model-metrics">
                  <div className="model-metric">
                    <div className="model-metric-label">F1</div>
                    <div className="model-metric-value">{m.f1.toFixed(3)}</div>
                  </div>
                  <div className="model-metric">
                    <div className="model-metric-label">Prec.</div>
                    <div className="model-metric-value">{m.precision.toFixed(3)}</div>
                  </div>
                  <div className="model-metric">
                    <div className="model-metric-label">Rec.</div>
                    <div className="model-metric-value">{m.recall.toFixed(3)}</div>
                  </div>
                </div>
                <svg viewBox="0 0 16 16" width="14" height="14" style={{color:"var(--fg-3)"}}><path d="M6 3 L11 8 L6 13" fill="none" stroke="currentColor" strokeWidth="1.4"/></svg>
              </div>
            ))}
          </div>
        </div>

        {/* Tensor inspector */}
        <div className="detail">
          <div className="detail-head">
            <div>
              <div className="detail-id">{model.file} · 2.3 GB</div>
              <div style={{fontSize:14, fontWeight:600, marginTop:4}}>{model.name}</div>
              <div className="mini muted mt-4">
                Base: <span className="mono">{model.base}</span> · calibrated {new Date(model.calibrated).toISOString().slice(0,10)}
              </div>
            </div>
            <div className="row" style={{gap:6}}>
              <Chip tone={model.status === "deployed" ? "ok" : model.status === "staging" ? "warn" : "ghost"} dot>{model.status}</Chip>
            </div>
          </div>

          <div className="detail-body">
            <div className="mini muted" style={{marginBottom:6}}>EMBEDDED METADATA · safety.*</div>
            <div style={{border:"1px solid var(--rule)", borderRadius:5, background:"var(--bg-sunken)"}}>
              {tensorEntries.map(e => (
                <div className="tensor-row" key={e.k}>
                  <div className="tensor-key">{e.k}</div>
                  <div className="tensor-type">{e.t}</div>
                  <div className="tensor-val">{e.v}</div>
                </div>
              ))}
            </div>

            <div className="mini muted mt-16" style={{marginBottom:6}}>ENERGY AXIS · first 32 components</div>
            <div className="prompt-block" style={{maxHeight:120, fontSize:11.5}}>
{`[ 0.0231, -0.1148,  0.0042,  0.0827, -0.0361,  0.1402,  0.0518, -0.0079,
 -0.0212,  0.0690,  0.1178,  0.0094, -0.0553, -0.0318,  0.0247,  0.0841,
  0.0173, -0.0928,  0.0062,  0.0504, -0.0111,  0.0378, -0.0421,  0.0288,
  0.0917, -0.0142,  0.0606, -0.0031,  0.0452,  0.0189, -0.0767,  0.0325, … ]`}
            </div>

            <div className="row mt-16" style={{gap:6}}>
              <button className="btn">Download .gguf</button>
              <button className="btn">Re-extract axis</button>
              <button className="btn">Deploy to prod</button>
              <button className="btn ghost">Archive</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

window.ModelsScreen = ModelsScreen;
