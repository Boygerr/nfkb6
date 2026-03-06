import { useState, useEffect, useRef } from "react";

// ── Runge-Kutta 4 solver (no scipy needed in browser) ──────────────────────
function solveODE(model, y0, tSpan, steps) {
  const dt = (tSpan[1] - tSpan[0]) / steps;
  const t = [];
  const ys = y0.map(() => []);
  let y = [...y0];
  let time = tSpan[0];
  for (let i = 0; i <= steps; i++) {
    t.push(time);
    y.forEach((v, j) => ys[j].push(v));
    const k1 = model(time, y);
    const k2 = model(time + dt / 2, y.map((v, j) => v + (dt / 2) * k1[j]));
    const k3 = model(time + dt / 2, y.map((v, j) => v + (dt / 2) * k2[j]));
    const k4 = model(time + dt, y.map((v, j) => v + dt * k3[j]));
    y = y.map((v, j) => Math.max(0, v + (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])));
    time += dt;
  }
  return { t, ys };
}

// ── ODE model ──────────────────────────────────────────────────────────────
function buildModel(tlr, brakeBlocker, alarmBlocker) {
  const k_import = 1.0, k_export = 0.5, k_deg_I = 1.5;
  const k_syn_I = 0.1, k_decay_I = 0.1;
  const k_tnf = 1.5, k_tnf_decay = 0.3;
  const k_tx = 0.2, k_mdeg = 0.05, k_tl = 0.1;
  return function (t, y) {
    const [Nc, Nn, Im, I, TNF] = y;
    const IKK = tlr;
    const tlr_deg = k_deg_I * IKK * I * (1 - brakeBlocker);
    const dIm = k_tx * Nn * (IKK / (IKK + 1e-6)) - k_mdeg * Im;
    const dI = k_syn_I - k_decay_I * I - tlr_deg + k_tl * Im;
    let effective_import = 0, export_N = 0;
    if (IKK > 0) {
      const degraded_fraction = tlr_deg / (k_deg_I * IKK * I + 1e-6);
      effective_import = k_import * Nc * degraded_fraction * (1 - alarmBlocker);
      export_N = k_export * Nn * (1 + I);
    }
    const dTNF = k_tnf * Nn - k_tnf_decay * TNF;
    const dNc = -effective_import + export_N;
    const dNn = effective_import - export_N;
    return [dNc, dNn, dIm, dI, dTNF];
  };
}

// ── Tiny SVG line chart ────────────────────────────────────────────────────
function LineChart({ series, colors, labels, width = 520, height = 220 }) {
  const pad = { top: 16, right: 16, bottom: 36, left: 42 };
  const W = width - pad.left - pad.right;
  const H = height - pad.top - pad.bottom;
  const allVals = series.flatMap(s => s);
  const maxY = Math.max(2, ...allVals) * 1.1;
  const maxX = series[0]?.length - 1 || 1;
  const toX = i => (i / maxX) * W;
  const toY = v => H - (v / maxY) * H;
  const polyline = (s) => s.map((v, i) => `${toX(i)},${toY(v)}`).join(" ");
  const yTicks = [0, 0.5, 1.0, 1.5, 2.0].filter(v => v <= maxY);
  const xTicks = [0, 10, 20, 30, 40, 50];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto", display: "block" }}>
      <g transform={`translate(${pad.left},${pad.top})`}>
        {/* grid */}
        {yTicks.map(v => (
          <line key={v} x1={0} x2={W} y1={toY(v)} y2={toY(v)}
            stroke="#e2e8f0" strokeWidth="1" strokeDasharray="4 3" />
        ))}
        {xTicks.map(x => (
          <line key={x} x1={toX(x * maxX / 50)} x2={toX(x * maxX / 50)} y1={0} y2={H}
            stroke="#e2e8f0" strokeWidth="1" strokeDasharray="4 3" />
        ))}
        {/* axes */}
        <line x1={0} x2={0} y1={0} y2={H} stroke="#94a3b8" strokeWidth="1.5" />
        <line x1={0} x2={W} y1={H} y2={H} stroke="#94a3b8" strokeWidth="1.5" />
        {/* y ticks */}
        {yTicks.map(v => (
          <text key={v} x={-8} y={toY(v) + 4} textAnchor="end"
            fontSize="11" fill="#64748b" fontFamily="'DM Mono', monospace">{v.toFixed(1)}</text>
        ))}
        {/* x ticks */}
        {xTicks.map(x => (
          <text key={x} x={toX(x * maxX / 50)} y={H + 18}
            textAnchor="middle" fontSize="11" fill="#64748b" fontFamily="'DM Mono', monospace">{x}</text>
        ))}
        <text x={W / 2} y={H + 32} textAnchor="middle" fontSize="12" fill="#94a3b8" fontFamily="'DM Mono', monospace">Time (minutes)</text>
        <text x={-H / 2} y={-30} textAnchor="middle" fontSize="12" fill="#94a3b8"
          fontFamily="'DM Mono', monospace" transform="rotate(-90)">Activity level</text>
        {/* lines */}
        {series.map((s, i) => (
          <polyline key={i} points={polyline(s)} fill="none"
            stroke={colors[i]} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
        ))}
      </g>
      {/* legend */}
      <g transform={`translate(${pad.left + 8}, ${pad.top + 8})`}>
        {labels.map((label, i) => (
          <g key={i} transform={`translate(0, ${i * 20})`}>
            <line x1={0} x2={20} y1={6} y2={6} stroke={colors[i]} strokeWidth="2.5" strokeLinecap="round" />
            <text x={26} y={11} fontSize="12" fill="#334155" fontFamily="'DM Mono', monospace">{label}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

// ── Experiment steps ───────────────────────────────────────────────────────
const STEPS = [
  {
    id: 0,
    title: "Baseline — No Threat",
    instruction: "This is the resting cell. No bacteria present. Observe the default state of all three signals.",
    hypothesis: "Predict: what happens to the Alarm Signal if bacteria are introduced?",
    hint: "Look at the Brake — it's high. What role might it play?",
    badge: "🔬 Observer"
  },
  {
    id: 1,
    title: "Introduce a Threat",
    instruction: "Press 'Introduce Bacteria' below. Watch what happens to the Alarm Signal and the Brake over time.",
    hypothesis: "Predict: why does the Alarm Signal rise, then fall back down?",
    hint: "Notice the Brake drops first, then recovers. Could these be linked?",
    badge: "🧫 Experimenter"
  },
  {
    id: 2,
    title: "Block the Brake",
    instruction: "With bacteria present, move the 'Brake Blocker' slider to maximum. This is like adding a drug that prevents the braking system from working.",
    hypothesis: "Predict: will the Alarm stay high, go higher, or change unexpectedly?",
    hint: "Think about what the Brake normally does to the Alarm Signal.",
    badge: "💊 Pharmacologist"
  },
  {
    id: 3,
    title: "Block the Alarm",
    instruction: "Reset the Brake Blocker. Now increase the 'Alarm Blocker' slider. This blocks the alarm from entering the nucleus.",
    hypothesis: "How does this differ from blocking the Brake?",
    hint: "Compare the Inflammatory Signal between this experiment and the last.",
    badge: "🔬 Cell Biologist"
  }
];

// ── Narrative engine ───────────────────────────────────────────────────────
function getNarrative(stimulated, brakeBlocker, alarmBlocker, Nn, I, TNF) {
  if (!stimulated) {
    return {
      headline: "Cell is at rest.",
      body: "The Brake (IκBα) is holding the Alarm Signal (NF-κB) in the cytoplasm. No threat detected — the cell is quiet."
    };
  }
  if (Nn > 0.4 && I > 0.5) {
    return {
      headline: "Alarm activated — brake rebuilding.",
      body: "Bacteria detected! The Alarm Signal flooded the nucleus. But the cell also started rebuilding the Brake to avoid over-reacting — a classic negative feedback loop."
    };
  }
  if (brakeBlocker > 0.6 && Nn > 0.3) {
    return {
      headline: "⚠️ Brake removed — Alarm stays high.",
      body: "With no brake, the Alarm Signal cannot be switched off. The Inflammatory Signal keeps rising. This mimics what happens in chronic inflammation diseases."
    };
  }
  if (alarmBlocker > 0.6) {
    return {
      headline: "Alarm blocked at the gate.",
      body: "The Alarm Signal is prevented from reaching the nucleus entirely. The Inflammatory Signal stays flat — even with bacteria present. The cell can't mount a response."
    };
  }
  return {
    headline: "Threat response underway.",
    body: "The cell is processing the bacterial signal. Watch the Brake and Alarm interact over time."
  };
}

// ── Main component ─────────────────────────────────────────────────────────
export default function CellAlarmSim() {
  const [stimulated, setStimulated] = useState(false);
  const [brakeBlocker, setBrakeBlocker] = useState(0);
  const [alarmBlocker, setAlarmBlocker] = useState(0);
  const [step, setStep] = useState(0);
  const [showAlarm, setShowAlarm] = useState(true);
  const [showBrake, setShowBrake] = useState(true);
  const [showInflammation, setShowInflammation] = useState(true);
  const [simData, setSimData] = useState(null);

  useEffect(() => {
    const tlr = stimulated ? 0.2 : 0.0;
    const model = buildModel(tlr, brakeBlocker, alarmBlocker);
    const result = solveODE(model, [1, 0, 0, 1, 0], [0, 50], 800);
    setSimData(result);
  }, [stimulated, brakeBlocker, alarmBlocker]);

  const lastNn = simData?.ys[1]?.at(-1) ?? 0;
  const lastI = simData?.ys[3]?.at(-1) ?? 0;
  const lastTNF = simData?.ys[4]?.at(-1) ?? 0;
  const narrative = getNarrative(stimulated, brakeBlocker, alarmBlocker, lastNn, lastI, lastTNF);

  const series = [];
  const colors = [];
  const labels = [];
  if (simData && showAlarm) { series.push(simData.ys[1]); colors.push("#ef4444"); labels.push("Alarm Signal (Nuclear NF-κB)"); }
  if (simData && showBrake) { series.push(simData.ys[3]); colors.push("#3b82f6"); labels.push("Brake (IκBα)"); }
  if (simData && showInflammation) { series.push(simData.ys[4]); colors.push("#f59e0b"); labels.push("Inflammatory Signal (TNF)"); }

  const currentStep = STEPS[step];

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2027 100%)",
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      color: "#f1f5f9",
      padding: "0"
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Fraunces:wght@700;900&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{
        background: "linear-gradient(90deg, #1e3a5f, #0f4c75)",
        borderBottom: "1px solid #1e40af",
        padding: "20px 32px",
        display: "flex", alignItems: "center", gap: "16px"
      }}>
        <div style={{
          width: 44, height: 44, borderRadius: "50%",
          background: "radial-gradient(circle, #ef4444, #7f1d1d)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 22, boxShadow: "0 0 20px rgba(239,68,68,0.4)"
        }}>🦠</div>
        <div>
          <div style={{ fontFamily: "Fraunces, serif", fontSize: 22, fontWeight: 900, letterSpacing: "-0.5px", color: "#f8fafc" }}>
            Cell Alarm System
          </div>
          <div style={{ fontSize: 12, color: "#94a3b8", fontFamily: "DM Mono, monospace", letterSpacing: "0.05em" }}>
            HOW YOUR CELLS DETECT AND RESPOND TO BACTERIA
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          {STEPS.map((s, i) => (
            <div key={i} onClick={() => setStep(i)} style={{
              width: 10, height: 10, borderRadius: "50%",
              background: i === step ? "#ef4444" : i < step ? "#22c55e" : "#334155",
              cursor: "pointer", transition: "all 0.2s",
              boxShadow: i === step ? "0 0 8px #ef4444" : "none"
            }} />
          ))}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 0, minHeight: "calc(100vh - 85px)" }}>

        {/* LEFT PANEL */}
        <div style={{
          background: "#0f172a",
          borderRight: "1px solid #1e293b",
          padding: "24px 20px",
          display: "flex", flexDirection: "column", gap: 20
        }}>

          {/* Step guide */}
          <div style={{
            background: "linear-gradient(135deg, #1e3a5f, #1e2d4f)",
            border: "1px solid #2563eb44",
            borderRadius: 12, padding: "16px"
          }}>
            <div style={{ fontSize: 11, color: "#60a5fa", fontFamily: "DM Mono, monospace", letterSpacing: "0.08em", marginBottom: 6 }}>
              EXPERIMENT {step + 1} OF {STEPS.length} · {currentStep.badge}
            </div>
            <div style={{ fontSize: 15, fontWeight: 600, color: "#f1f5f9", marginBottom: 8 }}>
              {currentStep.title}
            </div>
            <div style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.6, marginBottom: 10 }}>
              {currentStep.instruction}
            </div>
            <div style={{
              background: "#0f172a", borderRadius: 8, padding: "10px 12px",
              borderLeft: "3px solid #f59e0b"
            }}>
              <div style={{ fontSize: 11, color: "#f59e0b", fontFamily: "DM Mono, monospace", marginBottom: 3 }}>HYPOTHESIS</div>
              <div style={{ fontSize: 12, color: "#fef3c7", lineHeight: 1.5 }}>{currentStep.hypothesis}</div>
            </div>
            {currentStep.hint && (
              <div style={{ fontSize: 12, color: "#64748b", marginTop: 8, fontStyle: "italic" }}>
                💡 {currentStep.hint}
              </div>
            )}
          </div>

          {/* Stimulus */}
          <div>
            <div style={{ fontSize: 11, color: "#64748b", fontFamily: "DM Mono, monospace", letterSpacing: "0.08em", marginBottom: 10 }}>STIMULUS</div>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={() => { setStimulated(true); if (step === 0) setStep(1); }} style={{
                flex: 1, padding: "10px 8px", borderRadius: 8, border: "none", cursor: "pointer",
                background: stimulated ? "linear-gradient(135deg, #dc2626, #991b1b)" : "linear-gradient(135deg, #1d4ed8, #1e40af)",
                color: "#fff", fontSize: 12, fontWeight: 600, fontFamily: "DM Sans, sans-serif",
                boxShadow: stimulated ? "0 0 14px rgba(220,38,38,0.5)" : "0 0 14px rgba(29,78,216,0.3)",
                transition: "all 0.3s"
              }}>
                🦠 Introduce Bacteria
              </button>
              <button onClick={() => { setStimulated(false); setBrakeBlocker(0); setAlarmBlocker(0); setStep(0); }} style={{
                padding: "10px 12px", borderRadius: 8, border: "1px solid #334155", cursor: "pointer",
                background: "#1e293b", color: "#94a3b8", fontSize: 12, fontFamily: "DM Sans, sans-serif"
              }}>
                ↺ Reset
              </button>
            </div>
          </div>

          {/* Drugs */}
          <div>
            <div style={{ fontSize: 11, color: "#64748b", fontFamily: "DM Mono, monospace", letterSpacing: "0.08em", marginBottom: 10 }}>
              EXPERIMENTAL DRUGS
            </div>
            {[
              { label: "Brake Blocker", sublabel: "(IκBα inhibitor)", val: brakeBlocker, set: (v) => { setBrakeBlocker(v); if (step < 2) setStep(2); }, color: "#3b82f6" },
              { label: "Alarm Blocker", sublabel: "(NF-κB inhibitor)", val: alarmBlocker, set: (v) => { setAlarmBlocker(v); if (step < 3) setStep(3); }, color: "#ef4444" }
            ].map(({ label, sublabel, val, set, color }) => (
              <div key={label} style={{ marginBottom: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 500, color: "#e2e8f0" }}>{label}</div>
                    <div style={{ fontSize: 11, color: "#64748b", fontFamily: "DM Mono, monospace" }}>{sublabel}</div>
                  </div>
                  <div style={{
                    fontFamily: "DM Mono, monospace", fontSize: 13,
                    color: color, background: `${color}22`,
                    padding: "2px 8px", borderRadius: 4
                  }}>
                    {Math.round(val * 100)}%
                  </div>
                </div>
                <input type="range" min={0} max={1} step={0.01} value={val}
                  onChange={e => set(parseFloat(e.target.value))}
                  style={{ width: "100%", accentColor: color, cursor: "pointer" }} />
              </div>
            ))}
          </div>

          {/* Show/hide toggles */}
          <div>
            <div style={{ fontSize: 11, color: "#64748b", fontFamily: "DM Mono, monospace", letterSpacing: "0.08em", marginBottom: 10 }}>SHOW ON GRAPH</div>
            {[
              { label: "Alarm Signal", color: "#ef4444", val: showAlarm, set: setShowAlarm },
              { label: "Brake", color: "#3b82f6", val: showBrake, set: setShowBrake },
              { label: "Inflammatory Signal", color: "#f59e0b", val: showInflammation, set: setShowInflammation },
            ].map(({ label, color, val, set }) => (
              <label key={label} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <div onClick={() => set(!val)} style={{
                  width: 32, height: 18, borderRadius: 9,
                  background: val ? color : "#334155",
                  position: "relative", transition: "background 0.2s", flexShrink: 0
                }}>
                  <div style={{
                    width: 12, height: 12, borderRadius: "50%", background: "#fff",
                    position: "absolute", top: 3, left: val ? 16 : 3, transition: "left 0.2s"
                  }} />
                </div>
                <span style={{ fontSize: 12, color: val ? "#e2e8f0" : "#64748b" }}>{label}</span>
              </label>
            ))}
          </div>

          {/* Nav */}
          <div style={{ marginTop: "auto", display: "flex", gap: 8 }}>
            <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0} style={{
              flex: 1, padding: "8px", borderRadius: 8, border: "1px solid #334155",
              background: step === 0 ? "#0f172a" : "#1e293b", color: step === 0 ? "#334155" : "#94a3b8",
              cursor: step === 0 ? "not-allowed" : "pointer", fontSize: 12
            }}>← Back</button>
            <button onClick={() => setStep(Math.min(STEPS.length - 1, step + 1))} disabled={step === STEPS.length - 1} style={{
              flex: 1, padding: "8px", borderRadius: 8, border: "none",
              background: step === STEPS.length - 1 ? "#334155" : "linear-gradient(135deg, #1d4ed8, #7c3aed)",
              color: "#fff", cursor: step === STEPS.length - 1 ? "not-allowed" : "pointer", fontSize: 12, fontWeight: 600
            }}>Next →</button>
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div style={{ padding: "28px 32px", display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Narrative card */}
          <div style={{
            background: "linear-gradient(135deg, #1e293b, #162032)",
            border: "1px solid #2d3f55",
            borderRadius: 14, padding: "18px 22px",
            display: "flex", gap: 14, alignItems: "flex-start"
          }}>
            <div style={{
              width: 36, height: 36, borderRadius: "50%", flexShrink: 0,
              background: stimulated ? "radial-gradient(circle, #dc2626, #7f1d1d)" : "radial-gradient(circle, #1d4ed8, #1e3a5f)",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
              boxShadow: stimulated ? "0 0 16px rgba(220,38,38,0.4)" : "0 0 16px rgba(29,78,216,0.3)"
            }}>
              {stimulated ? "⚡" : "😴"}
            </div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600, color: "#f1f5f9", marginBottom: 4 }}>
                {narrative.headline}
              </div>
              <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.6 }}>
                {narrative.body}
              </div>
            </div>
          </div>

          {/* Chart */}
          <div style={{
            background: "#1a2332",
            border: "1px solid #2d3f55",
            borderRadius: 14, padding: "20px 16px 8px"
          }}>
            {series.length > 0 && simData ? (
              <LineChart series={series} colors={colors} labels={labels} />
            ) : (
              <div style={{ textAlign: "center", color: "#475569", padding: 40, fontFamily: "DM Mono, monospace", fontSize: 13 }}>
                Select at least one signal to display
              </div>
            )}
          </div>

          {/* Status indicators */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
            {[
              { label: "Alarm Signal", sublabel: "Nuclear NF-κB", value: lastNn, max: 1.5, color: "#ef4444", icon: "🔔" },
              { label: "Brake", sublabel: "IκBα level", value: lastI, max: 1.5, color: "#3b82f6", icon: "🛑" },
              { label: "Inflammatory Signal", sublabel: "TNF output", value: lastTNF, max: 3, color: "#f59e0b", icon: "🔥" },
            ].map(({ label, sublabel, value, max, color, icon }) => (
              <div key={label} style={{
                background: "#1a2332", border: "1px solid #2d3f55",
                borderRadius: 12, padding: "14px 16px"
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, color: "#e2e8f0" }}>{icon} {label}</div>
                    <div style={{ fontSize: 10, color: "#64748b", fontFamily: "DM Mono, monospace" }}>{sublabel}</div>
                  </div>
                  <div style={{ fontFamily: "DM Mono, monospace", fontSize: 14, color, alignSelf: "center" }}>
                    {value.toFixed(2)}
                  </div>
                </div>
                <div style={{ background: "#0f172a", borderRadius: 4, height: 6, overflow: "hidden" }}>
                  <div style={{
                    height: "100%", borderRadius: 4,
                    background: `linear-gradient(90deg, ${color}88, ${color})`,
                    width: `${Math.min(100, (value / max) * 100)}%`,
                    transition: "width 0.3s"
                  }} />
                </div>
              </div>
            ))}
          </div>

          {/* Analogy box */}
          <div style={{
            background: "#0f172a",
            border: "1px solid #1e3a5f",
            borderRadius: 12, padding: "14px 18px",
            display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12
          }}>
            {[
              { bio: "NF-κB", plain: "Alarm Signal", desc: "Transcription factor that activates immune genes when it enters the nucleus" },
              { bio: "IκBα", plain: "The Brake", desc: "Inhibitory protein that traps NF-κB in the cytoplasm, preventing overactivation" },
              { bio: "TNF-α", plain: "Inflammatory Signal", desc: "Cytokine secreted by the cell to recruit other immune cells to the infection site" },
            ].map(({ bio, plain, desc }) => (
              <div key={bio} style={{ fontSize: 12 }}>
                <div style={{ fontFamily: "DM Mono, monospace", color: "#60a5fa", fontSize: 11 }}>{bio}</div>
                <div style={{ fontWeight: 600, color: "#f1f5f9", marginBottom: 3 }}>"{plain}"</div>
                <div style={{ color: "#64748b", lineHeight: 1.4 }}>{desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
