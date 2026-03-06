import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cell Alarm System",
    page_icon="🦠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f1f5f9;
}

/* Header banner */
.header-banner {
    background: linear-gradient(90deg, #1e3a5f, #0f4c75);
    border: 1px solid #1e40af55;
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
}

/* Experiment step card */
.step-card {
    background: linear-gradient(135deg, #1e3a5f22, #1e2d4f33);
    border: 1px solid #2563eb44;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 16px;
}

/* Hypothesis box */
.hypothesis-box {
    background: #0f172a;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-top: 10px;
}

/* Narrative card */
.narrative-card {
    background: linear-gradient(135deg, #1e293b, #162032);
    border: 1px solid #2d3f55;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

/* Glossary row */
.glossary-row {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 16px;
}

/* Metric card */
.metric-card {
    background: #1a2332;
    border: 1px solid #2d3f55;
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

div[data-testid="stMetric"] {
    background: #1a2332;
    border: 1px solid #2d3f55;
    border-radius: 12px;
    padding: 12px 16px;
}

div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-family: 'DM Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "stimulated" not in st.session_state:
    st.session_state.stimulated = False
if "step" not in st.session_state:
    st.session_state.step = 0

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦠 Cell Alarm System")
    st.markdown("---")

    st.markdown("**STIMULUS**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🦠 Introduce\nBacteria", use_container_width=True):
            st.session_state.stimulated = True
            if st.session_state.step == 0:
                st.session_state.step = 1
    with col2:
        if st.button("↺ Reset\nAll", use_container_width=True):
            st.session_state.stimulated = False
            st.session_state.step = 0

    st.markdown("---")
    st.markdown("**EXPERIMENTAL DRUGS**")

    brake_blocker = st.slider(
        "Brake Blocker (IkBa inhibitor)",
        0.0, 1.0, 0.0, 0.01,
        help="Prevents IkBa (the Brake) from being rebuilt after activation"
    )
    alarm_blocker = st.slider(
        "Alarm Blocker (NF-kB inhibitor)",
        0.0, 1.0, 0.0, 0.01,
        help="Prevents NF-kB (the Alarm Signal) from entering the nucleus"
    )

    if brake_blocker > 0.05 and st.session_state.step < 2:
        st.session_state.step = 2
    if alarm_blocker > 0.05 and st.session_state.step < 3:
        st.session_state.step = 3

    st.markdown("---")
    st.markdown("**GRAPH DISPLAY**")
    show_alarm     = st.checkbox("Show Alarm Signal (NF-kB)", value=True)
    show_brake     = st.checkbox("Show Brake (IkBa)", value=True)
    show_inflam    = st.checkbox("Show Inflammatory Signal (TNF)", value=False)

    st.markdown("---")
    st.markdown("**EXPERIMENT PROGRESS**")
    step_labels = ["Baseline", "Bacteria", "Brake Drug", "Alarm Drug"]
    for i, label in enumerate(step_labels):
        icon = "✅" if i < st.session_state.step else ("▶️" if i == st.session_state.step else "⬜")
        st.markdown(f"{icon} Step {i+1}: {label}")

# ── ODE model ─────────────────────────────────────────────────────────────────
tlr_strength = 0.2 if st.session_state.stimulated else 0.0

def nfkb_model(t, y):
    Nc, Nn, Im, I, TNF = y
    IKK = tlr_strength
    k_import = 1.0; k_export = 0.5; k_deg_I = 1.5
    k_syn_I = 0.1;  k_decay_I = 0.1
    k_tnf = 1.5;    k_tnf_decay = 0.3
    k_tx = 0.2;     k_mdeg = 0.05; k_tl = 0.1

    tlr_deg = k_deg_I * IKK * I * (1 - brake_blocker)
    dIm = k_tx * Nn * (IKK / (IKK + 1e-6)) - k_mdeg * Im
    dI  = k_syn_I - k_decay_I * I - tlr_deg + k_tl * Im

    if IKK > 0:
        degraded_fraction  = tlr_deg / (k_deg_I * IKK * I + 1e-6)
        effective_import   = k_import * Nc * degraded_fraction * (1 - alarm_blocker)
        export_N           = k_export * Nn * (1 + I)
    else:
        effective_import = 0.0
        export_N         = 0.0

    dTNF = k_tnf * Nn - k_tnf_decay * TNF
    dNc  = -effective_import + export_N
    dNn  =  effective_import - export_N
    return [dNc, dNn, dIm, dI, dTNF]

t_eval = np.linspace(0, 50, 1000)
sol = solve_ivp(nfkb_model, [0, 50], [1.0, 0.0, 0.0, 1.0, 0.0], t_eval=t_eval)

Nn_end  = float(sol.y[1, -1])
I_end   = float(sol.y[3, -1])
TNF_end = float(sol.y[4, -1])

# ── Narrative logic ───────────────────────────────────────────────────────────
def get_narrative():
    if not st.session_state.stimulated:
        return "😴", "Cell is at rest.", \
               "The Brake (IkBa) is holding the Alarm Signal (NF-kB) in the cytoplasm. No threat detected — everything is quiet."
    if brake_blocker > 0.6 and Nn_end > 0.2:
        return "⚠️", "Brake removed — Alarm stays high!", \
               "With no Brake, the Alarm Signal cannot be switched off. The Inflammatory Signal keeps rising. This mimics what happens in chronic inflammation diseases like rheumatoid arthritis."
    if alarm_blocker > 0.6:
        return "🚫", "Alarm blocked at the gate.", \
               "The Alarm Signal is prevented from reaching the nucleus entirely. The Inflammatory Signal stays flat — even with bacteria present. The cell cannot mount a defence."
    if Nn_end > 0.3 and I_end > 0.4:
        return "⚡", "Alarm activated — Brake rebuilding.", \
               "Bacteria detected! The Alarm Signal flooded the nucleus. But the cell also started rebuilding the Brake to avoid over-reacting — a classic negative feedback loop."
    return "🔬", "Threat response underway.", \
           "The cell is processing the bacterial signal. Watch how the Brake and Alarm Signal interact over time."

emoji, headline, body = get_narrative()

# ── Experiment steps ──────────────────────────────────────────────────────────
STEPS = [
    {
        "badge": "Step 1 — Baseline",
        "title": "No Threat",
        "instruction": "This is the resting cell. No bacteria present. Observe the default state of both signals.",
        "hypothesis": "Predict: what happens to the Alarm Signal if bacteria are introduced?",
        "hint": "Look at the Brake — it starts high. What role might it play?"
    },
    {
        "badge": "Step 2 — Stimulate",
        "title": "Introduce a Threat",
        "instruction": "Press 'Introduce Bacteria' in the sidebar. Watch what happens to both signals over time.",
        "hypothesis": "Why does the Alarm Signal rise, then fall back down?",
        "hint": "Notice the Brake drops first, then recovers. Could these be linked?"
    },
    {
        "badge": "Step 3 — Drug 1",
        "title": "Block the Brake",
        "instruction": "With bacteria present, move the Brake Blocker slider to maximum. This mimics a drug that prevents the braking system from working.",
        "hypothesis": "Predict: will the Alarm stay high, go higher, or behave unexpectedly?",
        "hint": "Think about what the Brake normally does to the Alarm Signal."
    },
    {
        "badge": "Step 4 — Drug 2",
        "title": "Block the Alarm",
        "instruction": "Reset the Brake Blocker to zero. Now increase the Alarm Blocker. This blocks the alarm from entering the nucleus.",
        "hypothesis": "How does blocking the Alarm differ from blocking the Brake?",
        "hint": "Compare the Inflammatory Signal between this experiment and the last."
    },
]

step_data = STEPS[st.session_state.step]

# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <span style="font-size:2rem;">🦠</span>
  <div>
    <div style="font-size:1.5rem;font-weight:700;letter-spacing:-0.5px;">Cell Alarm System</div>
    <div style="font-size:0.75rem;color:#94a3b8;font-family:'DM Mono',monospace;letter-spacing:0.06em;">
      HOW YOUR CELLS DETECT AND RESPOND TO BACTERIA
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1.8], gap="large")

# ── LEFT: Experiment guide ────────────────────────────────────────────────────
with left_col:
    st.markdown(f"""
    <div class="step-card">
      <div style="font-size:11px;color:#60a5fa;font-family:'DM Mono',monospace;letter-spacing:0.08em;margin-bottom:6px;">
        {step_data['badge']}
      </div>
      <div style="font-size:1.05rem;font-weight:600;color:#f1f5f9;margin-bottom:8px;">
        {step_data['title']}
      </div>
      <div style="font-size:0.85rem;color:#cbd5e1;line-height:1.6;margin-bottom:8px;">
        {step_data['instruction']}
      </div>
      <div class="hypothesis-box">
        <div style="font-size:10px;color:#f59e0b;font-family:'DM Mono',monospace;margin-bottom:3px;">HYPOTHESIS</div>
        <div style="font-size:0.82rem;color:#fef3c7;line-height:1.5;">{step_data['hypothesis']}</div>
      </div>
      <div style="font-size:0.78rem;color:#64748b;margin-top:10px;font-style:italic;">
        💡 {step_data['hint']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Narrative card
    bg = "#3b1111" if st.session_state.stimulated else "#11233b"
    st.markdown(f"""
    <div class="narrative-card" style="background:{bg};">
      <div style="font-size:1.5rem;margin-bottom:6px;">{emoji}</div>
      <div style="font-size:1rem;font-weight:600;color:#f1f5f9;margin-bottom:6px;">{headline}</div>
      <div style="font-size:0.84rem;color:#94a3b8;line-height:1.6;">{body}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("🔔 Alarm", f"{Nn_end:.2f}")
    with m2:
        st.metric("🛑 Brake", f"{I_end:.2f}")
    with m3:
        st.metric("🔥 Inflam.", f"{TNF_end:.2f}")

# ── RIGHT: Chart ──────────────────────────────────────────────────────────────
with right_col:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1a2332")
    ax.set_facecolor("#111827")

    # Grid
    ax.grid(color="#2d3f55", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)

    plotted = False
    if show_alarm:
        ax.plot(sol.t, sol.y[1], color="#ef4444", linewidth=2.5,
                label="Alarm Signal (NF-kB)", solid_capstyle="round")
        plotted = True
    if show_brake:
        ax.plot(sol.t, sol.y[3], color="#3b82f6", linewidth=2.5,
                label="Brake (IkBa)", solid_capstyle="round")
        plotted = True
    if show_inflam:
        ax.plot(sol.t, sol.y[4], color="#f59e0b", linewidth=2.5,
                label="Inflammatory Signal (TNF)", solid_capstyle="round")
        plotted = True

    # Shade region when stimulated
    if st.session_state.stimulated:
        ax.axvspan(0, 50, alpha=0.04, color="#ef4444")
        ax.axvline(x=0, color="#ef4444", linewidth=1.2, linestyle=":", alpha=0.5)
        ax.text(1, 2.7, "Bacteria introduced", color="#ef444488",
                fontsize=9, fontstyle="italic")

    ax.set_xlabel("Time (minutes)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Activity Level", color="#94a3b8", fontsize=11)
    ax.set_ylim(0, 3)
    ax.set_xlim(0, 50)
    ax.tick_params(colors="#64748b", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3f55")

    if plotted:
        legend = ax.legend(
            facecolor="#0f172a", edgecolor="#2d3f55",
            labelcolor="#e2e8f0", fontsize=10,
            loc="upper right"
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Glossary strip
    st.markdown("""
    <div class="glossary-row">
      <div style="display:flex;gap:32px;flex-wrap:wrap;">
        <div>
          <div style="font-size:10px;color:#60a5fa;font-family:'DM Mono',monospace;">NF-kB</div>
          <div style="font-size:0.85rem;font-weight:600;color:#f1f5f9;">"Alarm Signal"</div>
          <div style="font-size:0.78rem;color:#64748b;max-width:180px;line-height:1.4;">
            Transcription factor that activates immune genes when it enters the nucleus
          </div>
        </div>
        <div>
          <div style="font-size:10px;color:#60a5fa;font-family:'DM Mono',monospace;">IkBa</div>
          <div style="font-size:0.85rem;font-weight:600;color:#f1f5f9;">"The Brake"</div>
          <div style="font-size:0.78rem;color:#64748b;max-width:180px;line-height:1.4;">
            Inhibitory protein that traps NF-kB in the cytoplasm, preventing over-activation
          </div>
        </div>
        <div>
          <div style="font-size:10px;color:#60a5fa;font-family:'DM Mono',monospace;">TNF-a</div>
          <div style="font-size:0.85rem;font-weight:600;color:#f1f5f9;">"Inflammatory Signal"</div>
          <div style="font-size:0.78rem;color:#64748b;max-width:180px;line-height:1.4;">
            Cytokine secreted by the cell to recruit other immune cells to the infection site
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
