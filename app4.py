import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cell Alarm System",
    page_icon="🦠",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,600;1,400&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0f172a;
    color: #f1f5f9;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Remove top padding on mobile */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 560px !important;
}

/* Title */
.app-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.3px;
    margin-bottom: 2px;
}
.app-subtitle {
    font-size: 0.7rem;
    color: #475569;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* Section labels */
.section-label {
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 6px;
    margin-top: 14px;
}

/* Stimulus buttons */
.stButton > button {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 0 !important;
    transition: all 0.15s !important;
    border: none !important;
    width: 100% !important;
}

/* Slider label sizing */
.stSlider label {
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
}

/* Narrative card */
.narrative {
    border-radius: 10px;
    padding: 12px 14px;
    margin: 12px 0 4px 0;
    font-size: 0.82rem;
    line-height: 1.55;
    color: #cbd5e1;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 14px 0;
}

/* Glossary */
.glossary {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 8px;
    font-size: 0.75rem;
    color: #475569;
    line-height: 1.6;
}
.glossary b { color: #64748b; }
.glossary span { color: #94a3b8; font-family: 'DM Mono', monospace; font-size: 0.7rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "stimulated" not in st.session_state:
    st.session_state.stimulated = False

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🦠 Cell Alarm System</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">NF-κB · IκBα signalling simulator</div>', unsafe_allow_html=True)

# ── STIMULUS ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Stimulus</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    if st.button("🦠  Introduce Bacteria", use_container_width=True):
        st.session_state.stimulated = True
with c2:
    if st.button("↺  Reset", use_container_width=True):
        st.session_state.stimulated = False

# ── DRUGS ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Experimental Drugs</div>', unsafe_allow_html=True)

brake_blocker = st.slider(
    "Chemically stop Brake degredation",
    0.0, 1.0, 0.0, 0.01,
    help="Blocks IKK-mediated phosphorylation of IkBa — prevents the Brake from being degraded"
)
alarm_blocker = st.slider(
    "Chemically stop Alarm from ringing",
    0.0, 1.0, 0.0, 0.01,
    help="Blocks NF-kB translocation into the nucleus without affecting IkBa"
)

show_alarm = True
show_brake = True

# ── ODE MODEL ─────────────────────────────────────────────────────────────────
tlr_strength = 0.2 if st.session_state.stimulated else 0.0

def nfkb_model(t, y):
    Nc, Nn, Im, I, TNF = y
    IKK = tlr_strength
    k_import = 1.0;  k_export = 0.5;  k_deg_I = 1.5
    k_syn_I  = 0.1;  k_decay_I = 0.1
    k_tnf    = 1.5;  k_tnf_decay = 0.3
    k_tx     = 0.2;  k_mdeg = 0.05;  k_tl = 0.1

    # IkBa phosphorylation/degradation driven by IKK; brake_blocker inhibits this step
    tlr_deg = k_deg_I * IKK * I * (1 - brake_blocker)
    dIm = k_tx * Nn * (IKK / (IKK + 1e-6)) - k_mdeg * Im
    dI  = k_syn_I - k_decay_I * I - tlr_deg + k_tl * Im

    if IKK > 0:
        degraded_fraction = tlr_deg / (k_deg_I * IKK * I + 1e-6)
        # alarm_blocker prevents nuclear import regardless of IkBa state
        effective_import  = k_import * Nc * degraded_fraction * (1 - alarm_blocker)
        export_N          = k_export * Nn * (1 + I)
    else:
        effective_import = 0.0
        export_N         = 0.0

    dTNF = k_tnf * Nn - k_tnf_decay * TNF
    dNc  = -effective_import + export_N
    dNn  =  effective_import - export_N
    return [dNc, dNn, dIm, dI, dTNF]

sol = solve_ivp(
    nfkb_model, [0, 50],
    [1.0, 0.0, 0.0, 1.0, 0.0],
    t_eval=np.linspace(0, 50, 1000)
)

Nn_end = float(sol.y[1, -1])
I_end  = float(sol.y[3, -1])

# ── NARRATIVE ─────────────────────────────────────────────────────────────────
def get_narrative():
    if not st.session_state.stimulated:
        return "#1e293b", "😴  Cell at rest, no threat detected.", \
            "The Brake (IκBα) is blocking the Alarm Signal (NF-κB)"
    if brake_blocker > 0.5:
        return "#1a1a2e", "🛑  Brake locked — Alarm cannot activate.", \
            "The 'Brake protector' prevents the Brake from being degraded. " \
            "NF-κB stays trapped, no immune response"
    if alarm_blocker > 0.5:
        return "#1a1a2e", "🚫  Brake released, but Alarm can't ring.", \
            "Bacteria degraded the Brake as normal but Alarm cannot activate. " \
            "The Brake is gone, yet no response fires. The two steps are independent."
    if Nn_end > 0.15:
        return "#1c1a0e", "⚡  Alarm activated, Brake rebuilding.", \
            "The brake has been broken, releasing the alarm which can ring. " \
            "The cell is rebuilding the Brake to limit the response."
    return "#1e293b", "🔬  Processing threat...", \
        "Watch how the Brake and Alarm Signal interact over time."

bg, headline, body = get_narrative()
st.markdown(
    f'<div class="narrative" style="background:{bg};">'
    f'<b style="color:#f1f5f9;">{headline}</b><br>{body}</div>',
    unsafe_allow_html=True
)

# ── GRAPH ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.2))
fig.patch.set_facecolor("#111827")
ax.set_facecolor("#111827")

ax.grid(color="#334155", linestyle="--", linewidth=0.7, alpha=1.0)
ax.set_axisbelow(True)

plotted = False
if show_alarm:
    ax.plot(sol.t, sol.y[1], color="#ef4444", linewidth=2.2,
            label="Alarm Signal (NF-κB)", solid_capstyle="round")
    plotted = True
if show_brake:
    ax.plot(sol.t, sol.y[3], color="#3b82f6", linewidth=2.2,
            label="Brake (IκBα)", solid_capstyle="round")
    plotted = True

if st.session_state.stimulated:
    ax.axvspan(0, 50, alpha=0.03, color="#ef4444")
    ax.text(1, 1.41, "⬤ bacteria present", fontsize=7.5,
            fontstyle="italic", color="#ffffff99")

ax.set_xlabel("Time (minutes)", color="#ffffff", fontsize=9)
ax.set_ylabel("Activity Level", color="#ffffff", fontsize=9)
ax.set_ylim(0, 1.5)
ax.set_xlim(0, 50)
ax.tick_params(colors="#ffffff", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#ffffff")

if plotted:
    ax.legend(
        facecolor="#0f172a", edgecolor="#334155",
        labelcolor="#ffffff", fontsize=8,
        loc="upper right", framealpha=1
    )

plt.tight_layout(pad=0.8)
st.pyplot(fig, use_container_width=True)
plt.close()

# ── GLOSSARY (collapsible) ────────────────────────────────────────────────────
with st.expander("📖  Glossary", expanded=False):
    st.markdown("""
    <div class="glossary">
    <span>NF-κB</span> &nbsp;·&nbsp; <b>Alarm Signal</b><br>
    Transcription factor held inactive in the cytoplasm by IκBα. When freed, it enters the nucleus and switches on immune response genes.<br><br>
    <span>IκBα</span> &nbsp;·&nbsp; <b>The Brake</b><br>
    Inhibitory protein that binds and sequesters NF-κB. Degraded by IKK kinase (activated by TLR signalling) to release NF-κB. Rebuilt via negative feedback.<br><br>
    <span>IKK</span> &nbsp;·&nbsp; <b>The Trigger</b><br>
    Kinase complex activated downstream of TLR (Toll-like receptor) when bacteria are detected. Phosphorylates IκBα, marking it for degradation.
    </div>
    """, unsafe_allow_html=True)



