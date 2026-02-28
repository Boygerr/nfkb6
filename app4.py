import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.title("NF-κB – IκBα – TNF Dynamics During TLR Signalling")

st.markdown("""
Model of NF-κB activation, IκBα feedback, and TNF gene expression 
during Toll-like receptor stimulation.
""")

# ----------------------------
# Sidebar Controls
# ----------------------------

st.sidebar.header("Stimulus")
tlr_strength = st.sidebar.slider("TLR Activation (IKK strength)", 0.0, 5.0, 2.0)

st.sidebar.header("Inhibitors")
nfkb_inhib = st.sidebar.slider("NF-κB Phosphorylation Inhibitor (blocks nuclear entry)", 0.0, 1.0, 0.0)
ikba_inhib = st.sidebar.slider("IκBα Phosphorylation Inhibitor (blocks IκB degradation)", 0.0, 1.0, 0.0)

st.sidebar.header("Kinetic Parameters")
k_import = 1.0
k_export = 0.5
k_deg_I = 1.5
k_syn_I = 0.1    # basal synthesis/decay balanced at 1
k_decay_I = 0.1
k_tnf = 1.5
k_tnf_decay = 0.3

# mRNA / translation rates (slow to prevent unrealistic spikes)
k_tx = 0.2
k_mdeg = 0.05
k_tl = 0.1

# ----------------------------
# Initial Conditions
# ----------------------------
Nc0 = 1.0
Nn0 = 0.0
Im0 = 0.0   # IκB mRNA
I0 = 1.0
TNF0 = 0.0

# ----------------------------
# Model
# ----------------------------
def nfkb_model(t, y):
    Nc, Nn, Im, I, TNF = y

    IKK = tlr_strength

    # Basal IκB synthesis and decay (keeps I stable at 1)
    basal_synthesis = k_syn_I
    basal_decay = k_decay_I

    # TLR-dependent IκB degradation (reduced by inhibitor)
    tlr_deg = k_deg_I * IKK * I * (1 - ikba_inhib)

    # mRNA dynamics (slow, NF-κB induced)
    dIm = k_tx * Nn - k_mdeg * Im

    # IκB protein dynamics
    dI = basal_synthesis - basal_decay - tlr_deg + k_tl * Im

    # NF-κB nuclear import proportional to TLR-induced IκB degradation
    degraded_fraction = tlr_deg / (tlr_deg + 1e-6) if tlr_deg > 0 else 0.0
    effective_import = k_import * Nc * degraded_fraction * (1 - nfkb_inhib)
    export_N = k_export * Nn

    # TNF transcription driven by nuclear NF-κB
    dTNF = k_tnf * Nn - k_tnf_decay * TNF

    dNc = -effective_import + export_N
    dNn = effective_import - export_N

    return [dNc, dNn, dIm, dI, dTNF]

# ----------------------------
# Solve ODE
# ----------------------------
t_span = [0, 50]
t_eval = np.linspace(0, 50, 1000)

sol = solve_ivp(nfkb_model, t_span, [Nc0, Nn0, Im0, I0, TNF0], t_eval=t_eval)

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(8,5))

ax.plot(sol.t, sol.y[1], label="Nuclear NF-κB")
ax.plot(sol.t, sol.y[3], label="IκBα")
ax.plot(sol.t, sol.y[4], label="TNF Gene Expression")

ax.set_xlabel("Time")
ax.set_ylabel("Concentration / Activity")
ax.legend()
st.pyplot(fig)

# ----------------------------
# Discussion Prompts
# ----------------------------
st.markdown("### Discussion Prompts")
st.markdown("""
- Why does TNF follow nuclear NF-κB with a delay?
- What happens when NF-κB phosphorylation is inhibited?
- Why does blocking IκB degradation suppress everything?
- Is inhibiting NF-κB entry equivalent to blocking IκB degradation?
""")
