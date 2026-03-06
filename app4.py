import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.title("NF-κB – IκBα Dynamics During TLR Signalling")

# ----------------------------
# Pretreatment
# ----------------------------

st.sidebar.header("Pretreatment")

nfkb_inhib = st.sidebar.slider(
    "NF-κB Transport Inhibitor",
    0.0, 1.0, 0.0
)

ikba_inhib = st.sidebar.slider(
    "IκBα Activation Inhibitor",
    0.0, 1.0, 0.0
)


# ----------------------------
# Stimulus Button (replaces slider)
# ----------------------------

st.sidebar.header("Stimulus")

# Initialize session state
if "tlr_strength" not in st.session_state:
    st.session_state.tlr_strength = 0.0  # default = unstimulated

if st.sidebar.button("Stimulate (Bacteria)"):
    st.session_state.tlr_strength = 0.2

if st.sidebar.button("Reset (No Stimulus)"):
    st.session_state.tlr_strength = 0.0

tlr_strength = st.session_state.tlr_strength

# ----------------------------
# Graph display controls
# ----------------------------

st.sidebar.header("Graph Display")

show_nfkb = st.sidebar.checkbox("Show Nuclear NF-κB", value=True)
show_ikba = st.sidebar.checkbox("Show IκBα", value=True)


# ----------------------------
# Parameters
# ----------------------------

k_import = 1.0
k_export = 0.5
k_deg_I = 1.5
k_syn_I = 0.1
k_decay_I = 0.1
k_tnf = 1.5
k_tnf_decay = 0.3

k_tx = 0.2
k_mdeg = 0.05
k_tl = 0.1

# ----------------------------
# Initial Conditions
# ----------------------------

Nc0 = 1.0
Nn0 = 0.0
Im0 = 0.0
I0 = 1.0
TNF0 = 0.0

# ----------------------------
# Model
# ----------------------------

def nfkb_model(t, y):
    Nc, Nn, Im, I, TNF = y
    IKK = tlr_strength

    basal_synthesis = k_syn_I
    basal_decay = k_decay_I

    tlr_deg = k_deg_I * IKK * I * (1 - ikba_inhib)

    dIm = k_tx * Nn * (IKK / (IKK + 1e-6)) - k_mdeg * Im
    dI = basal_synthesis - basal_decay - tlr_deg + k_tl * Im

    if IKK > 0:
        degraded_fraction = tlr_deg / (k_deg_I * IKK * I + 1e-6)
        effective_import = k_import * Nc * degraded_fraction * (1 - nfkb_inhib)
        export_N = k_export * Nn * (1 + I)
    else:
        effective_import = 0.0
        export_N = 0.0

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

if show_nfkb:
    ax.plot(sol.t, sol.y[1], label="Nuclear NF-κB")

if show_ikba:
    ax.plot(sol.t, sol.y[3], label="IκBα")

ax.set_xlabel("Time")
ax.set_ylabel("Concentration / Activity")
ax.set_ylim(0, 3)
ax.legend()

st.pyplot(fig)

# ----------------------------
# Tasks
# ----------------------------

st.markdown("### Do the following tasks")
st.markdown("""
- Press **Stimulate (LPS)**  
    - Why does IkB decrease then increase?
    - Why does TNF increase in line with nuclear NFKB, but delayed?
    - Why does nuclear NFkB appear to decrease after an initial burst?

Question: What should inhibiting ikba and NFkB phosphorylation acomplish?

- Start increasing the NFkB phosphorylation inhibitor
    - What do you see?
- Reset the NFkB phosphorylation inhibitor and start increasing the ikba phosphorylation inhibitor
    - How does this differ to the NFkB phosphorylation inhibitor
""")






