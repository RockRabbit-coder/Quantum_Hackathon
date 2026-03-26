"""
Combined Detection Lab
Quantum Optics (Homodyne vs Heterodyne) + RF Signal Mixing
Run with: streamlit run combined_detection_lab.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Combined Detection Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Teko:wght@400;600;700&family=Roboto+Condensed:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    background-color: #050a14 !important;
    color: #c8d8f0 !important;
    font-family: 'Space Mono', monospace !important;
}

/* Tab bar */
.stTabs [data-baseweb="tab-list"] {
    background: #080f1e !important;
    border-bottom: 1px solid #1a2a4a !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: #0a1628 !important;
    color: #4a7090 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    border-radius: 4px 4px 0 0 !important;
    border: 1px solid #1a2a4a !important;
    border-bottom: none !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: #050a14 !important;
    color: #00e5ff !important;
    border-color: #00e5ff !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f1e 0%, #060c19 100%) !important;
    border-right: 1px solid #1a2a4a !important;
}
section[data-testid="stSidebar"] * { color: #a0b8d8 !important; font-size: 0.78rem !important; }

/* Slider accent */
div[data-baseweb="slider"] div[role="slider"] { background: #00e5ff !important; }

/* Headers */
h1 { font-family: 'Orbitron', monospace !important; color: #00e5ff !important;
     letter-spacing: 0.08em; text-shadow: 0 0 24px #00e5ff66; }
h2, h3 { font-family: 'Orbitron', monospace !important; color: #4dd8ff !important;
          letter-spacing: 0.04em; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0a1628 !important; border: 1px solid #1a3060 !important;
    border-radius: 6px; padding: 8px 14px !important;
}
[data-testid="metric-container"] label { color: #5888bb !important; font-size: 0.7rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00e5ff !important; }

/* Buttons */
.stButton > button {
    background: #00e5ff15 !important;
    border: 1px solid #00e5ff !important;
    color: #00e5ff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em;
    border-radius: 3px;
    padding: 6px 16px;
}
.stButton > button:hover { background: #00e5ff30 !important; box-shadow: 0 0 12px #00e5ff44; }

/* Badges */
.row-badge, .step-badge {
    display: inline-block; background: #0d1e3a; border: 1px solid #1e3a6a;
    color: #4da8d8; font-family: 'Orbitron', monospace; font-size: 0.62rem;
    padding: 3px 12px; border-radius: 2px; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 4px;
}

/* Mode banner */
.mode-banner {
    border-radius: 3px; padding: 10px 20px;
    font-family: 'Orbitron', monospace; font-size: 0.9rem;
    letter-spacing: 0.1em; text-align: center; margin: 6px 0 10px 0;
}
.mode-homo   { background: #001f14; border: 1px solid #00e5ff; color: #00e5ff; }
.mode-hetero { background: #1a0e00; border: 1px solid #ffaa00; color: #ffaa00; }

/* Callout */
.callout {
    background: #070e1f; border-left: 3px solid #00e5ff;
    padding: 10px 16px; border-radius: 4px; margin: 8px 0;
    font-size: 0.72rem; line-height: 1.7; color: #8aafd0;
    font-family: 'Space Mono', monospace;
}
.callout b  { color: #00e5ff; }
.callout .warn { color: #ffaa00; }
</style>
""", unsafe_allow_html=True)

# ── Shared Matplotlib Style Helpers ──────────────────────────────────────────
PLT_BG  = "#050a14"
PLT_AX  = "#08111f"
GRID_C  = "#0e1e38"
TICK_C  = "#3a5580"
LABEL_C = "#6a90bb"
TITLE_C = "#90bde0"

# Quantum palette
Q_CYAN    = "#00e5ff"
Q_AMBER   = "#ffaa00"
Q_LIME    = "#7fff6a"
Q_MAGENTA = "#ff4dff"
Q_CORAL   = "#ff6060"
Q_VIOLET  = "#9d7aff"

# RF palette
R_TEAL   = "#00ffc3"
R_AMBER  = "#ff9500"
R_SKY    = "#00aaff"
R_CORAL  = "#ff4d6d"
R_LIME   = "#aaff44"
R_VIOLET = "#b06aff"
R_WHITE  = "#c8dde4"

def apply_ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PLT_AX)
    ax.spines[:].set_color(GRID_C)
    ax.tick_params(colors=TICK_C, labelsize=7)
    ax.xaxis.label.set_color(LABEL_C); ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_color(LABEL_C); ax.yaxis.label.set_size(8)
    ax.set_title(title, color=TITLE_C, fontsize=8.5, pad=6, fontfamily="monospace")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID_C, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

def make_fig(rows=1, cols=2, height=2.8):
    fig = plt.figure(figsize=(13, height * rows), facecolor=PLT_BG)
    gs  = gridspec.GridSpec(rows, cols, figure=fig,
                            hspace=0.55, wspace=0.35,
                            left=0.06, right=0.97, top=0.88, bottom=0.14)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(cols)] for r in range(rows)]
    return fig, axes

# ── Low-Pass Filter ───────────────────────────────────────────────────────────
def lowpass(signal, cutoff_norm, order=4):
    cutoff_norm = np.clip(cutoff_norm, 0.01, 0.49)
    b, a = butter(order, cutoff_norm, btype='low')
    return filtfilt(b, a, signal)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:1.7rem;margin-bottom:0">
  🔬📡 COMBINED DETECTION LAB
</h1>
<p style="color:#3a6090;font-size:0.72rem;margin-top:4px;font-family:monospace">
  QUANTUM OPTICS  ·  RF SIGNAL MIXING  ·  Homodyne vs Heterodyne Simulator
</p>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_quantum, tab_rf, tab_compare = st.tabs([
    "🔬  QUANTUM DETECTION",
    "📡  RF MIXING LAB",
    "⚖  SIDE-BY-SIDE COMPARISON",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUANTUM DETECTION
# ═════════════════════════════════════════════════════════════════════════════
with tab_quantum:

    # ── Sidebar (quantum controls) ────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p style="font-size:0.85rem;color:#00e5ff;font-family:Orbitron,monospace;letter-spacing:0.08em">⚙ QUANTUM CONTROLS</p>',
                    unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("**QUBIT SIGNAL**")
        amplitude = st.slider("Amplitude",          0.1, 1.0,  0.4, 0.05, key="q_amp")
        phase_pi  = st.slider("Phase (× π rad)",    0.0, 2.0,  0.0, 0.1,  key="q_phase")
        phase     = phase_pi * np.pi

        st.markdown("---")
        st.markdown("**LOCAL OSCILLATOR**")
        lo_strength = st.slider("LO Strength",      1.0, 10.0, 3.0, 0.5,  key="q_lo")

        st.markdown("---")
        st.markdown("**HETERODYNE SETTINGS**")
        freq_offset = st.slider("Frequency Offset (Δf × base)", 0.5, 4.0, 1.5, 0.1, key="q_dfreq")

        st.markdown("---")
        st.markdown("**DISPLAY**")
        n_cycles = st.slider("Cycles shown",        2, 8, 4, key="q_cyc")

    # Signal generation
    t_end = n_cycles / 1.0
    t     = np.linspace(0, t_end, 2000)
    f_base    = 1.0
    f_lo_hd   = f_base
    f_lo_ht   = f_base + freq_offset

    qubit         = amplitude * np.sin(2 * np.pi * f_base * t + phase)
    lo_homodyne   = lo_strength * np.sin(2 * np.pi * f_lo_hd * t)
    lo_heterodyne = lo_strength * np.sin(2 * np.pi * f_lo_ht * t)

    def balanced_detection(lo, sig):
        return ((lo + sig)**2 - (lo - sig)**2) / 2

    hd_sum    = lo_homodyne  + qubit
    hd_diff   = lo_homodyne  - qubit
    ht_sum    = lo_heterodyne + qubit
    ht_diff   = lo_heterodyne - qubit
    hd_result = balanced_detection(lo_homodyne,  qubit)
    ht_result = balanced_detection(lo_heterodyne, qubit)

    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Signal Amplitude", f"{amplitude:.2f}")
    m2.metric("Signal Phase",     f"{phase_pi:.1f}π rad")
    m3.metric("LO Strength",      f"{lo_strength:.1f}×")
    m4.metric("LO Freq (HD)",     f"{f_lo_hd:.1f} Hz  (matched)")
    m5.metric("LO Freq (HT)",     f"{f_lo_ht:.1f} Hz  (+Δ{freq_offset:.1f})")

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 1 — Input signals
    st.markdown('<div class="row-badge">ROW 1 — THE INPUT SIGNALS</div>', unsafe_allow_html=True)
    fig1, ax1 = make_fig(rows=1, cols=2, height=2.6)

    ax = ax1[0][0]
    ax.plot(t, qubit,       color=Q_CYAN,  lw=1.4, label=f"Qubit  (f={f_base} Hz)")
    ax.plot(t, lo_homodyne, color=Q_AMBER, lw=1.2, ls="--", alpha=0.85,
            label=f"LO  (f={f_lo_hd} Hz  ← matched)")
    apply_ax_style(ax, title="HOMODYNE INPUT  — Frequencies Matched",
                   xlabel="Time (s)", ylabel="Amplitude")
    ax.legend(fontsize=7, facecolor="#0a1628", edgecolor="#1a3060", labelcolor="white", loc="upper right")

    ax = ax1[0][1]
    ax.plot(t, qubit,         color=Q_CYAN,    lw=1.4, label=f"Qubit  (f={f_base} Hz)")
    ax.plot(t, lo_heterodyne, color=Q_MAGENTA, lw=1.2, ls="--", alpha=0.85,
            label=f"LO  (f={f_lo_ht:.1f} Hz  ← offset by Δ{freq_offset})")
    apply_ax_style(ax, title="HETERODYNE INPUT  — LO Frequency Offset",
                   xlabel="Time (s)", ylabel="Amplitude")
    ax.legend(fontsize=7, facecolor="#0a1628", edgecolor="#1a3060", labelcolor="white", loc="upper right")

    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # ROW 2 — Mixing
    st.markdown('<div class="row-badge">ROW 2 — MIXING AT THE BEAM SPLITTER</div>', unsafe_allow_html=True)
    fig2, ax2 = make_fig(rows=1, cols=2, height=2.6)

    for col, (lo, s, d, color_s, color_d, label_prefix) in enumerate([
        (lo_homodyne,  hd_sum, hd_diff, Q_LIME,   Q_CORAL,  "HD"),
        (lo_heterodyne,ht_sum, ht_diff, Q_VIOLET, Q_AMBER,  "HT"),
    ]):
        ax = ax2[0][col]
        ax.plot(t, s, color=color_s, lw=1.2, label=f"{label_prefix}: LO + Qubit  (port A)")
        ax.plot(t, d, color=color_d, lw=1.2, ls="--", alpha=0.9,
                label=f"{label_prefix}: LO − Qubit  (port B)")
        title = ("HOMODYNE MIXING  — Stable Envelope"
                 if col == 0 else "HETERODYNE MIXING  — Shifting Envelope")
        apply_ax_style(ax, title=title, xlabel="Time (s)", ylabel="Amplitude")
        ax.legend(fontsize=7, facecolor="#0a1628", edgecolor="#1a3060", labelcolor="white", loc="upper right")

    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # ROW 3 — Detection results
    st.markdown('<div class="row-badge">ROW 3 — FINAL DETECTION OUTPUT</div>', unsafe_allow_html=True)
    fig3, ax3 = make_fig(rows=1, cols=2, height=2.8)

    ax = ax3[0][0]
    ax.fill_between(t, hd_result, alpha=0.18, color=Q_CYAN)
    ax.plot(t, hd_result, color=Q_CYAN, lw=1.6, label="HD result = 2·LO·Qubit")
    env_val = 2 * lo_strength * amplitude * np.cos(phase)
    ax.axhline(env_val, color=Q_AMBER, lw=0.9, ls=":", alpha=0.7,
               label=f"DC component ≈ {env_val:.2f}")
    apply_ax_style(ax, title="HOMODYNE OUTPUT  — Steady Amplified Wave",
                   xlabel="Time (s)", ylabel="Photocurrent  (arb. units)")
    ax.legend(fontsize=7, facecolor="#0a1628", edgecolor="#1a3060", labelcolor="white")

    ax = ax3[0][1]
    beat_env = 2 * lo_strength * amplitude * np.ones_like(t)
    ax.fill_between(t, ht_result, alpha=0.18, color=Q_MAGENTA)
    ax.plot(t, ht_result,  color=Q_MAGENTA, lw=1.6, label="HT result = beat note")
    ax.plot(t,  beat_env,  color=Q_AMBER,   lw=0.9, ls=":", alpha=0.7,
            label=f"Envelope ±{2*lo_strength*amplitude:.2f}")
    ax.plot(t, -beat_env,  color=Q_AMBER,   lw=0.9, ls=":", alpha=0.7)
    apply_ax_style(ax, title="HETERODYNE OUTPUT  — Pulsing Beat Note",
                   xlabel="Time (s)", ylabel="Photocurrent  (arb. units)")
    ax.legend(fontsize=7, facecolor="#0a1628", edgecolor="#1a3060", labelcolor="white")

    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # Callout
    st.markdown("""
<div class="callout">
  <b>WHY does Homodyne give a steady wave while Heterodyne pulses?</b><br><br>
  The balanced detector computes <code>Result = ((LO + Q)² − (LO − Q)²) / 2 = 2·LO·Q</code>.<br><br>
  • <b>Homodyne</b>: LO and Qubit share the same frequency, so their product reduces to a
  <em>constant DC term</em> proportional to <code>cos(φ)</code> plus a high-frequency ripple — a stable,
  amplified readout of the quadrature set by your phase slider.<br><br>
  • <b>Heterodyne</b>: The LO runs at <code>ω + Δω</code>, so the product contains
  <code>sin(Δω·t)</code> — a <em>beat note</em> oscillating at the offset frequency Δf.
  The signal envelope swells and collapses at Δf, encoding <em>both</em> quadratures
  simultaneously at the cost of ½ the SNR (3 dB vacuum noise penalty).
</div>
""", unsafe_allow_html=True)

    # Signal power comparison
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="row-badge">SIGNAL POWER COMPARISON</div>', unsafe_allow_html=True)

    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 1.8), facecolor=PLT_BG)
    hd_rms = np.sqrt(np.mean(hd_result**2))
    ht_rms = np.sqrt(np.mean(ht_result**2))
    labels  = ["Homodyne\n(matched LO)", "Heterodyne\n(offset LO)"]
    values  = [hd_rms, ht_rms]
    colors  = [Q_CYAN, Q_MAGENTA]

    ax = axes4[0]
    bars = ax.bar(labels, values, color=colors, width=0.4, zorder=3,
                  edgecolor="#0a1628", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, color="white", fontfamily="monospace")
    apply_ax_style(ax, title="RMS Signal Power", ylabel="RMS Photocurrent")
    ax.set_ylim(0, max(values) * 1.3)

    ax = axes4[1]
    N   = len(ht_result)
    dt  = t[1] - t[0]
    fft = np.abs(np.fft.rfft(ht_result)) / N
    freq_axis = np.fft.rfftfreq(N, d=dt)
    mask = freq_axis < f_lo_ht * 3
    ax.fill_between(freq_axis[mask], fft[mask], alpha=0.4, color=Q_VIOLET)
    ax.plot(freq_axis[mask], fft[mask], color=Q_VIOLET, lw=1.2)
    ax.axvline(freq_offset, color=Q_AMBER, lw=1.0, ls="--",
               label=f"Beat freq = Δf = {freq_offset:.1f} Hz")
    apply_ax_style(ax, title="Heterodyne FFT — Beat Frequency Spectrum",
                   xlabel="Frequency (Hz)", ylabel="|FFT|")
    ax.legend(fontsize=7, facecolor="#0a1628", edgecolor="#1a3060", labelcolor="white")

    plt.tight_layout(pad=0.5)
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    st.markdown("""
<p style="color:#243860;font-size:0.65rem;text-align:center;margin-top:24px;font-family:monospace">
  QUANTUM DETECTION LAB · Balanced detector formula: Result = ((LO+Q)²−(LO−Q)²)/2 = 2·LO·Q
</p>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — RF MIXING LAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_rf:

    # RF sidebar controls rendered here (sidebar is shared, so we gate with tab logic)
    with st.sidebar:
        st.markdown("---")
        st.markdown('<p style="font-size:0.85rem;color:#00ffc3;font-family:Orbitron,monospace;letter-spacing:0.08em">📡 RF CONTROLS</p>',
                    unsafe_allow_html=True)

        st.markdown("**CARRIER / MESSAGE**")
        carrier_mhz = st.slider("Carrier Frequency (MHz)", 50, 200, 100, 5,  key="r_car")
        msg_freq_hz = st.slider("Message Frequency (kHz)", 1,  20,   5,  1,  key="r_msg")
        msg_amp_r   = st.slider("Message Amplitude",       0.1, 1.0, 0.6, 0.05, key="r_mamp")
        carrier_amp = st.slider("Carrier Amplitude",       0.5, 2.0, 1.0, 0.1,  key="r_camp")

        st.markdown("**LOCAL OSCILLATOR (LO)**")
        lo_mhz = st.slider("LO Frequency (MHz)", 50, 220, carrier_mhz, 1, key="r_lo")
        auto_match_r = st.button("⟳  AUTO-MATCH  (Homodyne)", key="r_auto")

        st.markdown("**DISPLAY**")
        show_cycles = st.slider("Message cycles shown", 2, 8, 3, key="r_cyc")
        filter_bw   = st.slider("LP Filter BW (×msg freq)", 1.5, 6.0, 2.5, 0.5, key="r_fbw")

    if auto_match_r:
        lo_mhz = carrier_mhz

    homodyne    = (lo_mhz == carrier_mhz)
    mode_label  = "HOMODYNE" if homodyne else "HETERODYNE"
    if_mhz      = abs(carrier_mhz - lo_mhz)

    # Time axis (normalised: message period = 1 s)
    f_msg_r = 1.0
    f_car_r = carrier_mhz / msg_freq_hz
    f_lo_r  = lo_mhz      / msg_freq_hz
    f_if_r  = abs(f_car_r - f_lo_r)

    t_end_r = show_cycles / f_msg_r
    t_r     = np.linspace(0, t_end_r, int(show_cycles * 1000))

    message_r = msg_amp_r  * np.sin(2 * np.pi * f_msg_r * t_r)
    carrier_r = carrier_amp * (1 + message_r) * np.cos(2 * np.pi * f_car_r * t_r)
    lo_wave_r = np.cos(2 * np.pi * f_lo_r * t_r)
    mixed_r   = carrier_r * lo_wave_r

    fs_r      = 1.0 / (t_r[1] - t_r[0])
    cutoff_n  = (filter_bw * f_msg_r) / (fs_r / 2)
    recovered_r = lowpass(mixed_r, cutoff_n)

    # Header
    st.markdown("""
<h2 style="font-size:1.4rem;margin-bottom:0">
  📡 RF SIGNAL MIXING LAB
</h2>
<p style="color:#1e6070;font-size:0.72rem;margin-top:2px;font-family:monospace">
  CLASSICAL HETERODYNE &amp; HOMODYNE SIMULATION  ·  CARRIER → MIXER → FILTER → RECOVERY
</p>
""", unsafe_allow_html=True)

    if homodyne:
        st.markdown('<div class="mode-banner mode-homo">▶ HOMODYNE MODE — LO FREQUENCY MATCHED TO CARRIER → DC BASEBAND OUTPUT</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="mode-banner mode-hetero">▶ HETERODYNE MODE — IF = |{carrier_mhz} − {lo_mhz}| = {if_mhz} MHz BEAT NOTE</div>',
                    unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Carrier",   f"{carrier_mhz} MHz")
    m2.metric("Message",   f"{msg_freq_hz} kHz")
    m3.metric("LO Freq",   f"{lo_mhz} MHz")
    m4.metric("IF (Beat)", f"{if_mhz} MHz")
    m5.metric("Mode",      mode_label)
    m6.metric("LP Cutoff", f"{filter_bw:.1f}× fₘ")

    st.markdown("<br>", unsafe_allow_html=True)

    # STEP 1
    st.markdown('<div class="step-badge">STEP 1 — INPUT SIGNALS</div>', unsafe_allow_html=True)
    fig_r1, ax_r1 = make_fig(1, 2, 2.6)

    ax = ax_r1[0][0]
    ax.plot(t_r, carrier_r, color=R_TEAL,  lw=0.9, label=f"AM Carrier  ({carrier_mhz} MHz)")
    ax.plot(t_r, message_r, color=R_AMBER, lw=1.2, ls="--", alpha=0.75,
            label=f"Message envelope  ({msg_freq_hz} kHz)")
    apply_ax_style(ax, title=f"GRAPH 1 — INCOMING CARRIER  ({carrier_mhz} MHz AM)",
                   xlabel="Time (normalised)", ylabel="Amplitude")
    ax.legend(fontsize=7, facecolor="#010e15", edgecolor="#083040", labelcolor="white")

    ax = ax_r1[0][1]
    lo_color = R_TEAL if homodyne else R_AMBER
    lo_label = f"LO  ({lo_mhz} MHz)  {'← MATCHED' if homodyne else f'← OFFSET  Δ={if_mhz} MHz'}"
    ax.plot(t_r, lo_wave_r, color=lo_color, lw=1.1, label=lo_label)
    ax.plot(t_r, carrier_amp * np.cos(2 * np.pi * f_car_r * t_r) * 0.3,
            color=R_TEAL, lw=0.5, ls=":", alpha=0.4, label="Carrier ref (faded)")
    apply_ax_style(ax, title=f"GRAPH 2 — LOCAL OSCILLATOR  ({lo_mhz} MHz)",
                   xlabel="Time (normalised)", ylabel="Amplitude")
    ax.legend(fontsize=7, facecolor="#010e15", edgecolor="#083040", labelcolor="white")

    st.pyplot(fig_r1, use_container_width=True)
    plt.close(fig_r1)

    # STEP 2
    st.markdown('<div class="step-badge">STEP 2 — MULTIPLICATION (MIXING) & LOW-PASS FILTER</div>',
                unsafe_allow_html=True)
    fig_r2, ax_r2 = make_fig(1, 2, 2.6)

    ax = ax_r2[0][0]
    ax.plot(t_r, mixed_r,     color=R_CORAL, lw=0.7, alpha=0.85,
            label="Signal × LO  (sum + difference freqs)")
    ax.plot(t_r, recovered_r, color=R_LIME,  lw=1.4, label="LP filtered (preview)")
    apply_ax_style(ax, title="GRAPH 3 — MIXER OUTPUT  (High-freq jitter + Beat Note)",
                   xlabel="Time (normalised)", ylabel="Amplitude")
    ax.legend(fontsize=7, facecolor="#010e15", edgecolor="#083040", labelcolor="white")
    ax.annotate("High-freq 'jitter'\n(sum freq 2·fc)", xy=(t_r[80], mixed_r[80]),
                xytext=(t_r[80] + t_end_r*0.06, mixed_r[80] + 0.5),
                color=R_CORAL, fontsize=6.5, fontfamily="monospace",
                arrowprops=dict(arrowstyle="->", color=R_CORAL, lw=0.8))

    ax = ax_r2[0][1]
    beat_color = R_TEAL if homodyne else R_AMBER
    if homodyne:
        scale    = np.max(np.abs(recovered_r)) if np.max(np.abs(recovered_r)) > 0 else 1
        rec_norm = recovered_r / scale * msg_amp_r
        ax.fill_between(t_r, rec_norm, alpha=0.15, color=R_TEAL)
        ax.plot(t_r, rec_norm,  color=R_TEAL,  lw=1.6, label="Recovered signal  (DC baseband)")
        ax.plot(t_r, message_r, color=R_AMBER, lw=1.0, ls="--", alpha=0.7, label="Original message")
        ax.axhline(0, color=R_WHITE, lw=0.4, ls=":", alpha=0.4)
    else:
        beat_envelope = msg_amp_r * np.abs(np.sin(np.pi * f_if_r * t_r))
        scale    = np.max(np.abs(recovered_r)) if np.max(np.abs(recovered_r)) > 0 else 1
        rec_norm = recovered_r / scale * msg_amp_r
        ax.fill_between(t_r, rec_norm, alpha=0.15, color=R_AMBER)
        ax.plot(t_r, rec_norm,      color=R_AMBER,  lw=1.4, label=f"IF Beat Note  ({if_mhz} MHz)")
        ax.plot(t_r, beat_envelope, color=R_VIOLET, lw=0.9, ls="--", alpha=0.8, label="Beat envelope")
        ax.plot(t_r, message_r,     color=R_TEAL,   lw=0.8, ls=":", alpha=0.6, label="Original message")

    apply_ax_style(ax, title="GRAPH 4 — RECOVERED MESSAGE  (After Low-Pass Filter)",
                   xlabel="Time (normalised)", ylabel="Amplitude")
    ax.legend(fontsize=7, facecolor="#010e15", edgecolor="#083040", labelcolor="white")

    st.pyplot(fig_r2, use_container_width=True)
    plt.close(fig_r2)

    # STEP 3 — Frequency domain
    st.markdown('<div class="step-badge">STEP 3 — FREQUENCY DOMAIN ANALYSIS</div>',
                unsafe_allow_html=True)
    fig_r3, ax_r3 = make_fig(1, 2, 2.4)

    def plot_spectrum(ax, sig, fs_norm, color, title, max_freq=None, label=""):
        N   = len(sig)
        fft = np.abs(np.fft.rfft(sig)) / N
        fq  = np.fft.rfftfreq(N, d=1.0/fs_norm)
        if max_freq:
            mask = fq <= max_freq
            fq, fft = fq[mask], fft[mask]
        ax.fill_between(fq, fft, alpha=0.25, color=color)
        ax.plot(fq, fft, color=color, lw=1.1, label=label)
        apply_ax_style(ax, title=title, xlabel="Frequency (normalised)", ylabel="|FFT|")
        return fq, fft

    ax = ax_r3[0][0]
    plot_spectrum(ax, mixed_r, fs_r, R_CORAL,
                  "MIXER SPECTRUM — Sum & Difference Frequencies",
                  max_freq=f_car_r * 2.4)
    for freq, col, lbl in [
        (f_car_r - f_lo_r, R_AMBER, f"|fc−flo|={if_mhz} MHz (IF)"),
        (f_car_r + f_lo_r, R_CORAL, f"fc+flo (image)"),
        (f_car_r,          R_TEAL,  f"fc={carrier_mhz} MHz"),
    ]:
        if freq >= 0:
            ax.axvline(freq, color=col, lw=0.8, ls="--", alpha=0.7, label=lbl)
    ax.legend(fontsize=6.5, facecolor="#010e15", edgecolor="#083040", labelcolor="white")

    ax = ax_r3[0][1]
    plot_spectrum(ax, recovered_r, fs_r, R_TEAL if homodyne else R_AMBER,
                  "RECOVERED SPECTRUM — After Low-Pass Filter",
                  max_freq=f_car_r * 0.5)
    ax.axvline(f_msg_r, color=R_LIME, lw=0.9, ls="--", alpha=0.8,
               label=f"Message  ({msg_freq_hz} kHz)")
    if not homodyne and f_if_r > 0:
        ax.axvline(f_if_r, color=R_AMBER, lw=0.9, ls="--", alpha=0.8,
                   label=f"IF beat  ({if_mhz} MHz)")
    ax.legend(fontsize=6.5, facecolor="#010e15", edgecolor="#083040", labelcolor="white")

    st.pyplot(fig_r3, use_container_width=True)
    plt.close(fig_r3)

    # Explanation callout
    if homodyne:
        st.markdown("""
<div class="callout">
<b>▶ HOMODYNE MODE — How it works</b><br><br>
The LO frequency exactly matches the carrier (<b>fₗₒ = fc</b>). When the carrier and LO are multiplied:<br>
<code>cos(2πfct) × cos(2πfct) = ½[1 + cos(4πfct)]</code><br><br>
The LP filter removes the high-frequency term, leaving just the <b>½ DC component</b> modulated by the message.
Result: a clean, steady amplitude shift proportional to the message — <b>no intermediate frequency needed</b>, directly at baseband.
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="callout">
<b>▶ HETERODYNE MODE — How it works</b><br><br>
The LO runs at <b>fₗₒ = {lo_mhz} MHz</b> while the carrier is at <b>fc = {carrier_mhz} MHz</b>.
Multiplication produces two new frequencies:<br>
<code>cos(2πfct) × cos(2πflot) = ½[cos(2π(fc−flo)t) + cos(2π(fc+flo)t)]</code><br><br>
• <span class="warn">Difference (IF) = |fc − flo| = {if_mhz} MHz</span> — the useful <b>beat note</b> carrying the message<br>
• Sum (image) = fc + flo = {carrier_mhz + lo_mhz} MHz — discarded by the LP filter<br><br>
The electronics only need to process <b>{if_mhz} MHz</b> instead of {carrier_mhz} MHz. This is why all
superheterodyne radios (AM, FM, TV, Wi-Fi) use this technique.
</div>
""", unsafe_allow_html=True)

    # Mode comparison table
    st.markdown('<div class="step-badge">REFERENCE — MODE COMPARISON TABLE</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
<div class="callout" style="border-left-color:#00ffc3">
<b style="color:#00ffc3">HOMODYNE (fₗₒ = fc)</b><br>
─────────────────────<br>
Output type     : <b>DC / Baseband</b><br>
Phase sensitive : <b>Yes</b> (detects quadrature)<br>
SNR             : Optimal (no image noise)<br>
Complexity      : Requires phase-locked LO<br>
Use cases       : Optical coherent, Quantum<br>
Filter needed   : Very narrow (≈ msg BW)<br>
</div>
""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
<div class="callout" style="border-left-color:#ffaa00">
<b style="color:#ffaa00">HETERODYNE (fₗₒ ≠ fc)</b><br>
─────────────────────<br>
Output type     : <b>IF Beat Note  ({if_mhz} MHz)</b><br>
Phase sensitive : No (amplitude detection)<br>
SNR             : −3 dB vs homodyne<br>
Complexity      : Simple, robust LO<br>
Use cases       : AM/FM radio, TV, Wi-Fi<br>
Filter needed   : Bandpass around IF<br>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<p style="color:#082030;font-size:0.6rem;text-align:center;margin-top:18px;font-family:monospace">
RF MIXING LAB · MIXER FORMULA: output = Signal × LO → LP FILTER → Recovered Message
</p>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIDE-BY-SIDE COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("""
<h2 style="font-size:1.3rem;margin-bottom:4px">⚖ DOMAIN COMPARISON</h2>
<p style="color:#3a6090;font-size:0.72rem;margin-top:0;font-family:monospace">
  Quantum Optics  vs  RF / Classical Radio  —  Same Principles, Different Scales
</p>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="callout" style="border-left-color:#00e5ff">
<b style="color:#00e5ff;font-size:0.85rem">🔬 QUANTUM OPTICS</b><br><br>
<b>Signal domain  :</b> Optical (light quanta / qubits)<br>
<b>Freq scale     :</b> 100s of THz (visible / near-IR)<br>
<b>LO type        :</b> Coherent laser (local oscillator)<br>
<b>Detector       :</b> Balanced photodetector pair<br>
<b>Math           :</b> ((LO+Q)² − (LO−Q)²) / 2 = 2·LO·Q<br>
<b>Homodyne out   :</b> DC quadrature (phase-sensitive)<br>
<b>Heterodyne out :</b> Beat note at Δf (both quadratures)<br>
<b>Noise limit    :</b> Shot noise / vacuum fluctuations<br>
<b>Key use        :</b> QKD, quantum state tomography<br>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="callout" style="border-left-color:#00ffc3">
<b style="color:#00ffc3;font-size:0.85rem">📡 RF / CLASSICAL RADIO</b><br><br>
<b>Signal domain  :</b> RF electromagnetic waves<br>
<b>Freq scale     :</b> kHz – GHz (radio spectrum)<br>
<b>LO type        :</b> Voltage-controlled oscillator (VCO)<br>
<b>Detector       :</b> Electronic multiplier (mixer chip)<br>
<b>Math           :</b> carrier × LO → LP filter → message<br>
<b>Homodyne out   :</b> Baseband message (direct conversion)<br>
<b>Heterodyne out :</b> IF beat note (easier to process)<br>
<b>Noise limit    :</b> Thermal / Johnson noise<br>
<b>Key use        :</b> AM/FM radio, Wi-Fi, radar<br>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Shared principle diagram (matplotlib)
    fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(13, 2.0), facecolor=PLT_BG)
    t_demo = np.linspace(0, 4, 1000)

    # Homodyne summary
    ax = axes_cmp[0]
    sig_demo = 0.4 * np.sin(2 * np.pi * t_demo)
    lo_demo  = 3.0 * np.sin(2 * np.pi * t_demo)
    out_demo = ((lo_demo + sig_demo)**2 - (lo_demo - sig_demo)**2) / 2
    dc_line  = 2 * 3.0 * 0.4 * np.cos(0)
    ax.plot(t_demo, out_demo, color=Q_CYAN,  lw=1.2, label="Balanced output")
    ax.axhline(dc_line, color=Q_AMBER, lw=0.9, ls=":", label=f"DC ≈ {dc_line:.2f}")
    apply_ax_style(ax, title="HOMODYNE — Steady DC readout (both domains)",
                   xlabel="Time", ylabel="Output")
    ax.legend(fontsize=7, facecolor="#0a1628", edgecolor="#1a3060", labelcolor="white")

    # Heterodyne summary
    ax = axes_cmp[1]
    lo_ht_d  = 3.0 * np.sin(2 * np.pi * (1 + 1.5) * t_demo)
    out_ht_d = ((lo_ht_d + sig_demo)**2 - (lo_ht_d - sig_demo)**2) / 2
    env_ht   = 2 * 3.0 * 0.4 * np.ones_like(t_demo)
    ax.plot(t_demo, out_ht_d, color=Q_MAGENTA, lw=1.2, label="Beat output")
    ax.plot(t_demo,  env_ht,  color=Q_AMBER, lw=0.8, ls=":", label="Envelope ±max")
    ax.plot(t_demo, -env_ht,  color=Q_AMBER, lw=0.8, ls=":")
    apply_ax_style(ax, title="HETERODYNE — Oscillating beat note (both domains)",
                   xlabel="Time", ylabel="Output")
    ax.legend(fontsize=7, facecolor="#0a1628", edgecolor="#1a3060", labelcolor="white")

    plt.tight_layout(pad=0.4)
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close(fig_cmp)

    st.markdown("""
<div class="callout" style="border-left-color:#9d7aff">
<b>🔗 THE UNIFYING PRINCIPLE</b><br><br>
Both labs exploit the same mathematical identity:<br>
<code>2 · A·sin(ωt + φ) · B·sin(ωt)  =  AB·cos(φ)  −  AB·cos(2ωt + φ)</code><br><br>
• When LO and signal share the same frequency (<b>Homodyne</b>), the product collapses to a
  DC term <code>AB·cos(φ)</code> — encoding the <em>phase</em> in both optical and RF systems.<br><br>
• When LO is offset by Δω (<b>Heterodyne</b>), the cosine term becomes <code>AB·cos(Δω·t + φ)</code> —
  a beat note oscillating at the <em>difference</em> frequency. In quantum optics this encodes both
  quadratures simultaneously; in radio it creates a convenient intermediate frequency (IF) for
  downstream electronics.<br><br>
• The only difference between the two tabs is <b>scale</b>: quantum optics operates at ~100 THz with
  photon shot noise, while RF operates at MHz–GHz with thermal noise. The math is identical.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<p style="color:#243860;font-size:0.62rem;text-align:center;margin-top:20px;font-family:monospace">
  COMBINED DETECTION LAB  ·  Quantum Optics + RF Mixing  ·  Unified Homodyne / Heterodyne Simulator
</p>
""", unsafe_allow_html=True)
