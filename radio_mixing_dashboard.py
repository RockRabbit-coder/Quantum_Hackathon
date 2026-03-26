"""
RF Signal Mixing Lab — Classical Heterodyne & Homodyne Dashboard
Simulates radio/telecom mixing, filtering, and message recovery.
Run with: streamlit run radio_mixing_dashboard.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RF Mixing Lab",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS / Fonts ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Teko:wght@400;600;700&family=Roboto+Condensed:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    background-color: #010b0f !important;
    color: #b0ccd4 !important;
    font-family: 'Roboto Condensed', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #01131a 0%, #010e14 100%) !important;
    border-right: 1px solid #0a2a35 !important;
}
section[data-testid="stSidebar"] * { color: #7aabb8 !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.78rem !important; }

/* Slider thumb */
div[data-baseweb="slider"] div[role="slider"] { background: #00ffc3 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #010f16 !important;
    border: 1px solid #083040 !important;
    border-radius: 3px;
    padding: 8px 14px !important;
}
[data-testid="metric-container"] label { color: #2a6a80 !important; font-size: 0.65rem !important; font-family: 'Share Tech Mono', monospace !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00ffc3 !important; font-family: 'Teko', sans-serif !important; font-size: 1.6rem !important; }

/* Buttons */
.stButton > button {
    background: #00ffc315 !important;
    border: 1px solid #00ffc3 !important;
    color: #00ffc3 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em;
    border-radius: 2px;
    padding: 6px 18px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #00ffc330 !important;
    box-shadow: 0 0 12px #00ffc355;
}
.stButton > button:active { background: #00ffc350 !important; }

/* Badge */
.step-badge {
    display: inline-block;
    background: #010f16;
    border: 1px solid #083a50;
    color: #00ffc3;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 12px;
    border-radius: 2px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2px;
}

/* Mode banner */
.mode-banner {
    border-radius: 3px;
    padding: 10px 20px;
    font-family: 'Teko', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.12em;
    text-align: center;
    margin: 6px 0 10px 0;
}
.mode-homo  { background: #001f14; border: 1px solid #00ffc3; color: #00ffc3; }
.mode-hetero{ background: #1a0e00; border: 1px solid #ff9500; color: #ff9500; }

/* Callout */
.callout {
    background: #010f16;
    border-left: 3px solid #00ffc3;
    padding: 10px 16px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    line-height: 1.7;
    color: #5a9aaa;
    margin: 8px 0;
}
.callout b { color: #00ffc3; }
.callout .warn { color: #ff9500; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib Theme ──────────────────────────────────────────────────────────
BG     = "#010b0f"
AX_BG  = "#010e15"
GRID   = "#061e28"
TICK   = "#1e5060"
LAB    = "#3a7a90"
TITLE  = "#60a0b8"

TEAL   = "#00ffc3"
AMBER  = "#ff9500"
SKY    = "#00aaff"
CORAL  = "#ff4d6d"
LIME   = "#aaff44"
VIOLET = "#b06aff"
WHITE  = "#c8dde4"

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=TICK, labelsize=7)
    ax.xaxis.label.set_color(LAB);  ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_color(LAB);  ax.yaxis.label.set_size(8)
    ax.set_title(title, color=TITLE, fontsize=8.5, pad=6, fontfamily="monospace")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, lw=0.6, ls="--", alpha=0.8)
    ax.set_axisbelow(True)

def make_fig(rows, cols, h):
    fig = plt.figure(figsize=(13, h * rows), facecolor=BG)
    gs  = gridspec.GridSpec(rows, cols, figure=fig,
                            hspace=0.55, wspace=0.32,
                            left=0.06, right=0.97, top=0.90, bottom=0.12)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(cols)] for r in range(rows)]
    return fig, axes

# ── Low-Pass Filter ───────────────────────────────────────────────────────────
def lowpass(signal, cutoff_norm, order=4):
    """Butterworth LP filter; cutoff_norm in (0, 1) relative to Nyquist."""
    cutoff_norm = np.clip(cutoff_norm, 0.01, 0.49)
    b, a = butter(order, cutoff_norm, btype='low')
    return filtfilt(b, a, signal)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-size:1rem;color:#00ffc3;font-family:Teko,sans-serif;letter-spacing:0.12em">📡 RF MIXING LAB</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**CARRIER / MESSAGE**")
    carrier_mhz = st.slider("Carrier Frequency (MHz)", 50, 200, 100, 5)
    msg_freq_hz = st.slider("Message Frequency (kHz)", 1, 20, 5, 1)
    msg_amp     = st.slider("Message Amplitude",       0.1, 1.0, 0.6, 0.05)
    carrier_amp = st.slider("Carrier Amplitude",       0.5, 2.0, 1.0, 0.1)

    st.markdown("---")
    st.markdown("**LOCAL OSCILLATOR (LO)**")
    lo_mhz_default = carrier_mhz
    lo_mhz = st.slider("LO Frequency (MHz)", 50, 220, lo_mhz_default, 1)

    auto_match = st.button("⟳  AUTO-MATCH  (Homodyne Mode)")

    st.markdown("---")
    st.markdown("**DISPLAY**")
    show_cycles = st.slider("Message cycles shown", 2, 8, 3)
    filter_bw   = st.slider("LP Filter Bandwidth  (×msg freq)", 1.5, 6.0, 2.5, 0.5,
                             help="How wide the low-pass filter is relative to the message frequency")

# ── Auto-match logic ──────────────────────────────────────────────────────────
if auto_match:
    lo_mhz = carrier_mhz
    st.session_state["lo_override"] = carrier_mhz

if "lo_override" in st.session_state and not auto_match:
    pass  # slider already holds new value after re-render

homodyne = (lo_mhz == carrier_mhz)
mode_label = "HOMODYNE" if homodyne else "HETERODYNE"
if_mhz = abs(carrier_mhz - lo_mhz)

# ── Time Axis ─────────────────────────────────────────────────────────────────
# Work in normalised time: message period = 1 s for clarity
# All MHz → relative units (we scale so carrier fits on screen)
# Sampling: 1000 pts / message cycle
f_msg  = 1.0                          # normalised message freq
f_car  = carrier_mhz / msg_freq_hz    # carrier in units of f_msg
f_lo   = lo_mhz      / msg_freq_hz
f_if   = abs(f_car - f_lo)

t_end = show_cycles / f_msg
t     = np.linspace(0, t_end, int(show_cycles * 1000))

# ── Signal Generation ─────────────────────────────────────────────────────────
message  = msg_amp * np.sin(2 * np.pi * f_msg * t)
carrier  = carrier_amp * (1 + message) * np.cos(2 * np.pi * f_car * t)   # AM modulation
lo_wave  = np.cos(2 * np.pi * f_lo * t)

# Mixer output: multiply
mixed    = carrier * lo_wave

# Low-pass filter — cutoff at filter_bw × message freq (normalised to Nyquist)
fs       = 1.0 / (t[1] - t[0])
cutoff_n = (filter_bw * f_msg) / (fs / 2)
recovered = lowpass(mixed, cutoff_n)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-family:'Teko',sans-serif;font-size:2rem;color:#00ffc3;
           letter-spacing:0.1em;margin-bottom:0;text-shadow:0 0 20px #00ffc355">
  📡 RF SIGNAL MIXING LAB
</h1>
<p style="font-family:'Share Tech Mono',monospace;color:#1e6070;font-size:0.72rem;margin-top:2px">
  CLASSICAL HETERODYNE &amp; HOMODYNE SIMULATION  ·  CARRIER → MIXER → FILTER → RECOVERY
</p>
""", unsafe_allow_html=True)

# Mode banner
if homodyne:
    st.markdown('<div class="mode-banner mode-homo">▶ HOMODYNE MODE — LO FREQUENCY MATCHED TO CARRIER → DC BASEBAND OUTPUT</div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div class="mode-banner mode-hetero">▶ HETERODYNE MODE — IF = |{carrier_mhz} − {lo_mhz}| = {if_mhz} MHz BEAT NOTE</div>',
                unsafe_allow_html=True)

# Metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Carrier",      f"{carrier_mhz} MHz")
m2.metric("Message",      f"{msg_freq_hz} kHz")
m3.metric("LO Freq",      f"{lo_mhz} MHz")
m4.metric("IF (Beat)",    f"{if_mhz} MHz")
m5.metric("Mode",         mode_label)
m6.metric("LP Cutoff",    f"{filter_bw:.1f}× fₘ")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Carrier Wave  +  Local Oscillator   (side by side)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">STEP 1 — INPUT SIGNALS</div>', unsafe_allow_html=True)

fig1, ax1 = make_fig(1, 2, 2.6)

# Graph 1: Carrier (AM modulated)
ax = ax1[0][0]
ax.plot(t, carrier, color=TEAL,  lw=0.9, label=f"AM Carrier  ({carrier_mhz} MHz)")
ax.plot(t, message, color=AMBER, lw=1.2, ls="--", alpha=0.75,
        label=f"Message envelope  ({msg_freq_hz} kHz)")
style_ax(ax, title=f"GRAPH 1 — INCOMING CARRIER  ({carrier_mhz} MHz AM)",
         xlabel="Time (normalised)", ylabel="Amplitude")
ax.legend(fontsize=7, facecolor="#010e15", edgecolor="#083040", labelcolor="white")

# Graph 2: Local Oscillator
ax = ax1[0][1]
lo_color = TEAL if homodyne else AMBER
lo_label = f"LO  ({lo_mhz} MHz)  {'← MATCHED' if homodyne else f'← OFFSET  Δ={if_mhz} MHz'}"
ax.plot(t, lo_wave, color=lo_color, lw=1.1, label=lo_label)
ax.plot(t, carrier_amp * np.cos(2 * np.pi * f_car * t) * 0.3,
        color=TEAL, lw=0.5, ls=":", alpha=0.4, label="Carrier ref (faded)")
style_ax(ax, title=f"GRAPH 2 — LOCAL OSCILLATOR  ({lo_mhz} MHz)",
         xlabel="Time (normalised)", ylabel="Amplitude")
ax.legend(fontsize=7, facecolor="#010e15", edgecolor="#083040", labelcolor="white")

st.pyplot(fig1, width='stretch')
plt.close(fig1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Mixer Output  +  LP Filter Comparison
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">STEP 2 — MULTIPLICATION (MIXING) & LOW-PASS FILTER</div>',
            unsafe_allow_html=True)

fig2, ax2 = make_fig(1, 2, 2.6)

# Graph 3: Raw mixer output
ax = ax2[0][0]
ax.plot(t, mixed, color=CORAL, lw=0.7, alpha=0.85,
        label="Signal × LO  (sum + difference freqs)")
ax.plot(t, recovered, color=LIME, lw=1.4, ls="-",
        label="LP filtered (preview)")
style_ax(ax, title="GRAPH 3 — MIXER OUTPUT  (High-freq jitter + Beat Note)",
         xlabel="Time (normalised)", ylabel="Amplitude")
ax.legend(fontsize=7, facecolor="#010e15", edgecolor="#083040", labelcolor="white")
ax.annotate("High-freq 'jitter'\n(sum freq 2·fc)", xy=(t[80], mixed[80]),
            xytext=(t[80] + t_end*0.06, mixed[80] + 0.5),
            color=CORAL, fontsize=6.5, fontfamily="monospace",
            arrowprops=dict(arrowstyle="->", color=CORAL, lw=0.8))

# Graph 4: Recovered message
ax = ax2[0][1]
beat_color = TEAL if homodyne else AMBER

if homodyne:
    # Normalise recovered to compare with message
    scale = np.max(np.abs(recovered)) if np.max(np.abs(recovered)) > 0 else 1
    rec_norm = recovered / scale * msg_amp
    ax.fill_between(t, rec_norm, alpha=0.15, color=TEAL)
    ax.plot(t, rec_norm,  color=TEAL,  lw=1.6, label="Recovered signal  (DC baseband)")
    ax.plot(t, message,   color=AMBER, lw=1.0, ls="--", alpha=0.7, label="Original message")
    ax.axhline(0, color=WHITE, lw=0.4, ls=":", alpha=0.4)
    note = "HOMODYNE: LO matched → message recovered at DC (baseband)"
else:
    beat_envelope = msg_amp * np.abs(np.sin(np.pi * f_if * t))
    scale = np.max(np.abs(recovered)) if np.max(np.abs(recovered)) > 0 else 1
    rec_norm = recovered / scale * msg_amp
    ax.fill_between(t, rec_norm, alpha=0.15, color=AMBER)
    ax.plot(t, rec_norm,      color=AMBER,  lw=1.4, label=f"IF Beat Note  ({if_mhz} MHz)")
    ax.plot(t, beat_envelope, color=VIOLET, lw=0.9, ls="--", alpha=0.8, label="Beat envelope")
    ax.plot(t, message,       color=TEAL,   lw=0.8, ls=":", alpha=0.6, label="Original message")
    note = f"HETERODYNE: IF beat = {if_mhz} MHz → easier for electronics to process"

style_ax(ax, title="GRAPH 4 — RECOVERED MESSAGE  (After Low-Pass Filter)",
         xlabel="Time (normalised)", ylabel="Amplitude")
ax.legend(fontsize=7, facecolor="#010e15", edgecolor="#083040", labelcolor="white")

st.pyplot(fig2, width='stretch')
plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Frequency Spectrum  (FFT)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">STEP 3 — FREQUENCY DOMAIN ANALYSIS</div>',
            unsafe_allow_html=True)

fig3, ax3 = make_fig(1, 2, 2.4)

def plot_spectrum(ax, sig, fs_norm, color, title, max_freq=None, label=""):
    N   = len(sig)
    fft = np.abs(np.fft.rfft(sig)) / N
    fq  = np.fft.rfftfreq(N, d=1.0/fs_norm)
    if max_freq:
        mask = fq <= max_freq
        fq, fft = fq[mask], fft[mask]
    ax.fill_between(fq, fft, alpha=0.25, color=color)
    ax.plot(fq, fft, color=color, lw=1.1, label=label)
    style_ax(ax, title=title, xlabel="Frequency (normalised)", ylabel="|FFT|")
    return fq, fft

fs_norm = 1.0 / (t[1] - t[0])

# Mixer output spectrum
ax = ax3[0][0]
fq, fft = plot_spectrum(ax, mixed, fs_norm, CORAL,
                        "MIXER SPECTRUM — Sum & Difference Frequencies",
                        max_freq=f_car * 2.4)
# Mark key frequencies
for freq, col, lbl in [
    (f_car - f_lo, AMBER, f"|fc−flo|={if_mhz} MHz (IF)"),
    (f_car + f_lo, CORAL, f"fc+flo (image)"),
    (f_car,        TEAL,  f"fc={carrier_mhz} MHz"),
]:
    if freq >= 0:
        ax.axvline(freq, color=col, lw=0.8, ls="--", alpha=0.7, label=lbl)
ax.legend(fontsize=6.5, facecolor="#010e15", edgecolor="#083040", labelcolor="white")

# Recovered signal spectrum
ax = ax3[0][1]
fq2, fft2 = plot_spectrum(ax, recovered, fs_norm, TEAL if homodyne else AMBER,
                          "RECOVERED SPECTRUM — After Low-Pass Filter",
                          max_freq=f_car * 0.5)
ax.axvline(f_msg, color=LIME, lw=0.9, ls="--", alpha=0.8,
           label=f"Message  ({msg_freq_hz} kHz)")
if not homodyne and f_if > 0:
    ax.axvline(f_if, color=AMBER, lw=0.9, ls="--", alpha=0.8,
               label=f"IF beat  ({if_mhz} MHz)")
ax.legend(fontsize=6.5, facecolor="#010e15", edgecolor="#083040", labelcolor="white")

st.pyplot(fig3, width='stretch')
plt.close(fig3)

# ─────────────────────────────────────────────────────────────────────────────
# Explanation callout
# ─────────────────────────────────────────────────────────────────────────────
if homodyne:
    st.markdown("""
<div class="callout">
<b>▶ HOMODYNE MODE — How it works</b><br><br>
The LO frequency exactly matches the carrier (<b>fₗₒ = fc</b>). When the carrier and LO are multiplied:<br>
<code>cos(2πfct) × cos(2πfct) = ½[1 + cos(4πfct)]</code><br><br>
The LP filter removes the high-frequency term <code>cos(4πfct)</code>, leaving just the <b>½ DC component</b>
modulated by the message. Result: a clean, steady amplitude shift proportional to the message —
<b>no intermediate frequency needed</b>, directly at baseband. Phase difference between LO and carrier
controls the recovered amplitude (hence the I/Q quadrature scheme in modern radios).
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
The electronics only need to process <b>{if_mhz} MHz</b> instead of {carrier_mhz} MHz — far easier to filter, amplify,
and digitise. This is why all superheterodyne radios (AM, FM, TV, Wi-Fi) use this technique.
Move the LO Frequency slider to change the IF and watch the beat note oscillate faster or slower.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side mode comparison mini-panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">REFERENCE — MODE COMPARISON TABLE</div>',
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
<div class="callout" style="border-left-color:#00ffc3">
<b style="color:#00ffc3">HOMODYNE (fₗₒ = fc)</b><br>
─────────────────────<br>
Output type  : <b>DC / Baseband</b><br>
Phase sensitive : <b>Yes</b> (detects quadrature)<br>
SNR          : Optimal (no image noise)<br>
Complexity   : Requires phase-locked LO<br>
Use cases    : Optical coherent, Quantum<br>
Filter needed: Very narrow (≈ msg BW)<br>
</div>
""", unsafe_allow_html=True)
with col_b:
    st.markdown(f"""
<div class="callout" style="border-left-color:#ff9500">
<b style="color:#ff9500">HETERODYNE (fₗₒ ≠ fc)</b><br>
─────────────────────<br>
Output type  : <b>IF Beat Note  ({if_mhz} MHz)</b><br>
Phase sensitive : No (amplitude detection)<br>
SNR          : −3 dB vs homodyne<br>
Complexity   : Simple, robust LO<br>
Use cases    : AM/FM radio, TV, Wi-Fi<br>
Filter needed: Bandpass around IF<br>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="color:#082030;font-size:0.6rem;text-align:center;margin-top:18px;font-family:'Share Tech Mono',monospace">
RF MIXING LAB · MIXER FORMULA: output = Signal × LO → LP FILTER → Recovered Message
</p>
""", unsafe_allow_html=True)
