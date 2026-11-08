"""
Diamond Price Predictor
-----------------------
A friendly Gradio front-end for the model trained in diamond_prediction.ipynb.

Required files (saved by the notebook, keep them next to this app.py):
    diamond_price_model.pkl
    diamond_encoder.pkl
    diamond_scaler.pkl
    diamond_numerical_features.pkl
    diamond_categorical_features.pkl
    diamond_feature_columns.pkl

Run:
    pip install gradio joblib pandas numpy scikit-learn
    python app.py
"""

import joblib
import numpy as np
import pandas as pd
import gradio as gr

# --------------------------------------------------------------------------
# Load the model + preprocessing objects saved by the notebook
# --------------------------------------------------------------------------
model = joblib.load("diamond_price_model.pkl")
encoder = joblib.load("diamond_encoder.pkl")
scaler = joblib.load("diamond_scaler.pkl")
numerical_features = joblib.load("diamond_numerical_features.pkl")
categorical_features = joblib.load("diamond_categorical_features.pkl")
feature_columns = joblib.load("diamond_feature_columns.pkl")

CUTS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLORS = ["D", "E", "F", "G", "H", "I", "J"]
CLARITIES = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

CUT_NOTES = {
    "Fair": "Fair cut: the sparkle is modest.",
    "Good": "Good cut: decent brilliance for the price.",
    "Very Good": "Very Good cut: lots of light return.",
    "Premium": "Premium cut: excellent sparkle.",
    "Ideal": "Ideal cut: the best light performance there is.",
}
COLOR_NOTES = {
    "D": "colourless (the rarest)", "E": "colourless", "F": "colourless",
    "G": "nearly colourless", "H": "nearly colourless",
    "I": "nearly colourless, faint warmth", "J": "nearly colourless, faint warmth",
}
CLARITY_NOTES = {
    "I1": "visible inclusions", "SI2": "slight inclusions", "SI1": "slight inclusions",
    "VS2": "very small inclusions", "VS1": "very small inclusions",
    "VVS2": "barely-there inclusions", "VVS1": "barely-there inclusions",
    "IF": "internally flawless",
}


# --------------------------------------------------------------------------
# Prediction logic
# --------------------------------------------------------------------------
def build_features(carat, cut, color, clarity, depth, table, x, y, z):
    """Turn raw inputs into the exact feature layout used in training."""
    raw = pd.DataFrame([{
        "carat": carat, "cut": cut, "color": color, "clarity": clarity,
        "depth": depth, "table": table, "x": x, "y": y, "z": z,
    }])

    num = pd.DataFrame(
        scaler.transform(raw[numerical_features]), columns=numerical_features
    )
    cat = pd.DataFrame(
        encoder.transform(raw[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features),
    )
    return pd.concat([num, cat], axis=1).reindex(columns=feature_columns, fill_value=0)


def price_range(features, point_estimate):
    """A realistic range from the spread of the individual trees, if available."""
    try:
        preds = np.array([t.predict(features.values)[0] for t in model.estimators_])
        low, high = np.percentile(preds, [10, 90])
        # keep the range sensible around the point estimate
        return min(low, point_estimate), max(high, point_estimate)
    except Exception:
        return point_estimate * 0.92, point_estimate * 1.08


def sanity_notes(depth, x, y, z):
    """Gentle heads-up if the numbers don't look like they belong together."""
    notes = []
    if x and y and z:
        implied_depth = 2 * z / (x + y) * 100
        if abs(implied_depth - depth) > 3:
            notes.append(
                f"Your measurements imply a depth of about <b>{implied_depth:.1f}%</b>, "
                f"but you entered <b>{depth:.1f}%</b>. Worth double-checking."
            )
    return notes


def money(v, symbol="$"):
    return f"{symbol}{v:,.0f}"


def render_error(message):
    return f"""
    <div class="result-card error">
        <div class="eyebrow">Hmm, something's off</div>
        <p class="muted">{message}</p>
    </div>"""


def predict(carat, cut, color, clarity, depth, table, x, y, z, inr_rate):
    try:
        if None in (carat, depth, table, x, y, z):
            return render_error("Please fill in every measurement and I'll take a look.")
        if min(carat, depth, table, x, y, z) <= 0:
            return render_error("All measurements need to be greater than zero.")

        features = build_features(carat, cut, color, clarity, depth, table, x, y, z)
        price = float(model.predict(features)[0])
        low, high = price_range(features, price)

        inr_line = ""
        if inr_rate and inr_rate > 0:
            inr_line = f'<div class="inr">≈ {money(price * inr_rate, "₹")}</div>'

        per_carat = price / carat
        notes = sanity_notes(depth, x, y, z)
        notes_html = "".join(f'<div class="note">⚠️ {n}</div>' for n in notes)

        story = (
            f"A <b>{carat:.2f} ct</b> diamond with a <b>{cut.lower()}</b> cut, "
            f"<b>{color}</b> colour ({COLOR_NOTES[color]}) and <b>{clarity}</b> "
            f"clarity ({CLARITY_NOTES[clarity]})."
        )

        return f"""
        <div class="result-card">
            <div class="eyebrow">Estimated value</div>
            <div class="price">{money(price)}</div>
            {inr_line}
            <div class="range">Likely range&nbsp; {money(low)} &ndash; {money(high)}</div>
            <hr/>
            <p class="story">{story}</p>
            <div class="chips">
                <span class="chip">{money(per_carat)} per carat</span>
                <span class="chip">{CUT_NOTES[cut].split(':')[0]}</span>
                <span class="chip">{color} colour</span>
                <span class="chip">{clarity} clarity</span>
            </div>
            {notes_html}
            <p class="fine">Prices are in US dollars, based on historical sales data.
            Treat this as a well-informed starting point, not a quote.</p>
        </div>"""

    except Exception as e:  # noqa: BLE001
        return render_error(f"I couldn't work that out ({e}).")


# --------------------------------------------------------------------------
# Look & feel
# --------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Header */
#hero { text-align: center; padding: 28px 12px 8px; }
#hero h1 {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600; font-size: 2.5rem; margin: 0 0 6px;
    letter-spacing: -0.5px;
}
#hero p { font-size: 1.05rem; opacity: .75; margin: 0; }

/* Cards */
.panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 18px !important;
    padding: 18px 20px !important;
    background: var(--background-fill-primary);
}
.section-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.2rem; font-weight: 500; margin: 0 0 2px;
}
.section-sub { font-size: .9rem; opacity: .65; margin: 0 0 8px; }

/* Primary button */
#go-btn {
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 14px !important;
    background: linear-gradient(135deg, #4f46e5, #0ea5e9) !important;
    border: none !important; color: #fff !important;
    transition: transform .15s ease, box-shadow .15s ease;
}
#go-btn:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(79,70,229,.28); }

/* Result */
.result-card {
    border-radius: 18px; padding: 26px 26px 18px;
    background: linear-gradient(160deg, rgba(79,70,229,.10), rgba(14,165,233,.10));
    border: 1px solid rgba(79,70,229,.25);
    animation: rise .35s ease;
}
.result-card.error { background: rgba(239,68,68,.08); border-color: rgba(239,68,68,.3); }
.result-card.empty { text-align: center; padding: 42px 20px; }
.result-card .eyebrow {
    text-transform: uppercase; letter-spacing: .12em;
    font-size: .72rem; font-weight: 600; opacity: .6;
}
.result-card .price {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 3.2rem; font-weight: 600; line-height: 1.1; margin: 6px 0 2px;
}
.result-card .inr { font-size: 1.1rem; opacity: .75; }
.result-card .range { font-size: .95rem; opacity: .75; margin-top: 4px; }
.result-card hr { border: none; border-top: 1px solid rgba(128,128,128,.25); margin: 16px 0; }
.result-card .story { line-height: 1.55; margin: 0 0 12px; }
.result-card .muted { opacity: .75; margin: 6px 0 0; }
.result-card .fine { font-size: .8rem; opacity: .55; margin: 14px 0 0; line-height: 1.45; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    font-size: .8rem; padding: 5px 12px; border-radius: 999px;
    background: rgba(128,128,128,.14); font-weight: 500;
}
.note {
    margin-top: 12px; font-size: .88rem; padding: 10px 12px;
    border-radius: 10px; background: rgba(245,158,11,.14);
}
@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

footer { display: none !important; }
"""

EMPTY_STATE = """
<div class="result-card empty">
    <div style="font-size:2.6rem">💎</div>
    <div class="eyebrow" style="margin-top:6px">Your estimate will appear here</div>
    <p class="muted">Describe your diamond on the left, then press <b>Estimate price</b>.</p>
</div>"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="sky",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
)

# Handy starting points
EXAMPLES = [
    [0.30, "Ideal", "G", "VS1", 61.5, 56.0, 4.3, 4.3, 2.65, None],
    [0.70, "Premium", "F", "VS2", 61.8, 58.0, 5.7, 5.7, 3.5, None],
    [1.00, "Very Good", "H", "SI1", 62.0, 57.0, 6.4, 6.4, 3.97, None],
    [2.01, "Ideal", "D", "VVS1", 61.6, 56.0, 8.1, 8.1, 5.0, None],
]

# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
with gr.Blocks(title="Diamond Price Predictor", theme=theme, css=CSS) as demo:

    gr.HTML("""
    <div id="hero">
        <h1>💎 What's your diamond worth?</h1>
        <p>Tell me about the stone and I'll give you a price estimate.</p>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ---------- Left: inputs ----------
        with gr.Column(scale=6):
            with gr.Group(elem_classes="panel"):
                gr.HTML('<div class="section-title">The 4 Cs</div>'
                        '<div class="section-sub">The qualities jewellers care about most.</div>')
                carat = gr.Slider(0.2, 5.0, value=0.7, step=0.01, label="Carat (weight)")
                with gr.Row():
                    cut = gr.Dropdown(CUTS, value="Ideal", label="Cut")
                    color = gr.Dropdown(COLORS, value="G", label="Colour (D is best)")
                    clarity = gr.Dropdown(CLARITIES, value="VS1", label="Clarity")

            with gr.Group(elem_classes="panel"):
                gr.HTML('<div class="section-title">Measurements</div>'
                        '<div class="section-sub">You can find these on the grading report.</div>')
                with gr.Row():
                    depth = gr.Number(value=61.5, label="Depth %", minimum=40, maximum=80)
                    table = gr.Number(value=57.0, label="Table %", minimum=40, maximum=100)
                with gr.Row():
                    x = gr.Number(value=5.7, label="Length (mm)", minimum=0.1)
                    y = gr.Number(value=5.7, label="Width (mm)", minimum=0.1)
                    z = gr.Number(value=3.5, label="Height (mm)", minimum=0.1)

            with gr.Accordion("Show the price in rupees too (optional)", open=False):
                inr_rate = gr.Number(
                    value=None,
                    label="Exchange rate: ₹ per $1",
                    info="Enter today's rate, e.g. the one on your bank or Google.",
                )

            with gr.Row():
                go = gr.Button("Estimate price", variant="primary", elem_id="go-btn", scale=3)
                reset = gr.Button("Reset", scale=1)

        # ---------- Right: result ----------
        with gr.Column(scale=5):
            result = gr.HTML(EMPTY_STATE)
            gr.Examples(
                examples=EXAMPLES,
                inputs=[carat, cut, color, clarity, depth, table, x, y, z, inr_rate],
                label="Not sure where to start? Try one of these",
            )

    all_inputs = [carat, cut, color, clarity, depth, table, x, y, z, inr_rate]
    go.click(predict, inputs=all_inputs, outputs=result)

    reset.click(
        lambda: (0.7, "Ideal", "G", "VS1", 61.5, 57.0, 5.7, 5.7, 3.5, None, EMPTY_STATE),
        outputs=all_inputs + [result],
    )

    gr.HTML(
        '<p style="text-align:center;font-size:.82rem;opacity:.5;margin:18px 0 6px">'
        "Built with a machine-learning model trained on historical diamond sales. "
        "Not a certified appraisal.</p>"
    )

if __name__ == "__main__":
    demo.launch()
