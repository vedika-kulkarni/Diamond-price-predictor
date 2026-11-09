"""
Diamond Price Predictor
Pure Streamlit front-end for the model trained in diamond_prediction.ipynb.

Keep these model artifacts in the same directory as this file:
    diamond_price_model.pkl
    diamond_encoder.pkl
    diamond_scaler.pkl
    diamond_numerical_features.pkl
    diamond_categorical_features.pkl
    diamond_feature_columns.pkl

Run locally:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');

    .stApp {
        font-family: 'Inter', system-ui, sans-serif;
    }
    .main .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        text-align: center;
        padding: 1.2rem 0 1.6rem;
    }
    .hero h1 {
        font-family: 'Fraunces', Georgia, serif;
        font-size: clamp(2rem, 5vw, 3rem);
        margin-bottom: .35rem;
    }
    .hero p {
        opacity: .72;
        font-size: 1.05rem;
    }
    .panel {
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 18px;
        padding: 1.15rem 1.25rem .8rem;
        margin-bottom: 1rem;
    }
    .section-title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: .1rem;
    }
    .section-sub {
        opacity: .68;
        font-size: .9rem;
        margin-bottom: .8rem;
    }
    .result-card {
        border: 1px solid rgba(79,70,229,.3);
        border-radius: 18px;
        padding: 1.5rem;
        background: linear-gradient(145deg, rgba(79,70,229,.10), rgba(14,165,233,.08));
    }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .75rem;
        font-weight: 600;
        opacity: .65;
    }
    .price {
        font-family: 'Fraunces', Georgia, serif;
        font-size: clamp(2.3rem, 5vw, 3.2rem);
        font-weight: 600;
        line-height: 1.15;
        margin: .35rem 0;
    }
    .muted {
        opacity: .7;
    }
    .chip {
        display: inline-block;
        padding: .35rem .7rem;
        margin: .2rem .25rem .2rem 0;
        border-radius: 999px;
        background: rgba(128,128,128,.15);
        font-size: .82rem;
    }
    .fine {
        font-size: .8rem;
        opacity: .6;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Load trained model and preprocessing artifacts
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_FILES = {
    "model": "diamond_price_model.pkl",
    "encoder": "diamond_encoder.pkl",
    "scaler": "diamond_scaler.pkl",
    "numerical_features": "diamond_numerical_features.pkl",
    "categorical_features": "diamond_categorical_features.pkl",
    "feature_columns": "diamond_feature_columns.pkl",
}


@st.cache_resource
def load_artifacts():
    missing = [
        filename
        for filename in ARTIFACT_FILES.values()
        if not (BASE_DIR / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): " + ", ".join(missing)
        )

    return {
        key: joblib.load(BASE_DIR / filename)
        for key, filename in ARTIFACT_FILES.items()
    }


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
    "D": "colourless (the rarest)",
    "E": "colourless",
    "F": "colourless",
    "G": "nearly colourless",
    "H": "nearly colourless",
    "I": "nearly colourless, faint warmth",
    "J": "nearly colourless, faint warmth",
}
CLARITY_NOTES = {
    "I1": "visible inclusions",
    "SI2": "slight inclusions",
    "SI1": "slight inclusions",
    "VS2": "very small inclusions",
    "VS1": "very small inclusions",
    "VVS2": "barely-there inclusions",
    "VVS1": "barely-there inclusions",
    "IF": "internally flawless",
}

EXAMPLES = {
    "Balanced 0.30 ct": [0.30, "Ideal", "G", "VS1", 61.5, 56.0, 4.3, 4.3, 2.65],
    "Popular 0.70 ct": [0.70, "Premium", "F", "VS2", 61.8, 58.0, 5.7, 5.7, 3.5],
    "Classic 1.00 ct": [1.00, "Very Good", "H", "SI1", 62.0, 57.0, 6.4, 6.4, 3.97],
    "Large 2.01 ct": [2.01, "Ideal", "D", "VVS1", 61.6, 56.0, 8.1, 8.1, 5.0],
}


# ---------------------------------------------------------------------
# Prediction logic (preserves the notebook's preprocessing pipeline)
# ---------------------------------------------------------------------
def build_features(artifacts, carat, cut, color, clarity, depth, table, x, y, z):
    raw = pd.DataFrame(
        [{
            "carat": carat,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
        }]
    )

    numerical_features = artifacts["numerical_features"]
    categorical_features = artifacts["categorical_features"]
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]

    num = pd.DataFrame(
        scaler.transform(raw[numerical_features]),
        columns=numerical_features,
    )
    cat = pd.DataFrame(
        encoder.transform(raw[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features),
    )

    return pd.concat([num, cat], axis=1).reindex(
        columns=artifacts["feature_columns"],
        fill_value=0,
    )


def get_price_range(model, features, point_estimate):
    try:
        predictions = np.array(
            [tree.predict(features.values)[0] for tree in model.estimators_]
        )
        low, high = np.percentile(predictions, [10, 90])
        return min(float(low), point_estimate), max(float(high), point_estimate)
    except Exception:
        return point_estimate * 0.92, point_estimate * 1.08


def money(value, symbol="$"):
    return f"{symbol}{value:,.0f}"


def sanity_notes(depth, x, y, z):
    notes = []
    if x > 0 and y > 0 and z > 0:
        implied_depth = 2 * z / (x + y) * 100
        if abs(implied_depth - depth) > 3:
            notes.append(
                f"Your measurements imply a depth of about {implied_depth:.1f}%, "
                f"but you entered {depth:.1f}%. Worth double-checking."
            )
    return notes


# ---------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>💎 What's your diamond worth?</h1>
        <p>Describe your diamond and get a machine-learning price estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error("The prediction model could not be loaded.")
    st.exception(exc)
    st.info(
        "Upload all six .pkl model files to the same GitHub folder as app.py, "
        "then redeploy the app."
    )
    st.stop()

left, right = st.columns([1.12, 0.88], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">The 4 Cs</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">The qualities jewellers care about most.</div>',
        unsafe_allow_html=True,
    )

    example_name = st.selectbox(
        "Load an example (optional)",
        ["Custom diamond"] + list(EXAMPLES.keys()),
    )
    defaults = EXAMPLES.get(
        example_name,
        [0.70, "Ideal", "G", "VS1", 61.5, 57.0, 5.7, 5.7, 3.5],
    )

    carat = st.slider("Carat (weight)", 0.2, 5.0, float(defaults[0]), 0.01)
    c1, c2, c3 = st.columns(3)
    with c1:
        cut = st.selectbox("Cut", CUTS, index=CUTS.index(defaults[1]))
    with c2:
        color = st.selectbox(
            "Colour (D is best)", COLORS, index=COLORS.index(defaults[2])
        )
    with c3:
        clarity = st.selectbox(
            "Clarity", CLARITIES, index=CLARITIES.index(defaults[3])
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Measurements</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">You can find these on the grading report.</div>',
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    with m1:
        depth = st.number_input(
            "Depth %", min_value=40.0, max_value=80.0,
            value=float(defaults[4]), step=0.1,
        )
    with m2:
        table = st.number_input(
            "Table %", min_value=40.0, max_value=100.0,
            value=float(defaults[5]), step=0.1,
        )
    d1, d2, d3 = st.columns(3)
    with d1:
        x = st.number_input(
            "Length (mm)", min_value=0.1, value=float(defaults[6]), step=0.1
        )
    with d2:
        y = st.number_input(
            "Width (mm)", min_value=0.1, value=float(defaults[7]), step=0.1
        )
    with d3:
        z = st.number_input(
            "Height (mm)", min_value=0.1, value=float(defaults[8]), step=0.1
        )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Show the price in rupees too (optional)"):
        inr_rate = st.number_input(
            "Exchange rate: ₹ per $1",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help="Enter the exchange rate you want to use.",
        )

    estimate = st.button("💎 Estimate price", type="primary", use_container_width=True)

with right:
    st.markdown('<div class="section-title">Your estimate</div>', unsafe_allow_html=True)
    st.caption("Your result will appear here after you estimate the price.")

    if estimate:
        if min(carat, depth, table, x, y, z) <= 0:
            st.error("All measurements need to be greater than zero.")
        else:
            try:
                features = build_features(
                    artifacts, carat, cut, color, clarity, depth, table, x, y, z
                )
                model = artifacts["model"]
                price = float(model.predict(features)[0])
                low, high = get_price_range(model, features, price)
                per_carat = price / carat

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="eyebrow">Estimated value</div>
                        <div class="price">{money(price)}</div>
                        <div class="muted">Likely range: {money(low)} – {money(high)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if inr_rate > 0:
                    st.metric("Approximate value in Indian rupees", money(price * inr_rate, "₹"))

                st.markdown("#### Diamond summary")
                st.write(
                    f"A **{carat:.2f} ct** diamond with a **{cut.lower()}** cut, "
                    f"**{color}** colour ({COLOR_NOTES[color]}) and **{clarity}** "
                    f"clarity ({CLARITY_NOTES[clarity]})."
                )
                st.markdown(
                    f'<span class="chip">{money(per_carat)} per carat</span>'
                    f'<span class="chip">{CUT_NOTES[cut].split(":")[0]}</span>'
                    f'<span class="chip">{color} colour</span>'
                    f'<span class="chip">{clarity} clarity</span>',
                    unsafe_allow_html=True,
                )

                for note in sanity_notes(depth, x, y, z):
                    st.warning(note)

                st.caption(
                    "Prices are in US dollars, based on historical sales data. "
                    "Treat this as a well-informed starting point, not a certified appraisal."
                )
            except Exception as exc:
                st.error(f"I couldn't calculate the estimate: {exc}")
    else:
        st.markdown(
            """
            <div class="result-card" style="text-align:center;padding:2.4rem 1rem">
                <div style="font-size:2.8rem">💎</div>
                <div class="eyebrow">Your estimate will appear here</div>
                <p class="muted">Enter the diamond details on the left, then press
                <b>Estimate price</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<p style='text-align:center;opacity:.6;font-size:.85rem'>"
    "Built with a machine-learning model trained on historical diamond sales. "
    "Not a certified appraisal.</p>",
    unsafe_allow_html=True,
)
