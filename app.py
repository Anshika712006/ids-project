import streamlit as st, pandas as pd, numpy as np, joblib
from tensorflow import keras

st.set_page_config(page_title="Explainable AI IDS", layout="wide")

@st.cache_resource
def load():
    ae = keras.models.load_model("ae_model.keras")
    return ae, joblib.load("scaler.pkl"), joblib.load("features.pkl"), float(np.load("threshold.npy"))

ae, scaler, feat, thr = load()

st.title("Explainable AI-Based Intrusion Detection")
st.caption("Autoencoder trained only on normal traffic. Alerts are explained by feature-wise reconstruction error.")

tab1, tab2, tab3 = st.tabs(["Detect", "Insights", "About"])

with tab1:
    src = st.radio("Data source", ["Sample traffic", "Upload CSV"], horizontal=True)
    df = None
    if src == "Sample traffic":
        df = pd.read_csv("sample_traffic.csv")
    else:
        up = st.file_uploader("Upload CSV", type="csv")
        if up is not None:
            df = pd.read_csv(up)

    if df is not None:
        X = df.reindex(columns=feat, fill_value=0).astype("float32")
        Xs = np.clip(scaler.transform(X), 0, 1)
        rec = ae.predict(Xs, verbose=0)
        ferr = (Xs - rec) ** 2
        score = ferr.mean(axis=1)
        df["score"] = score.round(4)
        df["status"] = np.where(score > thr, "ALERT", "Normal")

        a, b, c = st.columns(3)
        a.metric("Rows", len(df))
        b.metric("Alerts", int((score > thr).sum()))
        c.metric("Threshold", round(thr, 4))

        show = [x for x in ["category", "score", "status"] if x in df.columns]
        st.dataframe(df[show], height=250)

        alerts = df.index[df["status"] == "ALERT"].tolist()
        if alerts:
            i = st.selectbox("Pick an alert to explain", alerts)
            top = np.argsort(ferr[i])[-5:][::-1]
            exp = pd.DataFrame({"feature": [feat[j] for j in top], "error": ferr[i][top]})
            st.subheader("Why was this flagged?")
            st.bar_chart(exp.set_index("feature"))

with tab2:
    st.subheader("Which features drive each attack type")
    st.image("heatmap.png")
    st.subheader("Alert rate by traffic type")
    st.table(pd.DataFrame({
        "Type": ["Normal (false alarm)", "DoS", "Probe", "R2L", "U2R"],
        "Alert rate %": [13.9, 92.4, 96.2, 83.9, 79.1]}))

with tab3:
    st.write("Dataset: NSL-KDD. Model: Autoencoder (trained on normal traffic only). "
             "Baseline: Isolation Forest. Explainability: feature-wise reconstruction error.")