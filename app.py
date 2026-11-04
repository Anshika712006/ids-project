import streamlit as st, pandas as pd, numpy as np, joblib

st.set_page_config(page_title="Explainable AI IDS", layout="wide")

@st.cache_resource
def load():
    w = np.load("ae_weights.npz")
    W = [w["arr_%d" % i] for i in range(12)]
    s = np.load("scaler_params.npz")
    feat = joblib.load("features.pkl")
    thr = float(np.load("threshold.npy"))
    return W, s["scale"], s["min"], feat, thr

W, sc, mn, feat, thr = load()

def forward(X):
    h = X
    for i in range(0, 10, 2):
        h = np.maximum(h @ W[i] + W[i + 1], 0)
    return 1 / (1 + np.exp(-(h @ W[10] + W[11])))

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
        X = df.reindex(columns=feat, fill_value=0).astype("float64").values
        Xs = np.clip(X * sc + mn, 0, 1)
        ferr = (Xs - forward(Xs)) ** 2
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