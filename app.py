import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="QoS Classifier", page_icon="📶")
st.title("📶 QoS Service Class & Quality Predictor")
st.caption("NGN Assignment 17 – predicts the service class and quality level from network measurements")


@st.cache_resource
def load_models():
    return joblib.load("qos_service_class_model.pkl"), joblib.load("qos_quality_model.pkl")


cls_bundle, q_bundle = load_models()

with st.sidebar:
    st.header("Flow measurements")
    latency = st.number_input("Latency (ms)", 0.0, 3000.0, 60.0)
    jitter = st.number_input("Jitter (ms)", 0.0, 500.0, 8.0)
    loss = st.number_input("Packet loss (%)", 0.0, 100.0, 0.2)
    pkt = st.number_input("Avg packet size (bytes)", 64.0, 1500.0, 180.0)
    thr = st.number_input("Throughput (Mbps)", 0.01, 200.0, 0.09)

if st.button("Predict", type="primary"):
    # Model A: service class
    x = pd.DataFrame([dict(latency_ms=latency, jitter_ms=jitter, packet_loss_pct=loss,
                           pkt_size_bytes=pkt, throughput_mbps=thr)])[cls_bundle["features"]]
    model = cls_bundle["model"]
    probs = model.predict_proba(x)[0]
    classes = cls_bundle["label_encoder"].inverse_transform(model.classes_)
    service = classes[probs.argmax()]

    # Model B: quality level (needs one-hot service class)
    xq = pd.DataFrame([dict(latency_ms=latency, jitter_ms=jitter, packet_loss_pct=loss)])
    for c in q_bundle["features"]:
        if c not in xq:
            xq[c] = int(c == service)
    quality = q_bundle["model"].predict(xq[q_bundle["features"]])[0]

    # Threshold policy tier
    thr_cls = cls_bundle["thresholds"][service]
    vals = dict(latency_ms=latency, jitter_ms=jitter, packet_loss_pct=loss)
    exceeded = [k for k, v in vals.items() if v > thr_cls[k]]
    severe = any(v > 2 * thr_cls[k] for k, v in vals.items())
    tier = "Red" if (len(exceeded) >= 2 or severe) else "Amber" if exceeded else "Green"

    c1, c2, c3 = st.columns(3)
    c1.metric("Service class", service, f"{probs.max():.0%} confidence")
    c2.metric("Quality level", quality)
    c3.metric("Policy tier", tier)
    st.bar_chart(pd.Series(probs, index=classes, name="Probability"))
    st.write("**Thresholds for this class:**", thr_cls)
    if exceeded:
        st.warning("Parameters above threshold: " + ", ".join(exceeded))
