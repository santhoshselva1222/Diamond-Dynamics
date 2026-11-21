
import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

st.set_page_config(
    page_title="Diamond Dynamics",
    page_icon="💎",
    layout="wide"
)

@st.cache_resource
def load_models():
    preprocessor = joblib.load("diamond_preprocessor.pkl")

    regression = xgb.Booster()
    regression.load_model("diamond_xgboost_model.json")

    clustering = joblib.load("diamond_clustering_bundle.pkl")

    return preprocessor, regression, clustering


preprocessor, regression, clustering = load_models()

st.title("💎 Diamond Dynamics")
st.subheader("Diamond Price Prediction & Market Segmentation")

st.write(
    "Predict diamond prices and identify market segments using Machine Learning."
)

st.divider()
st.header("Diamond Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    carat = st.number_input("Carat", 0.01, 5.0, 0.50, 0.01)
    cut = st.selectbox(
        "Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    )
    color = st.selectbox(
        "Color", ["D", "E", "F", "G", "H", "I", "J"]
    )

with col2:
    clarity = st.selectbox(
        "Clarity",
        ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    )
    depth = st.number_input("Depth (%)", 40.0, 80.0, 61.5)
    table = st.number_input("Table (%)", 40.0, 80.0, 57.0)

with col3:
    x = st.number_input("Length (x mm)", 0.1, 11.0, 5.0)
    y = st.number_input("Width (y mm)", 0.1, 11.0, 5.0)
    z = st.number_input("Depth (z mm)", 0.1, 7.0, 3.0)

if st.button("Predict Diamond Price", type="primary"):

    volume = x * y * z
    dimension_ratio = (x + y) / (2 * z)

    if carat <= 0.5:
        carat_category = "Light"
    elif carat <= 1.5:
        carat_category = "Medium"
    else:
        carat_category = "Heavy"

    input_data = pd.DataFrame([{
        "carat": carat,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
        "volume": volume,
        "dimension_ratio": dimension_ratio,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "carat_category": carat_category
    }])

    try:
        processed = preprocessor.transform(input_data)

        prediction = regression.predict(
            xgb.DMatrix(processed)
        )[0]

        cluster_input = input_data[
            clustering["numeric_features"]
            + clustering["categorical_features"]
        ]

        processed_cluster = clustering["preprocessor"].transform(
            cluster_input
        )

        cluster_id = clustering["model"].predict(
            processed_cluster
        )[0]

        cluster_name = clustering["cluster_names"][int(cluster_id)]

        st.divider()
        st.header("Prediction Results")

        result1, result2 = st.columns(2)

        result1.metric(
            "Estimated Diamond Price",
            f"₹ {prediction:,.2f}"
        )

        result2.metric(
            "Market Segment",
            cluster_name
        )

        st.success("Diamond analysis completed successfully!")

        st.dataframe(input_data, use_container_width=True)

        st.caption(
            "Price is a model estimate, not a live market quotation."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
