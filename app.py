
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diamond Dynamics",
    page_icon="💎",
    layout="wide"
)

@st.cache_resource
def load_models():
    regression = joblib.load(
        "Diamond_Dynamics/diamond_model_bundle.pkl"
    )
    clustering = joblib.load(
        "Diamond_Dynamics/diamond_clustering_bundle.pkl"
    )
    return regression, clustering

regression, clustering = load_models()

st.title("💎 Diamond Dynamics")
st.subheader("Diamond Price Prediction & Market Segmentation")
st.write(
    "Enter diamond specifications to predict its estimated "
    "price and identify its market segment."
)

st.divider()

st.header("Diamond Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    carat = st.number_input(
        "Carat",
        min_value=0.01,
        max_value=5.0,
        value=0.50,
        step=0.01
    )

    cut = st.selectbox(
        "Cut",
        ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    )

    color = st.selectbox(
        "Color",
        ["D", "E", "F", "G", "H", "I", "J"]
    )

with col2:
    clarity = st.selectbox(
        "Clarity",
        ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    )

    depth = st.number_input(
        "Depth (%)",
        min_value=40.0,
        max_value=80.0,
        value=61.5
    )

    table = st.number_input(
        "Table (%)",
        min_value=40.0,
        max_value=80.0,
        value=57.0
    )

with col3:
    x = st.number_input(
        "Length (x mm)",
        min_value=0.1,
        max_value=11.0,
        value=5.0
    )

    y = st.number_input(
        "Width (y mm)",
        min_value=0.1,
        max_value=11.0,
        value=5.0
    )

    z = st.number_input(
        "Depth (z mm)",
        min_value=0.1,
        max_value=7.0,
        value=3.0
    )

if st.button("Predict Diamond Price", type="primary"):

    if x <= 0 or y <= 0 or z <= 0:
        st.error("Diamond dimensions must be greater than zero.")

    else:
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
            # Price prediction
            processed_input = regression["preprocessor"].transform(
                input_data
            )

            predicted_price = regression["model"].predict(
                processed_input
            )[0]

            # Market segmentation
            cluster_input = input_data[
                clustering["numeric_features"] +
                clustering["categorical_features"]
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

            with result1:
                st.metric(
                    "Estimated Diamond Price",
                    f"₹ {predicted_price:,.2f}"
                )

            with result2:
                st.metric(
                    "Market Segment",
                    cluster_name
                )

            st.success("Diamond analysis completed successfully!")

            st.subheader("Input Details")
            st.dataframe(input_data, use_container_width=True)

            st.caption(
                "Price is a model estimate, not a live market quotation."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")
