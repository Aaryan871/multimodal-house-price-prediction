import streamlit as st
from PIL import Image
from predict import predict_price

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
def load_css():
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("🏠 House Price Prediction using Deep Learning")
st.write("Predict house price using image + tabular features.")

st.markdown("---")

# -------------------------------------------------
# Layout
# -------------------------------------------------

left, right = st.columns([1, 1])

# =================================================
# LEFT SIDE
# =================================================

with left:

    st.header("📤 Upload House Image")

    uploaded_image = st.file_uploader(
        "Choose a House Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

# =================================================
# RIGHT SIDE
# =================================================

with right:

    st.header("🏡 House Details")

    bed = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=20,
        value=3
    )

    bath = st.number_input(
        "Bathrooms",
        min_value=0,
        max_value=20,
        value=2
    )

    sqft = st.number_input(
        "Square Feet",
        min_value=100,
        value=1500
    )

    st.markdown("")

    predict_button = st.button(
        "Predict Price",
        use_container_width=True
    )

# =================================================
# Prediction
# =================================================

if predict_button:

    if uploaded_image is None:

        st.warning("Please upload a house image.")

    else:

        with st.spinner("Predicting House Price..."):

            price = predict_price(
                image,
                bed,
                bath,
                sqft
            )

        st.success("Prediction Completed")

        st.markdown("---")

        st.subheader("💰 Predicted House Price")

        st.metric(
            label="Estimated Price",
            value=f"${price:,.2f}"
        )
