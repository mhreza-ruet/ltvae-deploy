import io
import base64
import requests
import streamlit as st
from PIL import Image

API_BASE = "https://fscvjbxjt2.us-east-1.awsapprunner.com"

st.set_page_config(page_title="LTVAE SMILES Explorer", layout="wide")
st.title("LTVAE: Get 2D Graph, Latent Vectors and Properties from SMILES")

smiles = st.text_input("Enter SMILES", value="", placeholder="Paste a SMILES string…")

col1, col2 = st.columns([1, 1])

if st.button("Run model", type="primary") and smiles.strip():
    with st.spinner("Calling API…"):
        # 1) JSON prediction
        resp = requests.post(f"{API_BASE}/predict", json={"smiles": smiles})
        resp.raise_for_status()
        data = resp.json()

        # 2) PNG for 2D graph
        img_resp = requests.post(f"{API_BASE}/predict?view=true", json={"smiles": smiles})
        img_resp.raise_for_status()
        img = Image.open(io.BytesIO(img_resp.content))

    with col1:
        st.subheader("2D molecular graph (RDKit)")
        st.image(img, use_column_width=True)
        st.caption(f"Input SMILES: `{smiles}`")

    with col2:
        st.subheader("Predicted properties")
        st.json(data["properties"])

        st.subheader("Latent vector (first 10 dims)")
        st.write(data["latent"][:10])
        with st.expander("Show full latent vector (64 dims)"):
            st.write(data["latent"])
else:
    st.info("Enter a SMILES string and click **Run model**.")
