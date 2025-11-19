import io
import requests
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# ---- API base (your App Runner URL, no trailing slash) ----
API_BASE = "https://fscvjbxjt2.us-east-1.awsapprunner.com"

# ---- Page layout ----
st.set_page_config(page_title="LTVAE SMILES Explorer", layout="wide")
st.title("LTVAE: Latent Vectors & Property Prediction from SMILES")
st.markdown(
    "Enter a SMILES string to get:\n"
    "- A 2D RDKit rendering\n"
    "- A 64-dimensional latent vector\n"
    "- Predicted molecular properties"
)

# ---- Input ----
smiles = st.text_input(
    "Enter SMILES",
    value="",
    placeholder="Paste a SMILES string…",
)

def centered_responsive_image(img, width_pct=80):
    """
    Display a PIL image centered using a percentage width inside its container.
    """
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    html_code = f"""
    <div style='text-align: center;'>
        <img src="data:image/png;base64,{b64}" style="width:{width_pct}%;"/>
    </div>
    """

    st.markdown(html_code, unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

run_clicked = st.button("Run model", type="primary")

if run_clicked and smiles.strip():
    try:
        with st.spinner("Calling API…"):

            # 1) JSON prediction (view = False)
            json_resp = requests.post(
                f"{API_BASE}/predict",
                json={"smiles": smiles},
                timeout=30,
            )
            json_resp.raise_for_status()
            data = json_resp.json()

            # Handle invalid SMILES early
            if not data.get("valid", False):
                st.error("The SMILES is invalid according to the RDKit validator.")
                note = data.get("meta", {}).get("validator_note")
                if note:
                    st.info(f"Validator note: {note}")
                st.stop()

            # 2) PNG for 2D graph (view = True → returns image/png)
            img_resp = requests.post(
                f"{API_BASE}/predict?view=true",
                json={"smiles": smiles},
                timeout=30,
            )
            img_resp.raise_for_status()
            img = Image.open(io.BytesIO(img_resp.content))

        latent = data.get("latent", [])
        props = data.get("properties", {})
        meta = data.get("meta", {})

        # ------------------------------------------------------------------
        # Left column: 2D image
        # ------------------------------------------------------------------
        with col1:
            st.subheader("2D Molecular Graph (RDKit)")
            centered_responsive_image(img, width_pct=70)  # 70% of col width, centered
            st.caption(f"Input SMILES: `{smiles}`")

        # ------------------------------------------------------------------
        # Right column: properties, latent, metadata
        # ------------------------------------------------------------------
        with col2:
            st.subheader("Predicted Properties")
            if props:
                st.json(props)
            else:
                st.write("No properties returned.")

            # --- Latent vector ---
            st.subheader("Latent Vector (first 10 dimensions)")
            if latent:
                st.write(latent[:10])
                with st.expander("Show full latent vector (64 dimensions)"):
                    st.write(latent)
            else:
                st.write("Latent vector not available.")

            # --- NEW: Metadata & quality info ---
            with st.expander("Show model metadata & training info"):
                st.markdown("### Summary")
                st.write("**Model:**", meta.get("model", "unknown"))
                st.write("**Latent dimension:**", meta.get("latent_dim", "n/a"))
                st.write("**Has property head:**", meta.get("has_property_head", False))
                st.write("**Latency (ms):**", meta.get("latency_ms", "n/a"))
                st.write("**SMILES validator note:**", meta.get("validator_note", "n/a"))
                st.write("**Image generated:**", meta.get("image_generated", False))

                # Property-head metrics (if present)
                prop_head = meta.get("prop_head")
                if prop_head:
                    st.markdown("### Property head metrics")
                    best_mse = prop_head.get("best_val_mse_stdspace", None)
                    if best_mse is not None:
                        st.write(
                            "**Best validation MSE (standardized space):**",
                            best_mse,
                        )

                    metrics = prop_head.get("metrics", {})
                    if metrics:
                        st.markdown("Per-property metrics (MAE, R²):")
                        st.json(metrics)

                    props_order = prop_head.get("props_order")
                    if props_order:
                        st.markdown("**Properties modeled by the head:**")
                        st.write(", ".join(props_order))

                st.markdown("### Raw metadata JSON")
                st.json(meta)

    except requests.HTTPError as e:
        st.error(f"HTTP error from API: {e}")
    except requests.RequestException as e:
        st.error(f"Network error while calling the API: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

elif not run_clicked:
    st.info("Enter a SMILES string and click **Run model**.")
