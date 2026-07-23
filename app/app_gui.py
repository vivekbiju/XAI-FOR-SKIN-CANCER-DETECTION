# import streamlit as st
# import requests
# from PIL import Image
# import io

# st.set_page_config(page_title="Skin Cancer AI Diagnostic", layout="wide")

# st.title("🩺 Clinical AI Diagnostic Assistant")
# st.write("Upload a dermoscopic image to get an AI-powered diagnosis and localization heatmap.")

# # 1. Sidebar for Server Status
# st.sidebar.header("System Status")
# try:
#     # Check if FastAPI is running
#     requests.get("http://127.0.0.1:8000/docs")
#     st.sidebar.success("API Server: Connected")
# except:
#     st.sidebar.error("API Server: Disconnected (Please run main.py)")

# # 2. File Uploader
# uploaded_file = st.file_uploader("Choose a lesion image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
#     if st.button("Run Diagnostic"):
#         with st.spinner('Analyzing...'):
#             # Convert uploaded file to format FastAPI expects
#             files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
#             # Send to your FastAPI server
#             response = requests.post("http://127.0.0.1:8080/diagnose", files=files)
            
#             if response.status_code == 200:
#                 result = response.json()
                
#                 with col2:
#                     st.subheader(f"Diagnosis: {result['diagnosis']}")
#                     st.write(f"**Confidence:** {result['confidence']}")
                    
#                     # Fetch the XAI Map from the static URL provided by the API
#                     full_url = f"http://127.0.0.1:8000{result['report_url']}"
#                     st.image(full_url, caption="AI Interpretability Report", use_column_width=True)
#             else:
#                 st.error("Diagnostic Failed. Please check the API server logs.")
                
                
                
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Skin Cancer AI Diagnostic", layout="wide")

st.title("🧬 Clinical AI Diagnostic Assistant")
st.write("Upload a dermoscopic image to get an AI-powered diagnosis and localization heatmap.")

# Standardize our Base URL to match our Docker port mapping
API_BASE_URL = "http://localhost:8080"

# 1. Sidebar for Server Status
st.sidebar.header("System Status")
try:
    # Use the /health endpoint we built into FastAPI instead of hitting /docs
    health_check = requests.get(f"{API_BASE_URL}/health", timeout=2)
    if health_check.status_code == 200:
        st.sidebar.success("API Server: Connected (Docker)")
    else:
        st.sidebar.error("API Server: Unhealthy Status")
except Exception:
    st.sidebar.error("API Server: Disconnected (Ensure Docker container is running on port 8080)")

# 2. File Uploader
uploaded_file = st.file_uploader("Choose a lesion image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Original Image", use_container_width=True)
    
    if st.button("Run Diagnostic"):
        with st.spinner('Running Hybrid Inference & XAI Suite...'):
            # Prepare the multipart payload for FastAPI
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                # Send request to our containerized diagnose route
                response = requests.post(f"{API_BASE_URL}/api/v1/diagnose", files=files)
                
                if response.status_code == 200:
                    # Extract medical metadata from custom response headers
                    diagnosis = response.headers.get("X-Diagnostic-Prediction", "Unknown")
                    confidence = response.headers.get("X-Confidence-Score", "N/A")
                    traceability = response.headers.get("X-Clinical-Traceability", "N/A")
                    
                    # Convert the raw streaming image bytes into a PIL image for Streamlit
                    xai_image = Image.open(io.BytesIO(response.content))
                    
                    with col2:
                        # Display the Grad-CAM visualization output
                        st.image(xai_image, caption="Grad-CAM Localization Map & Visual Justification", use_container_width=True)
                        st.subheader(f"Diagnosis: {diagnosis}")
                        st.metric(label="Confidence Score", value=f"{float(confidence)*100:.2f}%" if confidence != "N/A" else "N/A")
                        
                        
                        
                        # Add Senior-level clinical disclosure 
                        st.caption(f"**Clinical Traceability Note:** {traceability}")
                else:
                    st.error(f"Diagnostic Failed. Server returned status code: {response.status_code}")
                    st.json(response.json()) # Display error details from FastAPI if available
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API server. Ensure your Docker container is up on port 8080.")