import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# Fix for Windows OMP Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add src to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.model import NIDSModel
from src.attack import fgsm_attack

# Page Config
st.set_page_config(page_title="Adversarial NIDS Defense", layout="wide")

# --- 1. LOAD ASSETS (Cached for speed) ---
@st.cache_resource
def load_data():
    # Load only the Test set for the dashboard
    X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'))
    return X_test, y_test

@st.cache_resource
def load_models(input_dim):
    device = torch.device('cpu') # Force CPU for the web app
    
    # Load Standard Model
    model_std = NIDSModel(input_dim)
    model_std.load_state_dict(torch.load(
        os.path.join(config.PROJECT_ROOT, 'results', 'model.pth'), 
        map_location=device
    ))
    model_std.eval()
    
    # Load Defended Model
    model_adv = NIDSModel(input_dim)
    model_adv.load_state_dict(torch.load(
        os.path.join(config.PROJECT_ROOT, 'results', 'model_adversarial.pth'), 
        map_location=device
    ))
    model_adv.eval()
    
    return model_std, model_adv

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ Control Panel")

# Load Data
X_test, y_test = load_data()
input_dim = X_test.shape[1]

# Selector: Pick a malicious sample
# We only want to test on ACTUAL attacks to see if we can hide them
malicious_indices = np.where(y_test == 1)[0]
selected_idx = st.sidebar.selectbox("Select a Malicious Network Sample:", malicious_indices[:50]) # Limit to first 50 for UI

# Slider: Attack Strength
epsilon = st.sidebar.slider("Attack Strength (Epsilon)", 0.0, 0.3, 0.1, 0.01)

# Load Models
model_std, model_adv = load_models(input_dim)

# --- 3. MAIN INTERFACE ---
st.title("🛡️ AI-Powered NIDS Defense System")
st.markdown("### Interactive Adversarial Attack Simulator")
st.markdown("---")

# Get the specific sample
sample_np = X_test[selected_idx]
sample_tensor = torch.tensor(sample_np, dtype=torch.float32).unsqueeze(0)
label_tensor = torch.tensor([1.0]).unsqueeze(0) # We know it's malicious (1)

# Generate Attack
perturbed_tensor = fgsm_attack(model_std, sample_tensor, label_tensor, epsilon)
perturbation = perturbed_tensor.detach().numpy() - sample_np

# Get Predictions
with torch.no_grad():
    # Standard Model
    pred_std_clean = model_std(sample_tensor).item()
    pred_std_adv = model_std(perturbed_tensor).item()
    
    # Defended Model
    pred_adv_clean = model_adv(sample_tensor).item()
    pred_adv_adv = model_adv(perturbed_tensor).item()

# --- 4. DISPLAY RESULTS ---

col1, col2 = st.columns(2)

# --- Standard Model Column ---
with col1:
    st.header("Standard NIDS (Vulnerable)")
    
    # Clean Data Result
    st.subheader("1. On Clean Data")
    if pred_std_clean > 0.5:
        st.success(f"✅ DETECTED (Confidence: {pred_std_clean:.2%})")
    else:
        st.error(f"❌ MISSED (Confidence: {pred_std_clean:.2%})")
        
    # Attacked Data Result
    st.subheader(f"2. Under Attack (Epsilon={epsilon})")
    if pred_std_adv > 0.5:
        st.success(f"✅ DETECTED (Confidence: {pred_std_adv:.2%})")
    else:
        st.error(f"⚠️ BYPASSED! (Confidence: {pred_std_adv:.2%})")
        st.caption("The attacker successfully hid the malicious packet.")

# --- Defended Model Column ---
with col2:
    st.header("🛡️ Robust NIDS (Defended)")
    
    # Clean Data Result
    st.subheader("1. On Clean Data")
    if pred_adv_clean > 0.5:
        st.success(f"✅ DETECTED (Confidence: {pred_adv_clean:.2%})")
    else:
        st.error(f"❌ MISSED (Confidence: {pred_adv_clean:.2%})")
        
    # Attacked Data Result
    st.subheader(f"2. Under Attack (Epsilon={epsilon})")
    if pred_adv_adv > 0.5:
        st.success(f"✅ BLOCKED (Confidence: {pred_adv_adv:.2%})")
    else:
        st.error(f"❌ FAILED (Confidence: {pred_adv_adv:.2%})")
        
st.markdown("---")

# --- 5. VISUALIZATION ---
st.subheader("🔍 Forensics: What changed?")
st.write("The chart below shows the **noise** added by the FGSM attack to the network features.")

# Plot only non-zero perturbations to keep chart clean
diff = perturbation.flatten()
nonzero_indices = np.where(np.abs(diff) > 0.001)[0]

if len(nonzero_indices) > 0:
    chart_data = pd.DataFrame({
        "Feature Index": nonzero_indices,
        "Perturbation Value": diff[nonzero_indices]
    })
    st.bar_chart(chart_data, x="Feature Index", y="Perturbation Value")
else:
    st.info("No significant perturbation detected (Epsilon is too low).")
