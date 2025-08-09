import streamlit as st
import torch
import torch.nn as nn
import joblib

class AdvancedRankNet(nn.Module):
    def __init__(self, input_size=5):
        super(AdvancedRankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model_and_scaler():
    model = AdvancedRankNet()
    model_path = r"ranknet_model.pth"
    scaler_path = r"scaler.sav"
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("Product Ranking Optimization")

def input_features(product_name):
    st.subheader(product_name)
    price = st.number_input(f"Price (₹100-₹10000) for {product_name}", 100, 10000, 500)
    rating = st.slider(f"Rating (1.0-5.0) for {product_name}", 1.0, 5.0, 4.0, 0.1)
    reviews = st.number_input(f"Number of Reviews (0-5000) for {product_name}", 0, 5000, 100)
    discount = st.slider(f"Discount % (0-70) for {product_name}", 0, 70, 10)
    seller_rep = st.slider(f"Seller Reputation (1.0-5.0) for {product_name}", 1.0, 5.0, 4.0, 0.1)
    return [price, rating, reviews, discount, seller_rep]

with st.form("product_form"):
    features_A = input_features("Product A")
    features_B = input_features("Product B")
    submitted = st.form_submit_button("Predict Ranking")

if submitted:
    A_scaled = scaler.transform([features_A])
    B_scaled = scaler.transform([features_B])

    A_tensor = torch.tensor(A_scaled, dtype=torch.float32)
    B_tensor = torch.tensor(B_scaled, dtype=torch.float32)

    with torch.no_grad():
        score_A = model(A_tensor).item()
        score_B = model(B_tensor).item()

    st.write(f"**Product A score:** {score_A:.4f}")
    st.write(f"**Product B score:** {score_B:.4f}")

    if score_A > score_B:
        st.success("✅ Product A should rank higher.")
    elif score_B > score_A:
        st.success("✅ Product B should rank higher.")
    else:
        st.info("Both products have the same rank score.")
