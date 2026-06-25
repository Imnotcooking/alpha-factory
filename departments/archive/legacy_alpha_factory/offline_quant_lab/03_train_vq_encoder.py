import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import time

print("Booting Dual-Engine PyTorch Quant Lab...")

# 1. Hardware Check
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Apple Silicon (MPS) detected. GPU Acceleration ON.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 2. Load the Compiled State Space
data_dir = "data"
input_file = os.path.join(data_dir, "master_state_space.parquet")

print("Loading compiled 20-year State Space into RAM...")
df = pl.read_parquet(input_file)

# Dynamically detect all generated factors (Ignore metadata and target variables)
exclude_cols = ["Date", "Ticker", "Close", "Volume", "Target_Fwd_Ret_5D"]
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"Detected {len(feature_cols)} dynamic factors from the factor_library.")

# 3. Define the Universes
# We split the data into Standard Institutional vs High-Beta/Meme stocks
macro_tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "JPM", "XOM", "UNH", "JNJ", "V", "AAPL", "MSFT"]
meme_tickers = ["NVDA", "AMZN", "GOOGL", "TSLA", "COIN", "CIFR"] # Add TSLA, COIN, GME here if you download them later!

df_macro = df.filter(pl.col("Ticker").is_in(macro_tickers))
df_meme = df.filter(pl.col("Ticker").is_in(meme_tickers))

# 4. Define the Neural Network Architecture
class VQ_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super(VQ_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return latent, self.decoder(latent)

# 5. The Core Training Function (Reusable)
def train_and_extract_codebook(data, universe_name, k_archetypes=8):
    if data.is_empty():
        print(f"⚠️ Warning: No data found for {universe_name}. Skipping.")
        return
        
    print(f"\n--- Training Engine: {universe_name} Universe ---")
    print(f"Data rows: {data.height:,} | Target Archetypes (k): {k_archetypes}")
    
    # Scale Data
    X_raw = data.select(feature_cols).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    # Init Model
    model = VQ_Encoder(input_dim=len(feature_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    # Train
    epochs = 40
    batch_size = 2048
    model.train()
    
    start_time = time.time()
    for epoch in range(epochs):
        indices = torch.randperm(X_tensor.size(0))
        for i in range(0, X_tensor.size(0), batch_size):
            batch_X = X_tensor[indices[i:i+batch_size]]
            optimizer.zero_grad()
            _, reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_X)
            loss.backward()
            optimizer.step()
            
    print(f"✅ Autoencoder trained in {time.time() - start_time:.2f}s.")
    
    # Extract Latent Space & Cluster
    print(f"Extracting Latent Space and generating {k_archetypes} Archetypes...")
    model.eval()
    with torch.no_grad():
        latent_space, _ = model(X_tensor)
        latent_space_cpu = latent_space.cpu().numpy()
        
    kmeans = KMeans(n_clusters=k_archetypes, random_state=42, n_init=10)
    kmeans.fit(latent_space_cpu)
    
    # Save the specific Brain
    os.makedirs("models", exist_ok=True)
    payload = {
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'kmeans_centers': kmeans.cluster_centers_,
        'features': feature_cols # Save feature names to ensure alignment during live inference
    }
    
    save_path = f"models/vq_codebook_{universe_name.lower()}.pt"
    torch.save(payload, save_path)
    print(f"✅ {universe_name} Codebook saved to {save_path}")

# 6. Execute Dual Training Pipeline
# We use k=8 for both to provide deep strategic granularity
train_and_extract_codebook(df_macro, "Institutional_Macro", k_archetypes=8)
train_and_extract_codebook(df_meme, "HighBeta_Meme", k_archetypes=8)

print("\n🎉 ALL VQ-ENCODERS TRAINED SUCCESSFULLY.")