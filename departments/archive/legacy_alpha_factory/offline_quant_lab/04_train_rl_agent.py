import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
from stable_baselines3 import PPO
import torch
import os
import time

print("Booting Dual-Agent Reinforcement Learning Environment...")

# 1. Hardware Check
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 PPO Engine assigned to: {device}")

# 2. Load the Compiled State Space
data_dir = "data"
input_file = os.path.join(data_dir, "master_state_space.parquet")

df = pl.read_parquet(input_file).drop_nulls()

# Dynamically detect features to size the neural network input layer
exclude_cols = ["Date", "Ticker", "Close", "Volume", "Target_Fwd_Ret_5D"]
feature_cols = [col for col in df.columns if col not in exclude_cols]
num_features = len(feature_cols)

print(f"Detected {num_features} factor dimensions for the State Space (S_t).")

# 3. Define the Custom Trading Environment (The "Video Game")
class HRLTradingEnv(gym.Env):
    def __init__(self, states, future_returns, archetype_betas):
        super(HRLTradingEnv, self).__init__()
        self.states = states
        self.future_returns = future_returns
        self.archetype_betas = archetype_betas
        
        self.current_step = 0
        self.max_steps = len(states) - 1
        
        # Action Space: 8 Discrete Archetypes
        self.action_space = spaces.Discrete(8)
        
        # Observation Space: The dynamically sized array of quantitative factors
        self.observation_space = spaces.Box(low=-20, high=20, shape=(num_features,), dtype=np.float32)

    def reset(self, seed=None):
        self.current_step = 0
        return self.states[self.current_step].astype(np.float32), {}

    def step(self, action):
        # The true market return for the next 5 days
        market_return = self.future_returns[self.current_step]
        
        # Agent's Return
        agent_beta = self.archetype_betas[action]
        R_agent = agent_beta * market_return
        
        # Baseline Return (Beta = 1.0)
        R_base = 1.0 * market_return
        
        # DP Optimal Return (Perfect Hindsight - Max possible return given the 8 choices)
        possible_returns = [beta * market_return for beta in self.archetype_betas.values()]
        R_opt = max(possible_returns)
        
        # THE REGRET-AWARE REWARD FUNCTION
        beta_1 = 0.5 # Regret Penalty Weight
        reward_beat_base = R_agent - R_base
        reward_regret = beta_1 * (R_agent - R_opt) # Always <= 0
        
        total_reward = reward_beat_base + reward_regret
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        next_state = self.states[self.current_step].astype(np.float32) if not terminated else np.zeros(num_features, dtype=np.float32)
        
        return next_state, total_reward, terminated, truncated, {}

# 4. The Core Training Function
def train_agent(ticker_symbol, agent_name, beta_map, steps=20000):
    print(f"\n--- Training {agent_name} Agent on {ticker_symbol} ---")
    
    # Isolate data for the specific ticker to teach the agent the market physics
    ticker_df = df.filter(pl.col("Ticker") == ticker_symbol).to_pandas()
    
    if len(ticker_df) < 1000:
        print(f"⚠️ Warning: Not enough data for {ticker_symbol}. Need at least 1000 rows.")
        return
        
    X_states = ticker_df[feature_cols].to_numpy()
    y_future_returns = ticker_df["Target_Fwd_Ret_5D"].to_numpy()
    
    # Init Environment
    env = HRLTradingEnv(X_states, y_future_returns, archetype_betas=beta_map)
    
    # Init PPO Agent
    agent = PPO("MlpPolicy", env, verbose=0, device=device, learning_rate=0.0003)
    
    start_time = time.time()
    agent.learn(total_timesteps=steps)
    print(f"✅ Training Complete in {time.time() - start_time:.2f} seconds.")
    
    # Save Model
    os.makedirs("models", exist_ok=True)
    save_path = f"models/horizon_policy_{agent_name.lower()}"
    agent.save(save_path)
    print(f"✅ Brain saved to {save_path}.zip")

# 5. Execute Dual Training Pipeline

# MACRO AGENT (Trained on SPY)
# 8 Archetypes: [-0.5, -0.2, 0.0, 0.3, 0.6, 0.8, 1.0, 1.5]
macro_betas = {0: -0.5, 1: -0.2, 2: 0.0, 3: 0.3, 4: 0.6, 5: 0.8, 6: 1.0, 7: 1.5}
train_agent(ticker_symbol="SPY", agent_name="Macro", beta_map=macro_betas)

# MEME AGENT (Trained on NVDA as a high-beta proxy)
# 8 Archetypes: [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
meme_betas = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0, 5: 1.5, 6: 2.0, 7: 3.0}
train_agent(ticker_symbol="NVDA", agent_name="Meme", beta_map=meme_betas)

print("\n🎉 ALL HRL AGENTS TRAINED. OFFLINE QUANT LAB COMPLETE.")