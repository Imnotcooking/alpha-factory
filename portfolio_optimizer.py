# portfolio_optimizer.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class ConvexOptimizer:
    def __init__(self, target_beta: float, total_capital: float):
        """
        Translates Agent Conviction into real Dollars while minimizing Volatility.
        """
        self.target_beta = target_beta
        self.total_capital = total_capital

    def optimize(self, agent_signals_df: pd.DataFrame, cov_matrix: np.ndarray) -> pd.DataFrame:
        """
        agent_signals_df requires: ['symbol', 'close', 'Target_Weight']
        cov_matrix: A numpy covariance matrix of the target stocks' daily returns
        """
        print(f"⚖️ SciPy Optimizer Booting... Target Beta: {self.target_beta}")
        
        n_assets = len(agent_signals_df)
        initial_weights = agent_signals_df['Target_Weight'].values
        
        # 1. Objective Function: Minimize Portfolio Variance (Risk)
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
            
        # 2. Constraints
        constraints = [
            # Constraint 1: The sum of weights must equal the RL Agent's Target Beta
            # If Beta is 0.5, we only deploy 50% of our capital.
            {'type': 'eq', 'fun': lambda w: np.sum(w) - self.target_beta}
        ]
        
        # 3. Bounds (No shorting allowed in this specific run, weights between 0 and Target Beta)
        bounds = tuple((0.0, self.target_beta) for _ in range(n_assets))
        
        # 4. Run the SciPy Convex Optimization
        result = minimize(
            portfolio_variance, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        if not result.success:
            print("⚠️ Optimizer failed to converge. Defaulting to equal weight scaled by Beta.")
            optimal_weights = initial_weights * self.target_beta
        else:
            optimal_weights = result.x
            
        # 5. Translate the optimal math into real dollars and shares
        df = agent_signals_df.copy()
        df['Final_Weight'] = optimal_weights
        df['Capital_Allocated'] = self.total_capital * df['Final_Weight']
        df['Target_Shares'] = np.floor(df['Capital_Allocated'] / df['close'])
        
        print(f"✅ Optimization Complete. Total Capital Deployed: ${(df['Capital_Allocated'].sum()):,.2f}")
        
        return df[['symbol', 'close', 'Final_Weight', 'Capital_Allocated', 'Target_Shares']]

# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    # Mock data from a Strategy Agent
    mock_agent_data = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'NVDA'],
        'close': [150.0, 300.0, 800.0],
        'Target_Weight': [0.33, 0.33, 0.33] # Agent wants equal weight
    })
    
    # Mock Covariance Matrix (Normally calculated from your Parquet data)
    mock_cov = np.array([
        [0.0004, 0.0002, 0.0001],
        [0.0002, 0.0005, 0.0003],
        [0.0001, 0.0003, 0.0009]
    ])
    
    # Let's say the RL Agent detected a crash and wants a Beta of 0.5 (Half Cash)
    total_cash = 100000.0
    optimizer = ConvexOptimizer(target_beta=0.5, total_capital=total_cash)
    
    final_orders = optimizer.optimize(mock_agent_data, mock_cov)
    print("\n--- FINAL INTERACTIVE BROKERS EXECUTION ORDERS ---")
    print(final_orders)