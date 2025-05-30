# /Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/evaluate_agent.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback # Added for detailed error logging
from stable_baselines3 import SAC

# Import custom modules
from data_manager import load_and_preprocess_data, EXPECTED_FEATURE_COLUMNS
from trading_env import TradingEnv
from custom_cnn import CustomCNN # Needed for loading model with custom policy

# --- Configuration ---
DATA_CSV_PATH = '../../BTC_hourly_with_features.csv' # Corrected path to the CSV file
DATE_COLUMN_NAME = 'timestamp'
MODEL_PATH = "./models/best_sac_trader/best_model.zip" # Path to the trained model
# Or use: MODEL_PATH = "./models/sac_trader_final.zip"

# Environment Hyperparameters (should match training, but some can be different for eval)
WINDOW_SIZE = 100
INITIAL_BALANCE = 10000
# EPISODE_LENGTH for evaluation will be the full length of the test set.
FEE_RATE = 0.00075

# --- Helper function for HODL strategy ---
def calculate_hodl_performance(df, initial_balance, fee_rate):
    if df.empty:
        return pd.Series(dtype=float), 0 # Restored line
    
    hodl_balance = initial_balance
    entry_price = df['open'].iloc[0]
    
    # Buy at the first open price
    btc_bought = (initial_balance / entry_price) * (1 - fee_rate) # Account for entry fee
    hodl_balance = 0 # All in BTC
    
    hodl_equity_curve = []
    for current_price in df['close']: # Evaluate based on close prices for HODL curve
        current_value = btc_bought * current_price
        hodl_equity_curve.append(hodl_balance + current_value)
        
    # Sell at the last close price (optional, to realize final PnL after fees)
    # final_value = btc_bought * df['close'].iloc[-1]
    # final_balance_after_exit_fee = final_value * (1 - fee_rate)
    # hodl_equity_curve[-1] = final_balance_after_exit_fee # Update last point
    
    return pd.Series(hodl_equity_curve, index=df.index), btc_bought # Restored line

# --- Helper function for calculating Sharpe Ratio (simplified) ---
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    # Assuming returns are daily/hourly, adjust risk_free_rate and annualization factor accordingly
    # For simplicity, using per-period returns and a simple Sharpe
    if returns.std() == 0: # Avoid division by zero
        return 0.0 # Restored line
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() # * np.sqrt(ANNUALIZATION_FACTOR) for annualized

def main():
    # --- 1. Load and Preprocess Data ---
    print("Loading and preprocessing data for evaluation...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_full_path = os.path.join(script_dir, DATA_CSV_PATH)
    csv_full_path = os.path.normpath(csv_full_path)

    _, test_df_full, feature_cols = load_and_preprocess_data(
        csv_full_path,
        date_column_name=DATE_COLUMN_NAME,
        expected_features=EXPECTED_FEATURE_COLUMNS
    )
    print(f"Test data shape: {test_df_full.shape}")

    if test_df_full.empty:
        print("Error: Test data is empty. Exiting.") # Restored line
        return # Restored line
    
    # The TradingEnv expects df with reset index for iloc
    test_df_env = test_df_full.copy()

    # --- 2. Create Evaluation Environment ---
    # The episode length for evaluation should be the entire length of the test data
    # minus the window size, as the agent needs `window_size` initial data points.
    # The environment handles starting at window_size internally when is_training=False.
    # The actual number of steps the agent takes will be len(test_df_env) - window_size.
    eval_episode_length = len(test_df_env) - WINDOW_SIZE 

    print("Creating evaluation environment...")
    eval_env = TradingEnv(
        df=test_df_env, # Pass the full test_df
        feature_columns=feature_cols,
        window_size=WINDOW_SIZE,
        initial_balance=INITIAL_BALANCE,
        episode_length=eval_episode_length, # Agent will run for this many steps
        fee_rate=FEE_RATE,
        is_training=False, # Crucial: ensures sequential processing and starts at window_size
        reward_scale_factor=100.0,
        fee_penalty_factor=10.0,
        action_change_penalty_factor=0.5,
        max_leverage=1.0,
        carry_cost_rate=0.00005,
        position_penalty_factor=0.01,
        benchmark_weight=0.05,
        use_sortino_ratio=True,
        sortino_target_return=0.00001
    )
    print(f"Evaluation environment created. Agent will take {eval_episode_length} steps.")

    # --- 3. Load the Trained Model ---
    print(f"Loading trained model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train a model first or check the path.") # Restored line
        return # Restored line

    try:
        model = SAC.load(MODEL_PATH, env=eval_env)
        # Optionally, check policy_kwargs compatibility:
        # assert model.policy_kwargs['features_extractor_class'] == CustomCNN
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure CustomCNN is imported and policy_kwargs (if used for features_dim) are correct.")
        traceback.print_exc()
        return

    # --- 4. Run Evaluation Loop ---
    print("Running evaluation...")
    obs, info = eval_env.reset()
    
    agent_equity_curve = [INITIAL_BALANCE]
    agent_actions = []
    agent_rewards = []
    dates = [test_df_full.index[WINDOW_SIZE-1]] # Date for initial balance
                                               # Trades start from WINDOW_SIZE-th index of original df

    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True) # Use deterministic actions for evaluation
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        agent_equity_curve.append(info['total_equity'])
        agent_actions.append(action[0]) # Store the action value
        agent_rewards.append(reward)
        
        # The date corresponds to the 'open' price time of the bar the agent just acted on
        # which is `eval_env.current_decision_step_idx - 1` because step increments it
        current_df_time_idx = eval_env.current_decision_step_idx -1 
        if current_df_time_idx < len(test_df_full):
             dates.append(test_df_full.index[current_df_time_idx])
        else: # Should not happen if episode_length is set correctly
            print("Warning: current_df_time_idx out of bounds for dates.")
            break

        step_count += 1
        if step_count % 1000 == 0:
            print(f"Evaluated {step_count}/{eval_episode_length} steps...")

        if terminated:
            print("Agent terminated the episode (e.g., due to ruin).")
        if truncated:
            print("Agent truncated (reached end of evaluation period or data).")
            
    print(f"Evaluation complete. Total steps taken: {step_count}")

    # Ensure equity curve and dates have same length
    if len(agent_equity_curve) > len(dates):
        agent_equity_curve = agent_equity_curve[:len(dates)]
    elif len(dates) > len(agent_equity_curve):
        dates = dates[:len(agent_equity_curve)]

    agent_equity_ts = pd.Series(agent_equity_curve, index=pd.to_datetime(dates))


    # --- 5. Calculate HODL Performance on the same period ---
    # HODL starts from the first point the agent can trade
    hodl_eval_period_df = test_df_full.iloc[WINDOW_SIZE:]
    hodl_equity_ts, _ = calculate_hodl_performance(hodl_eval_period_df, INITIAL_BALANCE, FEE_RATE)
    
    # Align HODL series to start at the same time as agent's first actual equity point after initial balance
    if not hodl_equity_ts.empty and not agent_equity_ts.empty:
        hodl_equity_ts = hodl_equity_ts[hodl_equity_ts.index >= agent_equity_ts.index[0]]
        # Prepend initial balance to HODL if agent_equity_ts starts with it
        if agent_equity_ts.iloc[0] == INITIAL_BALANCE and hodl_equity_ts.iloc[0] != INITIAL_BALANCE:
             hodl_initial_balance_point = pd.Series([INITIAL_BALANCE], index=[agent_equity_ts.index[0]])
             hodl_equity_ts = pd.concat([hodl_initial_balance_point, hodl_equity_ts])


    # --- 6. Performance Metrics & Visualization ---
    print("\n--- Performance Summary ---")
    final_agent_equity = agent_equity_ts.iloc[-1]
    final_hodl_equity = hodl_equity_ts.iloc[-1] if not hodl_equity_ts.empty else INITIAL_BALANCE
    
    print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Final Agent Equity: ${final_agent_equity:,.2f}")
    print(f"Final HODL Equity: ${final_hodl_equity:,.2f}")

    agent_return = (final_agent_equity / INITIAL_BALANCE - 1) * 100
    hodl_return = (final_hodl_equity / INITIAL_BALANCE - 1) * 100
    print(f"Agent Total Return: {agent_return:.2f}%")
    print(f"HODL Total Return: {hodl_return:.2f}%")

    # Calculate returns for Sharpe Ratio (using pct_change)
    agent_returns_pct = agent_equity_ts.pct_change().dropna()
    hodl_returns_pct = hodl_equity_ts.pct_change().dropna()

    agent_sharpe = calculate_sharpe_ratio(agent_returns_pct)
    hodl_sharpe = calculate_sharpe_ratio(hodl_returns_pct)
    print(f"Agent Sharpe Ratio (simplified, per-period): {agent_sharpe:.4f}")
    print(f"HODL Sharpe Ratio (simplified, per-period): {hodl_sharpe:.4f}")
    
    # Max Drawdown
    def max_drawdown(equity_curve):
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        return float(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    agent_max_dd = max_drawdown(agent_equity_ts.values)
    hodl_max_dd = max_drawdown(hodl_equity_ts.values) if not hodl_equity_ts.empty else 0.0
    print(f"Agent Max Drawdown: {agent_max_dd:.2%}")
    print(f"HODL Max Drawdown: {hodl_max_dd:.2%}")
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(agent_equity_ts.index, agent_equity_ts.values, label=f"Agent (Final: ${final_agent_equity:,.0f})", color="blue")
    if not hodl_equity_ts.empty:
        plt.plot(hodl_equity_ts.index, hodl_equity_ts.values, label=f"HODL (Final: ${final_hodl_equity:,.0f})", color="orange")
    
    plt.title("Agent vs. HODL Performance (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Equity ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_save_path = os.path.join(script_dir, "results", "agent_vs_hodl_performance.png")
    os.makedirs(os.path.join(script_dir, "results"), exist_ok=True)
    plt.savefig(plot_save_path)
    print(f"\nPerformance plot saved to: {plot_save_path}")
    plt.show()

    # Plot actions over time
    if agent_actions:
        plt.figure(figsize=(14, 5))
        # Use the dates corresponding to the actions (all dates except the first initial balance date)
        action_dates = agent_equity_ts.index[1:] if len(agent_equity_ts.index) > len(agent_actions) else agent_equity_ts.index
        if len(action_dates) == len(agent_actions):
            plt.plot(action_dates, agent_actions, label="Agent Action (Target Allocation %)", color="green", alpha=0.7)
            plt.title("Agent Actions Over Time (Test Set)")
            plt.xlabel("Date")
            plt.ylabel("Action Value (-1 to 1)")
            plt.ylim(-1.1, 1.1)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            action_plot_save_path = os.path.join(script_dir, "results", "agent_actions.png")
            plt.savefig(action_plot_save_path)
            print(f"Actions plot saved to: {action_plot_save_path}")
            plt.show()
        else:
            print(f"Could not plot actions: length mismatch between dates ({len(action_dates)}) and actions ({len(agent_actions)}).")


if __name__ == '__main__':
    main()