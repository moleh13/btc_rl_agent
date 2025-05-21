# /Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.stats import zscore

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, df, feature_columns, window_size=100, initial_balance=10000,
                 episode_length=720, fee_rate=0.00075, is_training=True):
        super().__init__()

        self.df = df.reset_index() # Ensure simple integer indexing for iloc
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.episode_length = episode_length # 30 days * 24 hours
        self.fee_rate = fee_rate
        self.is_training = is_training # To control random episode starts

        # Define columns for Z-score normalization (price/volume related)
        self.z_score_cols = [
            'open', 'high', 'low', 'close', 'volume', 
            'SMA50', 'EMA12', 'EMA26', 'EMA100', 'EMA200',
            'BB_Middle_Band', 'BB_Upper_Band', 'BB_Lower_Band',
            'OBV', 'VWAP'
        ]
        # Verify all z_score_cols are in feature_columns
        for col in self.z_score_cols:
            if col not in self.feature_columns:
                raise ValueError(f"Column '{col}' specified for Z-score normalization is not in feature_columns.")
        
        self.num_market_features = len(self.feature_columns)

        # Action space: single continuous value in [-1, 1] (target fraction of equity)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: Dictionary with market data and current position ratio
        self.observation_space = spaces.Dict({
            "market_data": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.window_size, self.num_market_features),
                dtype=np.float32
            ),
            "position_ratio": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        
        # Episode state variables
        self.current_decision_step_idx = 0 # Index in self.df for the current bar decision
        self.episode_step_count = 0      # Number of steps taken in current episode
        self.balance = 0.0
        self.btc_held = 0.0
        self.current_position_value_btc = 0.0 # Value of BTC held, in USD
        self.total_equity = 0.0
        self.current_position_ratio = 0.0 # (btc_held * current_price) / total_equity
        
        self.max_possible_steps = len(self.df) - self.window_size - self.episode_length -1 # for training start idx

    def _get_observation(self):
        # Observation window: data from (current_decision_step_idx - window_size) to (current_decision_step_idx - 1)
        # So, if current_decision_step_idx is 100, window is df.iloc[0:100]
        start_idx = self.current_decision_step_idx - self.window_size
        end_idx = self.current_decision_step_idx # Exclusive end, so it takes up to current_decision_step_idx-1

        if start_idx < 0: # Should not happen if reset is handled correctly
            raise ValueError("start_idx for observation window is negative.")

        market_window_df = self.df.iloc[start_idx:end_idx][self.feature_columns].copy()

        # Apply Z-score normalization to specified columns *within the window*
        for col in self.z_score_cols:
            values = market_window_df[col].values
            # Handle potential division by zero if std is 0 (constant value in window)
            if values.std() == 0:
                market_window_df[col] = 0.0 # Or values - values.mean(), results in 0
            else:
                market_window_df[col] = zscore(values)
        
        # Handle any NaNs produced by zscore (e.g., if all values were identical and std was 0, then zscore might produce NaN)
        # This should be rare given the prior NaN handling but good for robustness.
        market_window_df.fillna(0.0, inplace=True)

        obs_market_data = market_window_df.values.astype(np.float32)
        
        # Update current_position_ratio based on price at end of observation window
        # (i.e., 'close' of the bar *before* the current decision bar)
        last_obs_close_price = self.df.iloc[end_idx - 1]['close']
        self.current_position_value_btc = self.btc_held * last_obs_close_price
        current_total_equity_at_obs_end = self.balance + self.current_position_value_btc
        
        if current_total_equity_at_obs_end <= 0: # Avoid division by zero if equity is wiped out
            self.current_position_ratio = 0.0
        else:
            self.current_position_ratio = self.current_position_value_btc / current_total_equity_at_obs_end
        
        # Clip position ratio to be safe, though it should naturally be within [-1, 1] if logic is correct
        self.current_position_ratio = np.clip(self.current_position_ratio, -1.0, 1.0)

        return {
            "market_data": obs_market_data,
            "position_ratio": np.array([self.current_position_ratio], dtype=np.float32)
        }

    def _take_action(self, action_value):
        # action_value is the target fraction of total equity for BTC position [-1.0, 1.0]
        trade_price = self.df.iloc[self.current_decision_step_idx]['open'] # Trade at open of current bar

        if trade_price <= 0: # Safety for bad data
            return 0.0 # No trade, no fees

        # Recalculate total equity just before trade decision based on trade_price
        # This ensures the target position is based on the most up-to-date equity valuation
        current_position_value_at_trade_price = self.btc_held * trade_price
        equity_at_trade_moment = self.balance + current_position_value_at_trade_price

        if equity_at_trade_moment <= 0: # Cannot trade if no equity
            return 0.0

        target_btc_value = action_value * equity_at_trade_moment
        target_btc_quantity = target_btc_value / trade_price

        trade_quantity_btc = target_btc_quantity - self.btc_held
        
        executed_trade_value_abs = 0.0
        fees_paid = 0.0

        if abs(trade_quantity_btc * trade_price) < 1e-3 : # Negligible trade, effectively no trade
            trade_quantity_btc = 0.0

        if trade_quantity_btc > 0: # Try to buy BTC
            value_to_buy_btc = trade_quantity_btc * trade_price
            cost_including_fees = value_to_buy_btc * (1 + self.fee_rate)

            if cost_including_fees > self.balance: # Not enough cash
                # Can only buy what balance allows after fees
                actual_value_can_buy_btc = self.balance / (1 + self.fee_rate)
                trade_quantity_btc = actual_value_can_buy_btc / trade_price if trade_price > 0 else 0
            
            if trade_quantity_btc < 0: trade_quantity_btc = 0 # Should not happen due to above logic

            executed_trade_value_abs = trade_quantity_btc * trade_price
            fees_paid = executed_trade_value_abs * self.fee_rate
            
            self.balance -= (executed_trade_value_abs + fees_paid)
            self.btc_held += trade_quantity_btc

        elif trade_quantity_btc < 0: # Try to sell BTC (or increase short)
            value_to_sell_btc = abs(trade_quantity_btc * trade_price)
            
            # No cash constraint on selling what you own, or shorting up to 1x equity.
            # The target_btc_quantity logic already limits short exposure to equity_at_trade_moment.
            
            executed_trade_value_abs = value_to_sell_btc
            fees_paid = executed_trade_value_abs * self.fee_rate

            self.balance += (executed_trade_value_abs - fees_paid)
            self.btc_held += trade_quantity_btc # trade_quantity_btc is negative

        self.current_position_value_btc = self.btc_held * trade_price # Update after trade
        self.total_equity = self.balance + self.current_position_value_btc
        
        return fees_paid # Return fees for info if needed

    def _calculate_reward(self, previous_total_equity):
        # PnL and HODL returns are based on open-to-open prices of consecutive bars
        # Price for current step's PnL calculation (P_t+1 for agent, P_t+1 for HODL)
        # current_decision_step_idx is 't'. We need open of 't+1'.
        if (self.current_decision_step_idx + 1) >= len(self.df):
            # This can happen if episode ends exactly at the end of df
            # or if df is too short for episode_length + 1 lookahead
            # For simplicity, assume no change for this rare edge case affecting last step's reward
            next_bar_open_price = self.df.iloc[self.current_decision_step_idx]['open']
        else:
            next_bar_open_price = self.df.iloc[self.current_decision_step_idx + 1]['open']

        # Update total equity based on the next bar's open price
        self.current_position_value_btc = self.btc_held * next_bar_open_price
        self.total_equity = self.balance + self.current_position_value_btc

        # Agent's log return for this step
        if previous_total_equity <= 0 or self.total_equity <= 0: # Avoid log(0) or division by zero
            agent_log_return = 0.0
        else:
            agent_log_return = np.log(self.total_equity / previous_total_equity)

        # HODL log return for this step
        current_bar_open_price = self.df.iloc[self.current_decision_step_idx]['open']
        if current_bar_open_price <= 0 or next_bar_open_price <= 0:
            hodl_log_return = 0.0
        else:
            hodl_log_return = np.log(next_bar_open_price / current_bar_open_price)
        
        reward = 100 * (agent_log_return - hodl_log_return)
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility with SB3

        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.current_position_value_btc = 0.0
        self.total_equity = self.initial_balance
        self.current_position_ratio = 0.0
        self.episode_step_count = 0

        if self.is_training:
            # Randomly select a start index for a 30-day (+ window_size for initial obs) segment
            # Max start index: len(df) - window_size - episode_length - 1 (for PnL calc of last step)
            if self.max_possible_steps <= 0: # df too short
                 raise ValueError("DataFrame is too short for the window size and episode length.")
            start_df_idx = self.np_random.integers(0, self.max_possible_steps + 1)
            self.current_decision_step_idx = start_df_idx + self.window_size
        else: # Testing mode: start from the beginning
            self.current_decision_step_idx = self.window_size 
            # Ensure test_df is long enough: len(test_df) >= window_size + episode_length_for_test + 1
            # For testing, episode_length might be the full test set. This will be handled by the loop.

        observation = self._get_observation()
        info = self._get_info()
        
        # Prime the equity for the first reward calculation
        self.current_position_value_btc = self.btc_held * self.df.iloc[self.current_decision_step_idx]['open']
        self.total_equity = self.balance + self.current_position_value_btc


        return observation, info

    def step(self, action):
        # Store equity before action for reward calculation
        previous_total_equity = self.total_equity 
        
        # Execute trade based on action
        action_value = action[0] # Action is a single float value
        self._take_action(action_value)

        # Calculate reward
        reward = self._calculate_reward(previous_total_equity) # Renamed for clarity

        self.episode_step_count += 1
        self.current_decision_step_idx += 1

        # Termination conditions
        terminated = False
        if self.total_equity <= (self.initial_balance / 2):
            terminated = True
            reward -= 100  # Harsh penalty for ruin

        truncated = False
        if self.episode_step_count >= self.episode_length:
            truncated = True
        
        # Check if we've run out of data (important for testing or short DFs)
        # Need self.current_decision_step_idx + 1 for next HODL price.
        if not terminated and not truncated and (self.current_decision_step_idx + 1 >= len(self.df)):
            truncated = True # Ran out of data
            # This situation might mean the last step's reward couldn't be fully calculated if next_bar_open_price was estimated.
            # Or, if we are at the very last bar, then there is no next bar.

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human()

        return observation, reward, terminated, truncated, info

    def _get_info(self):
        return {
            "total_equity": self.total_equity,
            "balance": self.balance,
            "btc_held": self.btc_held,
            "current_position_value_btc": self.current_position_value_btc,
            "current_position_ratio": self.current_position_ratio,
            "episode_step": self.episode_step_count,
            "current_df_idx": self.current_decision_step_idx
        }

    def _render_human(self):
        # Basic rendering for now
        print(f"Step: {self.episode_step_count}, Equity: {self.total_equity:.2f}, "
              f"Position Ratio: {self.current_position_ratio:.2f}, BTC Held: {self.btc_held:.4f}")

    def close(self):
        # Any cleanup needed
        pass

if __name__ == '__main__':
    # --- Example Usage ---
    # 1. Load data using your data_manager
    import os
    from src.data_manager import load_and_preprocess_data, EXPECTED_FEATURE_COLUMNS

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_script_dir, '..', '..', 'BTC_hourly_with_features.csv')
    csv_path = os.path.normpath(csv_path)
    date_col_in_csv = 'timestamp' 

    try:
        train_df, test_df, feature_cols = load_and_preprocess_data(
            csv_path, 
            date_column_name=date_col_in_csv,
            expected_features=EXPECTED_FEATURE_COLUMNS
        )

        # 2. Create the environment (using a small part of train_df for quick test)
        # Ensure the df passed is long enough for at least one episode
        # Min length = window_size + episode_length + 1
        sample_episode_len = 10 # For quick test
        sample_window_size = 5  # For quick test
        
        if len(train_df) < sample_window_size + sample_episode_len + 2:
             print(f"Train_df is too short for sample test. Length: {len(train_df)}")
        else:
            print(f"Using train_df for environment test. Length: {len(train_df)}")
            env = TradingEnv(
                df=train_df, # Use a portion or full train_df
                feature_columns=feature_cols,
                window_size=sample_window_size,
                initial_balance=10000,
                episode_length=sample_episode_len, # Short episode for testing
                is_training=True
            )

            # 3. Test the environment with check_env (from Stable Baselines3)
            from stable_baselines3.common.env_checker import check_env
            print("Checking environment with SB3 check_env...")
            try:
                check_env(env, warn=True, skip_render_check=True) # Skip render until properly implemented
                print("Environment check passed!")
            except Exception as e:
                print(f"Environment check failed: {e}")
                import traceback
                traceback.print_exc()


            # 4. Basic interaction loop (optional manual test)
            print("\n--- Manual Interaction Test ---")
            obs, info = env.reset()
            print("Initial Observation:", {k: v.shape for k,v in obs.items()})
            print("Initial Info:", info)
            
            terminated = False
            truncated = False
            total_reward_manual = 0
            
            for i in range(sample_episode_len + 5): # Run a bit longer to see truncation
                action = env.action_space.sample()  # Sample a random action
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward_manual += reward
                print(f"Step {info['episode_step']}: Action={action[0]:.2f}, Reward={reward:.4f}, Equity={info['total_equity']:.2f}, Term={terminated}, Trunc={truncated}")
                if terminated or truncated:
                    print("Episode finished.")
                    print(f"Total reward for manual episode: {total_reward_manual}")
                    obs, info = env.reset() # Test reset again
                    total_reward_manual = 0
                    if i >= sample_episode_len + 3: # Don't loop indefinitely for this simple test
                        break
            env.close()

    except Exception as e:
        print(f"Error during example usage: {e}")
        import traceback
        traceback.print_exc()