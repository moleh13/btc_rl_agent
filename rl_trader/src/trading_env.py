# /Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.stats import zscore

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, df, feature_columns, window_size=100, initial_balance=10000,
                 episode_length=720, fee_rate=0.00075, is_training=True,
                 reward_scale_factor=100.0,      # Primary scaler for performance and scaled costs
                 fee_penalty_factor=1.0,         # Weight for fee penalty (applied to fee amount)
                 action_change_penalty_factor=0.5, # Weight for action delta penalty (range [0,2] -> penalty)
                 max_leverage: float = 1.0,
                 carry_cost_rate: float = 0.00005,    # Actual per-period cost rate for holding a position
                 position_penalty_factor: float = 0.01, # Weight for quadratic position size penalty
                 benchmark_weight: float = 0.05,    # Weight for outperforming benchmark log-return
                 use_sortino_ratio: bool = False,   # Whether to use Sortino-like ratio
                 sortino_target_return: float = 0.0 # Target return for Sortino (e.g., risk-free rate)
                 ):
        super().__init__()

        self.df = df.reset_index() # Ensure simple integer indexing for iloc
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.episode_length = episode_length # 30 days * 24 hours
        self.fee_rate = fee_rate
        self.is_training = is_training # To control random episode starts

        # Reward parameters
        self.reward_scale_factor = reward_scale_factor
        self.fee_penalty_factor = fee_penalty_factor
        self.action_change_penalty_factor = action_change_penalty_factor
        self.max_leverage = max_leverage
        self.carry_cost_rate = carry_cost_rate # Store the raw rate
        self.position_penalty_factor = position_penalty_factor
        self.benchmark_weight = benchmark_weight
        self.use_sortino_ratio = use_sortino_ratio
        self.sortino_target_return = sortino_target_return
        self.risk_free_rate_for_performance = 0.0 # Can be set as param if needed

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

        # --- ADD HODL TRACKING VARIABLES ---
        self.hodl_btc_held = 0.0
        self.hodl_initial_price = 0.0
        self.hodl_equity = initial_balance # Starts at initial balance
        # --- END HODL TRACKING VARIABLES ---

        self.previous_action_value = 0.0 # <- NEW ATTRIBUTE
        # --- For risk-adjusted reward ---
        self.recent_log_returns = []  # buffer for rolling volatility
        self.rolling_vol_window = 20  # can be parameterized

        # For live/avg_position_ratio_1h and live/long_exposure_pct
        from collections import deque
        self._pos_history = deque(maxlen=24)  # for avg_position_ratio_1h
        self._long_exposure_count = 0
        self._short_exposure_count = 0
        self._exposure_window = deque(maxlen=100)  # for exposure %

    def _get_observation(self):
        # Observation window: data from (current_decision_step_idx - window_size) to (current_decision_step_idx - 1)
        # So, if current_decision_step_idx is 100, window is df.iloc[0:100]
        start_idx = self.current_decision_step_idx - self.window_size
        end_idx = self.current_decision_step_idx # Exclusive end, so it takes up to current_decision_step_idx-1

        if start_idx < 0: # Should not happen if reset is handled correctly
            raise ValueError("start_idx for observation window is negative.")

        market_window_df = self.df.iloc[start_idx:end_idx][self.feature_columns].copy()

        # Apply Z-score normalization to specified columns *within the window*
        if self.z_score_cols:          # only if the list is non-empty
            sub = market_window_df[self.z_score_cols].to_numpy(dtype=np.float32)
            mu  = sub.mean(axis=0, keepdims=True)
            sigma = sub.std(axis=0, keepdims=True)
            sigma[sigma == 0] = 1.0    # avoid div-by-zero
            market_window_df[self.z_score_cols] = (sub - mu) / sigma
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
        previous_btc_held_before_trade = self.btc_held # Store current BTC held

        if trade_price <= 0: # Safety for bad data
            return 0.0, False, 0.0 # No trade, no fees, no trade executed, no change

        # Recalculate total equity just before trade decision based on trade_price
        # This ensures the target position is based on the most up-to-date equity valuation
        current_position_value_at_trade_price = self.btc_held * trade_price
        equity_at_trade_moment = self.balance + current_position_value_at_trade_price

        if equity_at_trade_moment <= 0: # Cannot trade if no equity
            return 0.0, False, 0.0

        # --- clamp by max_leverage -----------------------------------
        # We allow |position| ≤ equity × max_leverage
        max_abs_position_value = equity_at_trade_moment * self.max_leverage
        target_btc_value = np.clip(action_value * equity_at_trade_moment,
                                   -max_abs_position_value,
                                   +max_abs_position_value)
        # --------------------------------------------------------------
        target_btc_quantity = target_btc_value / trade_price

        trade_quantity_btc = target_btc_quantity - self.btc_held
        executed_trade_value_abs = 0.0
        fees_paid = 0.0
        trade_executed_flag = False # Initialize

        if abs(trade_quantity_btc * trade_price) < 1e-3 : # Negligible trade, effectively no trade
            trade_quantity_btc = 0.0
        else:
            trade_executed_flag = True # A non-negligible trade is about to happen

        previous_btc_held = self.btc_held # Store before modification

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
            executed_trade_value_abs = value_to_sell_btc
            fees_paid = executed_trade_value_abs * self.fee_rate
            self.balance += (executed_trade_value_abs - fees_paid)
            self.btc_held += trade_quantity_btc # trade_quantity_btc is negative

        self.current_position_value_btc = self.btc_held * trade_price # Update after trade
        self.total_equity = self.balance + self.current_position_value_btc
        actual_trade_amount_btc = self.btc_held - previous_btc_held_before_trade # <- CALCULATE THIS
        return fees_paid, trade_executed_flag, actual_trade_amount_btc # <- MODIFIED RETURN

    def _calculate_reward(self, current_total_equity, equity_at_start_of_step):
        # Simple PnL reward: change in equity
        return current_total_equity - equity_at_start_of_step

    def step(self, action):
        # Store equity at the start of the step
        equity_at_start_of_step = self.total_equity
        current_action_value = action[0]
        current_bar_open_price_for_baseline = self.df.iloc[self.current_decision_step_idx]['open']
        equity_before_trade_at_current_open = self.balance + (self.btc_held * current_bar_open_price_for_baseline)
        if equity_before_trade_at_current_open <= 0:
            equity_before_trade_at_current_open = 1e-9
        fees_paid_this_step, trade_executed_flag, actual_trade_amount_btc = self._take_action(current_action_value)
        current_bar_data = self.df.iloc[self.current_decision_step_idx]
        # --- PnL logic: use next bar's open, or current bar's close if at end ---
        if (self.current_decision_step_idx + 1) >= len(self.df):
            price_at_end_of_step_period = current_bar_data['close']
        else:
            price_at_end_of_step_period = self.df.iloc[self.current_decision_step_idx + 1]['open']
        self.total_equity = self.balance + (self.btc_held * price_at_end_of_step_period)
        self.episode_step_count += 1
        self.current_decision_step_idx += 1
        if self.total_equity > 0:
            self.current_position_value_btc = self.btc_held * price_at_end_of_step_period
            self.current_position_ratio = self.current_position_value_btc / self.total_equity
        else:
            self.current_position_value_btc = 0.0
            self.current_position_ratio = 0.0
        self.current_position_ratio = np.clip(self.current_position_ratio, -self.max_leverage, self.max_leverage)
        # Calculate reward as pure PnL
        reward = self._calculate_reward(self.total_equity, equity_at_start_of_step)
        terminated = False
        if self.total_equity <= (self.initial_balance / 2):
            terminated = True
        # No explicit penalty; PnL will be strongly negative
        truncated = False
        if self.episode_step_count >= self.episode_length:
            truncated = True
        if not terminated and not truncated and (self.current_decision_step_idx >= len(self.df)):
            truncated = True
        observation = self._get_observation()
        info = self._get_info_for_step({
            "fees_paid": fees_paid_this_step,
            "trade_type": "TRADE" if trade_executed_flag else "NONE",
            "trade_amount_btc": actual_trade_amount_btc
        })
        if self.render_mode == "human":
            self._render_human()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.current_position_value_btc = 0.0
        self.total_equity = self.initial_balance
        self.current_position_ratio = 0.0
        self.episode_step_count = 0
        if self.is_training:
            if self.max_possible_steps <= 0:
                 raise ValueError("DataFrame is too short for window and episode length.")
            start_df_idx = self.np_random.integers(0, self.max_possible_steps + 1)
            self.current_decision_step_idx = start_df_idx + self.window_size
        else:
            self.current_decision_step_idx = self.window_size
        self.hodl_equity = self.initial_balance
        self.hodl_initial_price = self.df.iloc[self.current_decision_step_idx]['open']
        if self.hodl_initial_price > 0:
            self.hodl_btc_held = (self.initial_balance / self.hodl_initial_price) * (1 - self.fee_rate)
            self.hodl_equity = self.hodl_btc_held * self.hodl_initial_price
        else:
            self.hodl_btc_held = 0
            self.hodl_equity = self.initial_balance
        self.previous_action_value = 0.0
        self.recent_log_returns = []
        observation = self._get_observation()
        first_decision_bar_open = self.df.iloc[self.current_decision_step_idx]['open']
        self.current_position_value_btc = self.btc_held * first_decision_bar_open
        self.total_equity = self.balance + self.current_position_value_btc
        if self.total_equity > 0:
            self.current_position_ratio = self.current_position_value_btc / self.total_equity
        else:
            self.current_position_ratio = 0.0
        self.current_position_ratio = np.clip(self.current_position_ratio, -self.max_leverage, self.max_leverage)
        info = self._get_info_for_step({"fees_paid": 0.0, "trade_type": "NONE", "trade_amount_btc": 0.0})
        return observation, info

    def _get_info_for_step(self, step_results):
        """Helper to construct the info dictionary consistently."""
        current_bar_index_for_info = self.current_decision_step_idx - 1
        if current_bar_index_for_info < 0:
            current_bar_index_for_info = self.current_decision_step_idx
        current_bar_series = self.df.iloc[current_bar_index_for_info]
        # --- UPDATE HODL EQUITY FOR THE CURRENT STEP'S PRICE ---
        current_close_price_for_hodl_valuation = float(current_bar_series['close'])
        if current_close_price_for_hodl_valuation > 0:
            self.hodl_equity = self.hodl_btc_held * current_close_price_for_hodl_valuation
        # --- END HODL EQUITY UPDATE ---
        fees_paid_this_step = step_results.get("fees_paid", 0.0)
        trade_type_this_step = step_results.get("trade_type", "NONE")
        trade_amount_btc_this_step = step_results.get("trade_amount_btc", 0.0)
        return {
            "total_equity": self.total_equity,
            "balance": self.balance,
            "btc_held": self.btc_held,
            "current_position_value_btc": self.current_position_value_btc,
            "current_position_ratio": self.current_position_ratio,
            "episode_step": self.episode_step_count,
            "current_df_idx": self.current_decision_step_idx,
            "timestamp_dt": self.df.iloc[current_bar_index_for_info]['timestamp'],
            "open": float(current_bar_series['open']),
            "high": float(current_bar_series['high']),
            "low": float(current_bar_series['low']),
            "close": float(current_bar_series['close']),
            "fees_paid_this_step": fees_paid_this_step,
            "trade_type_this_step": trade_type_this_step,       # <<< NEW FIELD
            "trade_amount_btc_this_step": trade_amount_btc_this_step, # <<< NEW FIELD
            "hodl_equity": self.hodl_equity
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
    from data_manager import load_and_preprocess_data, EXPECTED_FEATURE_COLUMNS

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
                is_training=True,
                max_leverage=1.0 # ← ADDED
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