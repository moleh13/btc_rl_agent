# Trader Agent Project: To-Do List

This document outlines the steps to create the Deep Reinforcement Learning Trader Agent.

## Phase 1: Project Setup & Core Components

### 1. Project Initialization & Dependencies
    - [x] Create project directory structure (e.g., `rl_trader/`, `rl_trader/data/`, `rl_trader/src/`, `rl_trader/models/`, `rl_trader/results/`).
    - [x] Set up a Python virtual environment (e.g., using `venv` or `conda`).
    - [x] Install necessary libraries:
        - [x] `pandas` (for data manipulation)
        - [x] `numpy` (for numerical operations)
        - [x] `scikit-learn` (for Z-score normalization, if not implemented manually)
        - [x] `matplotlib` (for plotting results)
        - [x] `gymnasium` (for creating the custom environment, replaces `gym`)
        - [x] `torch` (PyTorch, the backend for Stable Baselines3)
        - [x] `stable-baselines3[extra]` (for DRL algorithms like SAC and utilities)
    - [x] Create a `requirements.txt` file.

### 2. Data Loading and Preprocessing (`src/data_manager.py` or similar)
    - [x] **Load Raw Data:**
        - [x] Function to load the BTC hourly CSV data into a pandas DataFrame.
    - [x] **Feature Engineering:**
        - [x] Function to calculate/verify all 33 specified features:
            - `log_returns`
            - `volatility_30_period`
            - `RSI14`
            - `SMA50`
            - `EMA12, EMA26, EMA100, EMA200`
            - `MACD_line, MACD_signal_line, MACD_histogram`
            - `%K, %D` (Stochastic Oscillator)
            - `ATR`
            - `BB_Middle_Band, BB_Upper_Band, BB_Lower_Band, BB_wWidth`
            - `price_vs_bb_upper, price_vs_bb_lower`
            - `OBV`
            - `VWAP`
            - `ROC10`
            - `Hour_of_Day, Day_of_Week, Day_of_Month, Month_of_Year, Week_of_Year`
    - [x] **Data Splitting:**
        - [x] Function to split the full DataFrame into:
            - Training set: `2017-08-27 13:00:00` to `2023-12-31 23:00:00`
            - Testing set: `2024-01-01 00:00:00` to `2024-12-31 23:00:00`
    - [x] Save preprocessed data if needed (or preprocess on-the-fly in the environment).

### 3. Custom Trading Environment (`src/trading_env.py`)
    - [x] Create `TradingEnv` class inheriting from `gymnasium.Env`.
    - [x] **`__init__(self, df, initial_balance, window_size, episode_length, fee_rate, ...) `:**
        - [x] Store DataFrame, initial balance ($10,000), fee (0.075%), lookback window size (100), episode length (30 days = 720 steps).
        - [x] Define `action_space` (continuous, `Box(-1.0, 1.0, shape=(1,), dtype=np.float32)`).
        - [x] Define `observation_space` (dictionary space: `market_data`: `Box` for 100x33, `position_ratio`: `Box` for 1 value).
    - [x] **`reset(self, seed=None, options=None)`:**
        - [x] If training mode: Randomly select a start index for a 30-day (+100 steps for initial window) segment from the training data.
        - [x] If testing mode (pass a flag or use a separate env instance): Iterate sequentially through test data.
        - [x] Initialize/reset:
            - `current_step` (relative to the start of the episode segment).
            - `balance` (to initial balance).
            - `btc_held` (to 0).
            - `current_position_value` (to 0).
            - `total_equity` (to balance).
            - `current_position_ratio` (0.0).
            - History of equity for plotting/info.
        - [x] Return `self._get_observation()`, `self._get_info()`.
    - [x] **`_get_observation(self)`:**
        - [x] Get the 100-step window of market data ending at `self.current_step -1`.
        - [x] **Normalization:**
            - [x] Apply Z-score normalization to specified price/volume-based columns *within this 100-step window*.
            - [x] Keep other indicator values as-is.
        - [x] Return a dictionary: `{"market_data": normalized_window_data, "position_ratio": self.current_position_ratio}`.
    - [x] **`_take_action(self, action)`:**
        - [x] `action_value = action[0]` (target fraction of equity, from -1.0 to 1.0).
        - [x] `current_price = self.df.loc[self.df.index[self.current_step], 'open']` (trade at next open).
        - [x] Calculate `target_position_value = action_value * self.total_equity`.
        - [x] Calculate `current_position_value_at_trade_price = self.btc_held * current_price`.
        - [x] Calculate `trade_value = target_position_value - current_position_value_at_trade_price`.
        - [x] Calculate `trade_amount_btc = trade_value / current_price`.
        - [x] Calculate `fees = abs(trade_value) * self.fee_rate`.
        - [x] **Execution Logic & Constraints:**
            - [x] If buying: `cost = trade_value + fees`. If `cost > self.balance`, clip `trade_value` so `cost <= self.balance`. Recalculate `trade_amount_btc`.
            - [x] If selling (reducing long or increasing short): `proceeds = abs(trade_value) - fees`.
            - [x] If shorting (or increasing short): Ensure `abs(target_position_value)` after fees doesn't exceed `self.total_equity` (1x leverage). This might require clipping `trade_value` if the requested short is too large relative to cash available to cover potential losses or if the fee makes it impossible. For 1.0x leverage, this basically means total short exposure cannot exceed total equity.
            - [x] The amount of BTC to trade is `trade_amount_btc`.
        - [x] Update `self.btc_held += trade_amount_btc`.
        - [x] Update `self.balance -= (trade_amount_btc * current_price) + fees`.
        - [x] Update `self.current_position_value = self.btc_held * current_price`.
    - [x] **`step(self, action)`:**
        - [x] `previous_total_equity = self.total_equity`.
        - [x] Call `self._take_action(action)`.
        - [x] Update PnL:
            - `current_price_at_step_close = self.df.loc[self.df.index[self.current_step], 'close']` (or next open for consistency if agent holds for full bar). For calculating returns of step `t`, we usually use price at `t+1` vs price at `t`. If agent acts at open of `t`, then its position changes based on `open_t` to `open_{t+1}`.
            - `self.current_position_value = self.btc_held * current_price_at_step_close` (or `open_{t+1}`).
            - `self.total_equity = self.balance + self.current_position_value`.
            - `self.current_position_ratio = self.current_position_value / self.total_equity` if `self.total_equity > 0` else 0.
        - [x] Calculate `agent_log_return = np.log(self.total_equity / previous_total_equity)` (handle division by zero if `previous_total_equity` is 0).
        - [x] Calculate `hodl_price_t = self.df.loc[self.df.index[self.current_step], 'open']`.
        - [x] Calculate `hodl_price_t_plus_1 = self.df.loc[self.df.index[self.current_step + 1], 'open']`.
        - [x] Calculate `hodl_log_return = np.log(hodl_price_t_plus_1 / hodl_price_t)`.
        - [x] `reward = 100 * (agent_log_return - hodl_log_return)`.
        - [x] `self.current_step += 1`.
        - [x] **Termination Conditions:**
            - `terminated = self.total_equity <= (self.initial_balance / 2)`.
            - If `terminated`, `reward -= 100`.
            - `truncated = self.current_step >= (self.start_step_in_df + self.episode_length -1)`. (Check indices carefully)
        - [x] `observation = self._get_observation()`.
        - [x] `info = self._get_info()`.
        - [x] Return `observation, reward, terminated, truncated, info`.
    - [x] **`_get_info(self)`:**
        - [x] Return dictionary with `total_equity`, `btc_held`, `balance`, `current_position_value`, etc. for logging.
    - [x] **`render(self, mode='human')` (Optional):**
        - [x] Simple printout of current status or for future plotting.
    - [x] **`close(self)`:**
        - [x] Cleanup if any.
    - [x] Test environment with `stable_baselines3.common.env_checker.check_env`.

### 4. Custom CNN Feature Extractor (`src/custom_cnn.py`)
    - [x] Create `CustomCNN` class inheriting from `stable_baselines3.common.torch_layers.BaseFeaturesExtractor`.
    - [x] **`__init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 128)`:**
        - [x] `super().__init__(observation_space, features_dim)`.
        - [x] Extract `market_data_shape` and `position_ratio_shape` from `observation_space`.
        - [x] Define CNN layers (e.g., 2 `Conv1D` layers with ReLU, then `Flatten`).
        - [x] Calculate the size of the flattened CNN output.
        - [x] The final combined feature dimension will be `cnn_output_flat_size + position_ratio_shape[0]`.
        - [x] Define a linear layer `self.linear` to project this combined feature vector to `features_dim` if needed, or ensure `features_dim` matches `cnn_output_flat_size + position_ratio_shape[0]`.
    - [x] **`forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:`:**
        - [x] Extract `market_data_tensor` and `position_ratio_tensor` from `observations`.
        - [x] Permute `market_data_tensor` if needed for Conv1D (e.g., `(batch, channels, seq_len)`).
        - [x] Pass `market_data_tensor` through CNN layers.
        - [x] Concatenate flattened CNN output with `position_ratio_tensor`.
        - [x] Return the combined tensor.

## Phase 2: Agent Training & Evaluation

### 5. Training Script (`train_agent.py`)
    - [ ] Import `TradingEnv`, `CustomCNN`, `SAC` from `stable_baselines3`.
    - [ ] Import `SB3` Callbacks (e.g., `EvalCallback`, `CheckpointCallback`).
    - [ ] Load and preprocess training data.
    - [ ] Instantiate `TradingEnv` for training.
    - [ ] (Optional) Wrap with `DummyVecEnv` for SB3 compatibility if not using multiple envs.
    - [ ] Define `policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=DESIRED_FEATURES_DIM))`. Calculate `DESIRED_FEATURES_DIM` based on CNN output + 1.
    - [ ] Hyperparameters for SAC (e.g., `learning_rate`, `buffer_size`, `batch_size`, `gamma`, `tau`, `train_freq`, `gradient_steps`, `learning_starts`).
    - [ ] Instantiate SAC model: `model = SAC("MultiInputPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tb_logs/")`.
    - [ ] (Optional) Set up `EvalCallback` using a separate validation instance of `TradingEnv`.
    - [ ] Train the agent: `model.learn(total_timesteps=YOUR_TOTAL_TIMESTEPS, callback=eval_callback)`.
    - [ ] Save the trained model: `model.save("sac_trader_model")`.

### 6. Evaluation Script (`evaluate_agent.py`)
    - [ ] Import `TradingEnv`, `SAC`, `CustomCNN`.
    - [ ] Load and preprocess testing data.
    - [ ] Instantiate `TradingEnv` for testing (ensure it iterates sequentially through test data and doesn't randomize episode starts).
    - [ ] Load the trained model: `model = SAC.load("sac_trader_model", env=test_env)`.
    - [ ] Run evaluation loop:
        - [ ] `obs, info = test_env.reset()`
        - [ ] `done = False`
        - [ ] `cumulative_reward = 0`
        - [ ] Store episode history (equity, trades, rewards, actions).
        - [ ] While not `done`:
            - [ ] `action, _ = model.predict(obs, deterministic=True)`
            - [ ] `obs, reward, terminated, truncated, info = test_env.step(action)`
            - [ ] `done = terminated or truncated`
            - [ ] Record step data.
    - [ ] **Performance Metrics & Visualization:**
        - [ ] Calculate Total Portfolio Value vs. Time.
        - [ ] Calculate HODL Portfolio Value vs. Time.
        - [ ] Plot both on the same graph.
        - [ ] Calculate Sharpe Ratio, Sortino Ratio.
        - [ ] Calculate Max Drawdown.
        - [ ] Calculate Total Number of Trades.
        - [ ] Calculate Win/Loss Ratio of trades.
        - [ ] Print summary statistics.

## Phase 3: Iteration & Refinement

### 7. Hyperparameter Tuning & Experimentation
    - [ ] Experiment with different SAC hyperparameters.
    - [ ] Experiment with CNN architecture (layers, kernel sizes, number of filters).
    - [ ] Experiment with `features_dim` from the custom extractor.
    - [ ] Experiment with reward shaping (if initial policy is not satisfactory).
    - [ ] Consider using Optuna or similar for hyperparameter optimization.

### 8. Advanced Considerations (Future)
    - [ ] Implement more sophisticated risk management in the environment or agent.
    - [ ] Explore other DRL algorithms (e.g., PPO, TD3).
    - [ ] Add more features or try different normalization schemes.
    - [ ] Walk-forward optimization/training.

### 9. Documentation (`README.md`)
    - [ ] Write a comprehensive `README.md`:
        - [ ] Project description.
        - [ ] How to set up the environment.
        - [ ] How to run training.
        - [ ] How to run evaluation.
        - [ ] Explanation of results and key files.

This provides a solid roadmap. Good luck, this will be a challenging but very interesting project!