# Trader Agent Project: To-Do List

This document outlines the steps to create the Deep Reinforcement Learning Trader Agent,
including live visualization of the training process.

## Phase 1: Project Setup & Core Components

### 1. Project Initialization & Dependencies
    - [x] Create project directory structure (e.g., `rl_trader/`, `rl_trader/data/`, `rl_trader/src/`, `rl_trader/models/`, `rl_trader/results/`, `rl_trader/visualizer/`).
    - [x] Set up a Python virtual environment.
    - [x] Install necessary Python libraries:
        - [x] `pandas`
        - [x] `numpy`
        - [x] `scikit-learn`
        - [x] `matplotlib`
        - [x] `gymnasium`
        - [x] `torch`
        - [x] `stable-baselines3[extra]`
        - [x] `websockets` (for live visualization)
    - [x] Create a `requirements.txt` file.

### 2. Data Loading and Preprocessing (`src/data_manager.py`)
    - [x] **Load Raw Data:** Function to load BTC hourly CSV.
    - [x] **Feature Engineering:** Function to calculate/verify all 33 specified features.
    - [x] **Data Splitting:** Function to split into training and testing sets.

### 3. Custom Trading Environment (`src/trading_env.py`)
    - [x] Create `TradingEnv` class inheriting from `gymnasium.Env`.
    - [x] Implement `__init__`, `reset`, `_get_observation`, `_take_action`, `step`, `_get_info`, `render` (basic), `close`.
    - [x] **(For Live Viz):** Ensure `_get_info()` returns all necessary data for visualization at each step (e.g., current bar's OHLC, precise timestamp, current action taken by agent, trade details if any).
    - [x] Test environment with `stable_baselines3.common.env_checker.check_env`.

### 4. Custom CNN Feature Extractor (`src/custom_cnn.py`)
    - [x] Create `CustomCNN` class inheriting from `stable_baselines3.common.torch_layers.BaseFeaturesExtractor`.
    - [x] Implement `__init__` and `forward` methods to process market data and agent's position.

## Phase 2: Agent Training (Core Loop)

### 5. Training Script (`train_agent.py`)
    - [x] Import `TradingEnv`, `CustomCNN`, `SAC` from `stable_baselines3`.
    - [x] Import SB3 Callbacks (`EvalCallback`, `CheckpointCallback`).
    - [x] Load and preprocess training and validation/test data.
    - [x] Instantiate `TradingEnv` for training and a separate instance for evaluation.
    - [x] Wrap environments with `Monitor` and `DummyVecEnv`.
    - [x] Define `policy_kwargs` for `CustomCNN`.
    - [x] Set up SAC hyperparameters.
    - [x] Configure and add `EvalCallback` (for saving best models and early stopping) and `CheckpointCallback`.
    - [x] Instantiate SAC model.
    - [x] Train the agent: `model.learn(total_timesteps=YOUR_TOTAL_TIMESTEPS, callback=[...])`.
    - [x] Save the final trained model.

## Phase 3: Live Training Visualization (Optional, for Observation)

### 6. Live Training WebSocket Server (`src/live_training_server.py`)
    - [x] **Server Class Implementation:**
        - [x] Create a class `LiveTrainingServer`.
        - [x] Method to start the `websockets` server in a separate `threading.Thread`.
        - [x] The thread's target function runs an `asyncio` event loop for `websockets.serve()`.
        - [x] Implement `async def client_handler(websocket, path)` to manage client connections (add to a set of clients).
        - [x] Implement a thread-safe method `broadcast_data(self, data_dict)`:
            - [x] This method will be callable from the main (SB3 training) thread.
            - [x] It uses `asyncio.run_coroutine_threadsafe(self._broadcast(data_dict), self.loop)` to schedule the actual broadcast on the server's event loop.
        - [x] Implement `async def _broadcast(self, data_dict)` to iterate through connected clients and send JSON data.
        - [x] Handle client disconnections gracefully (remove from the set of clients).
        - [x] Method to stop the server thread and `asyncio` loop.
    - [x] **Data Format:** Define a clear JSON structure for messages (e.g., `type: 'step_data'`, `data: {...}`, or `type: 'episode_reset'`).

### 7. Live Visualization Callback (`src/visualization_callback.py`)
    - [x] Create `VisualizationCallback` class inheriting from `stable_baselines3.common.callbacks.BaseCallback`.
    - [x] **`__init__(self, server_host, server_port, verbose=0)`:**
        - [x] Instantiate and start `LiveTrainingServer` (from `live_training_server.py`).
        - [x] Store the server instance.
    - [x] **`_on_step(self) -> bool:`:**
        - [x] Extract relevant data from `self.locals` and `self.globals`:
            - [x] `infos = self.locals.get("infos", [{}])[0]` (info dict from `TradingEnv.step`). This should contain OHLC of current bar, timestamp, agent action, equity, holdings, etc.
            - [x] `dones = self.locals.get("dones", [False])[0]` (to detect episode end).
        - [x] Format data into a dictionary according to the defined JSON structure.
        - [x] Call `self.server.broadcast_data(data_dict)`.
        - [x] If `dones` is true, also call `self.server.broadcast_data({'type': 'episode_reset'})`.
        - [x] Return `True` to continue training.
    - [x] **`_on_training_end(self)`:**
        - [x] Call the method on `self.server` to stop the WebSocket server thread.

### 8. Update Training Script (`train_agent.py`) for Live Visualization
    - [x] Add an argument/flag (e.g., `--visualize`) to `train_agent.py` to enable live visualization.
    - [x] If visualization is enabled:
        - [x] Import `VisualizationCallback`.
        - [x] Instantiate `vis_callback = VisualizationCallback(host="localhost", port=8765)`.
        - [x] Add `vis_callback` to the list of callbacks passed to `model.learn()`.
    - [x] Ensure graceful shutdown of the `vis_callback` (and its server thread) using a `try...finally` block around `model.learn()`.

### 9. Frontend for Live Training Visualization (`visualizer/`)
    - [ ] **`live_training_visualizer.html`:**
        - [ ] Basic HTML structure.
        - [ ] Include placeholders for: Price Chart (e.g., Lightweight Charts by TradingView or Chart.js), Agent Equity, BTC Holdings, Cash, Last Action, Current Simulated Date/Time, Current Training Episode indicators.
        - [ ] Link to `style.css` and `script.js`.
    - [ ] **`visualizer/style.css`:** Basic styling.
    - [ ] **`visualizer/script.js`:**
        - [ ] **WebSocket Client:** Establish connection, `onmessage` handler to parse JSON.
        - [ ] **DOM Updates:** Update HTML elements with new data.
        - [ ] **Charting Logic:**
            - [ ] Initialize and manage a real-time capable chart.
            - [ ] Update chart with new OHLC data points for each step.
            - [ ] Draw trade markers (buy/sell arrows) if trades occur.
            - [ ] Handle `episode_reset` messages (e.g., clear chart or add a visual separator).
        - [ ] Add a small visual delay or buffer if data arrives too fast for the chart to render smoothly.

## Phase 4: Batch Evaluation & Performance Analysis

### 10. Evaluation Script (`evaluate_agent.py`)
    - [x] Import `TradingEnv`, `SAC`, `CustomCNN`.
    - [x] Load and preprocess testing data.
    - [x] Instantiate `TradingEnv` for testing (`is_training=False`).
    - [x] Load a trained model (e.g., `best_model.zip`).
    - [x] Run evaluation loop (deterministic actions), storing equity history, actions, etc.
    - [x] **Performance Metrics & Visualization:**
        - [x] Calculate and plot Agent Portfolio Value vs. Time against HODL.
        - [ ] (Enhance) Calculate and display: Max Drawdown, Sharpe Ratio (annualized), Sortino Ratio, Total Trades, Win/Loss Ratio, Profit Factor.
        - [x] Plot agent actions over time.
        - [x] Print summary statistics.

## Phase 5: Iteration & Refinement

### 11. Hyperparameter Tuning & Experimentation
    - [ ] Experiment with SAC hyperparameters (`learning_rate`, `batch_size`, `gamma`, etc.).
    - [ ] Experiment with CNN architecture (`features_dim`, layers, filters, kernel sizes).
    - [ ] Experiment with reward shaping.
    - [ ] Experiment with observation space (features, `WINDOW_SIZE`, normalization).
    - [ ] Consider using Optuna for automated hyperparameter optimization.

### 12. Advanced Considerations (Future)
    - [ ] Implement more sophisticated risk management.
    - [ ] Explore other DRL algorithms.
    - [ ] Walk-forward optimization/training.
    - [ ] Multi-asset trading.

## Phase 6: Documentation

### 13. Project `README.md`
    - [ ] Write a comprehensive `README.md`:
        - [ ] Project description and goals.
        - [ ] Setup instructions (dependencies, environment).
        - [ ] How to run training (with and without visualization).
        - [ ] How to run batch evaluation.
        - [ ] Explanation of key scripts and project structure.
        - [ ] Summary of results and potential future work.