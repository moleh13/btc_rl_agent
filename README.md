# Deep RL Trading Agent for BTC/USD

This project implements a Deep Reinforcement Learning (DRL) agent to trade BTC/USD hourly data. The agent uses a Soft Actor-Critic (SAC) algorithm with a custom CNN feature extractor to process market data and make trading decisions. The project includes training, batch evaluation, and a live training visualizer using WebSockets and a web frontend.

## Project Structure

```
btc_rl/
|-- BTC_hourly_with_features.csv         # Main dataset
|-- requirements.txt                     # Python dependencies
|-- todo.md                              # Project to-do list (you are here!)
|-- .gitignore
|-- rl_trader/                           # Main project folder
|   |-- train_agent.py                   # Script to train the RL agent
|   |-- evaluate_agent.py                # Script to evaluate a trained agent (renamed from evalue_agent.py)
|   |-- src/                             # Source code for custom components
|   |   |-- __init__.py                  # Makes src a package
|   |   |-- data_manager.py              # Loads and preprocesses data
|   |   |-- trading_env.py               # Custom Gym/Gymnasium environment for trading
|   |   |-- custom_cnn.py                # Custom CNN feature extractor for SB3
|   |   |-- live_training_server.py      # WebSocket server for live visualization
|   |   |-- visualization_callback.py    # SB3 callback for live visualization
|   |-- data/                            # (Optional) If you move processed data here
|   |   |-- .gitkeep
|   |-- logs/                            # For TensorBoard logs, Monitor CSVs
|   |   |-- sac_trader_tensorboard/      # TensorBoard event files
|   |   |-- train_monitor.csv
|   |   |-- eval_monitor.csv
|   |-- models/                          # For saved agent models
|   |   |-- best_sac_trader/             # Best models saved by EvalCallback
|   |   |   |-- best_model.zip
|   |   |-- checkpoints/                 # Periodically saved model checkpoints
|   |   |-- sac_trader_final.zip         # Final model after training
|   |   |-- .gitkeep
|   |-- results/                         # For evaluation plots and metrics
|   |   |-- agent_actions.png
|   |   |-- agent_vs_hodl_performance.png
|   |   |-- .gitkeep
|   |-- visualizer/                      # Frontend for live training visualization
|   |   |-- live_training_visualizer.html
|   |   |-- .gitkeep
```

*(Note: I noticed `evalue_agent.py` in your list; I've assumed it's `evaluate_agent.py` in the README. Also, some `logs`, `models`, `results` folders appeared under `src/` in your list, which might be accidental; typically they are at the project root level as shown above.)*

## Features

* **Custom Trading Environment:** A `gymnasium.Env` that simulates hourly BTC trading, including:
    * Observation window of 100 past hours (33 features per hour).
    * Z-score normalization of price/volume features within the observation window.
    * Continuous action space: target portfolio allocation to BTC (-100% to +100% of equity).
    * 1.0x leverage.
    * 0.075% trading fee per transaction.
    * Reward policy designed to protect capital and beat a HODL BTC strategy.
    * Episodic training on random 30-day chunks of historical data.
* **Deep Reinforcement Learning Agent:**
    * Soft Actor-Critic (SAC) algorithm from Stable Baselines3.
    * Custom CNN feature extractor (`CustomCNN`) to process the 100x33 market data window and current portfolio state.
* **Data:**
    * BTC hourly data from `2017-08-27` to `2024-12-31`.
    * Features include OHLCV, log returns, volatility, RSI, SMAs, EMAs, MACD, Stochastic, ATR, Bollinger Bands, OBV, VWAP, ROC, and time-based features.
* **Training & Evaluation:**
    * Separate scripts for training (`train_agent.py`) and batch evaluation (`evaluate_agent.py`).
    * Callbacks for periodic evaluation, saving the best model, and checkpointing.
    * TensorBoard integration for monitoring training progress.
* **Live Training Visualization:**
    * A WebSocket server (`live_training_server.py`) run via a custom SB3 callback (`visualization_callback.py`).
    * An HTML/CSS/JS frontend (`visualizer/live_training_visualizer.html`) that connects to the WebSocket to display:
        * Real-time chart of Agent Equity and HODL Equity.
        * Secondary chart of BTC Open Price with Buy/Sell trade markers.
        * Live updates of portfolio metrics (equity, balance, BTC held, position ratio, etc.).
        * Information about the current training episode and step.

## Setup

1. **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd btc_rl # Or your root project folder name
    ```
2. **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv_rl_trader 
    # or: conda create -n venv_rl_trader python=3.10 (adjust Python version as needed)
    ```
3. **Activate the Environment:**
    * macOS/Linux: `source venv_rl_trader/bin/activate`
    * Windows: `.\venv_rl_trader\Scripts\activate`
    * Conda: `conda activate venv_rl_trader`
4. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure `requirements.txt` includes `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `gymnasium`, `torch`, `stable-baselines3[extra]`, `websockets`, `argparse`).
5. **Data:**
    * Place the `BTC_hourly_with_features.csv` file in the root of the `btc_rl` directory (or adjust paths in scripts accordingly).

## Usage

All scripts are typically run from the `rl_trader/` directory.

### 1. Training the Agent

* **Standard Training (without live visualization):**
    ```bash
    cd rl_trader
    python train_agent.py
    ```
* **Training with Live Visualization:**
    1. Start the training script with the `--visualize` flag:
        ```bash
        cd rl_trader
        python train_agent.py --visualize
        ```
    2. Wait for the console to show that the WebSocket server has started (e.g., "WebSocket server started on ws://localhost:8765").
    3. Open the `rl_trader/visualizer/live_training_visualizer.html` file in a web browser.

* **Monitoring Training:** Use TensorBoard by running the following command in a new terminal from the `rl_trader/` directory:
    ```bash
    tensorboard --logdir ./logs/sac_trader_tensorboard/
    ```
    Then open the provided URL (usually `http://localhost:6006/`) in your browser.

### 2. Evaluating a Trained Agent

After training, a `best_model.zip` (and/or `sac_trader_final.zip`) will be saved in the `rl_trader/models/` directory.

1. **Update Model Path (if needed):**
    Open `rl_trader/evaluate_agent.py` and ensure the `MODEL_PATH` variable points to the desired trained model file (e.g., `"./models/best_sac_trader/best_model.zip"`).
2. **Run Evaluation:**
    ```bash
    cd rl_trader
    python evaluate_agent.py
    ```
    This will run the agent on the test dataset and generate:
    * Console output with performance summary statistics.
    * Plots saved in `rl_trader/results/` (e.g., `agent_vs_hodl_performance.png`, `agent_actions.png`).

## Key Configuration Points

* **`rl_trader/train_agent.py`:** Contains main hyperparameters for training (total timesteps, learning rate, buffer size, batch size, evaluation frequency, etc.).
* **`rl_trader/src/trading_env.py`:** Defines environment parameters (window size, initial balance, fee rate, episode length).
* **`rl_trader/src/visualization_callback.py` & `live_training_server.py`:** WebSocket server host and port are defined here (default: `localhost:8765`).
* **`rl_trader/visualizer/live_training_visualizer.html`:** Contains frontend logic; WebSocket URL is hardcoded here.

## Potential Future Work & Refinements

* **Hyperparameter Optimization:** Use tools like Optuna to systematically find better hyperparameters for the SAC agent and network architecture.
* **Advanced Reward Shaping:** Experiment with more sophisticated reward functions (e.g., incorporating Sharpe ratio, Sortino ratio, or risk-adjusted return metrics).
* **Feature Engineering:** Explore different technical indicators, normalization methods, or ways to represent market state.
* **Alternative DRL Algorithms:** Test other algorithms like PPO or TD3.
* **Walk-Forward Validation/Training:** Implement a more robust training and evaluation scheme that better reflects real-world non-stationarity.
* **Enhanced Visualization:**
    * Add more interactive elements or detailed metrics to the live visualizer.
    * Switch to candlestick charts in the visualizer for more price detail.
* **Risk Management:** Integrate more explicit risk management rules into the agent or environment.

## Results (Placeholder)

*(This section should be updated after thorough training and evaluation)*

* **Training Duration:** X timesteps took approximately Y hours on Z hardware (CPU/GPU).
* **Best Model Performance on Test Set (2024):**
    * Agent Total Return: A%
    * HODL Total Return: B%
    * Agent Sharpe Ratio: C
    * HODL Sharpe Ratio: D
    * Max Drawdown (Agent): E%
    * Max Drawdown (HODL): F%
* **Key Observations:**
    * Describe the learned agent's behavior (e.g., does it manage risk, does it identify trends, common mistakes).
    * Discuss challenges encountered and how they were addressed.

---

This README provides a good overview and instructions for your project. Remember to replace placeholders and adjust any paths or details specific to your final setup.
