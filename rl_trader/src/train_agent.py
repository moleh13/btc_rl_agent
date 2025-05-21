# /Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/train_agent.py

import os
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor # To wrap env for logging episode stats

# Import custom modules
from data_manager import load_and_preprocess_data, EXPECTED_FEATURE_COLUMNS
from trading_env import TradingEnv
from custom_cnn import CustomCNN

# --- Configuration ---
DATA_CSV_PATH = 'BTC_hourly_with_features.csv' # Relative to train_agent.py
DATE_COLUMN_NAME = 'timestamp'

# Environment Hyperparameters
WINDOW_SIZE = 100
INITIAL_BALANCE = 10000
EPISODE_LENGTH = 30 * 24 # 30 days
FEE_RATE = 0.00075

# SAC & Training Hyperparameters
MODEL_FEATURES_DIM = 256 # Output dimension of CustomCNN
TOTAL_TIMESTEPS = 200_000 # Adjust as needed (start small, e.g., 50k, 100k)
LEARNING_RATE = 3e-4 # 0.0003, common default for SAC
BUFFER_SIZE = 100_000 # Size of replay buffer
BATCH_SIZE = 256
GAMMA = 0.99 # Discount factor
TAU = 0.005 # Soft update coefficient
TRAIN_FREQ = (1, "step") # Train every 1 environment step
GRADIENT_STEPS = 1 # How many gradient steps to do after each rollout
LEARNING_STARTS = 1000 # Number of steps to collect experience before training starts

# Callback configurations
EVAL_FREQ = int(5000 / (EPISODE_LENGTH if EPISODE_LENGTH > 0 else 1)) # Evaluate every N * num_envs steps
# Ensure eval_freq is at least 1, considering num_envs which is 1 here
EVAL_FREQ = max(1, EVAL_FREQ)
N_EVAL_EPISODES = 3 # Number of episodes to run for evaluation
PATIENCE_FOR_NO_IMPROVEMENT = 10 # Number of evaluations to wait before stopping if no improvement

# Paths for saving models and logs
LOG_DIR = "./logs/"
MODEL_SAVE_PATH = "./models/"
TENSORBOARD_LOG_PATH = os.path.join(LOG_DIR, "sac_trader_tensorboard/")
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, "best_sac_trader")
CHECKPOINT_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, "checkpoints/")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_SAVE_PATH, exist_ok=True)


def main():
    # --- 1. Load and Preprocess Data ---
    print("Loading and preprocessing data...")
    # Adjust path relative to where train_agent.py is located
    # If train_agent.py is in /rl_trader/src/, and CSV is in /btc_rl/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_full_path = os.path.normpath(os.path.join(script_dir, '..', '..', 'BTC_hourly_with_features.csv'))

    train_df, test_df, feature_cols = load_and_preprocess_data(
        csv_full_path,
        date_column_name=DATE_COLUMN_NAME,
        expected_features=EXPECTED_FEATURE_COLUMNS
    )
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data (for eval) shape: {test_df.shape}")

    if train_df.empty:
        print("Error: Training data is empty. Exiting.")
        return
    if test_df.empty:
        print("Warning: Test data (for evaluation) is empty. EvalCallback might not work as expected.")


    # --- 2. Create Training Environment ---
    print("Creating training environment...")
    train_env_raw = TradingEnv(
        df=train_df,
        feature_columns=feature_cols,
        window_size=WINDOW_SIZE,
        initial_balance=INITIAL_BALANCE,
        episode_length=EPISODE_LENGTH,
        fee_rate=FEE_RATE,
        is_training=True
    )
    # Wrap with Monitor for episode stats (rewards, lengths) for TensorBoard
    train_env_monitored = Monitor(train_env_raw, filename=os.path.join(LOG_DIR, "train_monitor.csv"))
    # Wrap with DummyVecEnv for SB3 compatibility (even for a single env)
    train_env = DummyVecEnv([lambda: train_env_monitored])
    print("Training environment created.")

    # --- 3. Create Evaluation Environment ---
    # Important: Use test_df for evaluation and set is_training=False
    print("Creating evaluation environment...")
    eval_env_raw = TradingEnv(
        df=test_df, # Use test_df for evaluation
        feature_columns=feature_cols,
        window_size=WINDOW_SIZE,
        initial_balance=INITIAL_BALANCE,
        episode_length=EPISODE_LENGTH, # Or test_df length for full test run
        fee_rate=FEE_RATE,
        is_training=False # Crucial: ensures sequential processing of test data
    )
    eval_env_monitored = Monitor(eval_env_raw, filename=os.path.join(LOG_DIR, "eval_monitor.csv"))
    eval_env = DummyVecEnv([lambda: eval_env_monitored])
    print("Evaluation environment created.")


    # --- 4. Define Policy Keywords for Custom CNN ---
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=MODEL_FEATURES_DIM)
    )

    # --- 5. Setup Callbacks ---
    print("Setting up callbacks...")
    # Stop training if there is no improvement after N evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=PATIENCE_FOR_NO_IMPROVEMENT,
        min_evals=PATIENCE_FOR_NO_IMPROVEMENT + 5, # Min evals before it can stop
        verbose=1
    )

    # Evaluate the agent periodically and save the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        log_path=LOG_DIR, # For evaluation logs
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        callback_after_eval=stop_train_callback # Trigger stop_train_callback after each eval
    )

    # Save checkpoints periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=EVAL_FREQ * 5, # Save less frequently than evals
        save_path=CHECKPOINT_SAVE_PATH,
        name_prefix="sac_trader_ckpt"
    )
    print("Callbacks configured.")


    # --- 6. Instantiate SAC Model ---
    print("Instantiating SAC model...")
    model = SAC(
        "MultiInputPolicy", # Required for Dict observation space
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        learning_starts=LEARNING_STARTS,
        tensorboard_log=TENSORBOARD_LOG_PATH,
        # device="mps" if torch.backends.mps.is_available() else "auto" # For M1/M2 Macs if PyTorch supports it well
        device="auto" # "cuda" if available, else "cpu"
    )
    print(f"SAC model instantiated. Using device: {model.device}")


    # --- 7. Train the Agent ---
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback], # Pass list of callbacks
            log_interval=1 # Log training stats every N episodes (adjust as needed for train_monitor.csv)
                           # For SAC, log_interval is usually per episode for Monitor wrapper.
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # --- 8. Save the Final Model ---
        final_model_path = os.path.join(MODEL_SAVE_PATH, "sac_trader_final.zip")
        print(f"Saving final model to {final_model_path}")
        model.save(final_model_path)
        print("Model saved.")
        print("Training complete.")

if __name__ == '__main__':
    main()