# /Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/train_agent.py

import os
import pandas as pd
import argparse  # For command line arguments
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor # To wrap env for logging episode stats

# Import custom modules
from data_manager import load_and_preprocess_data, EXPECTED_FEATURE_COLUMNS
from trading_env import TradingEnv
from custom_cnn import CustomCNN
from visualization_callback import VisualizationCallback  # For live visualization

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
EVAL_FREQ = 10000 # Evaluate every 10,000 environment steps (not episodes)
# Ensure eval_freq is at least 1, considering num_envs which is 1 here
EVAL_FREQ = max(1, EVAL_FREQ)
N_EVAL_EPISODES = 5 # Number of episodes to run for evaluation
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
    # --- Argument Parsing for Visualization ---
    parser = argparse.ArgumentParser(description="Train a Deep RL Trading Agent.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable live training visualization via WebSocket."
    )
    args = parser.parse_args()

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
    # Initialize a list to hold all callbacks
    active_callbacks = []
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
    active_callbacks.append(eval_callback)

    # Save checkpoints periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=EVAL_FREQ * 5, # Save less frequently than evals
        save_path=CHECKPOINT_SAVE_PATH,
        name_prefix="sac_trader_ckpt"
    )
    active_callbacks.append(checkpoint_callback)

    # Conditionally add VisualizationCallback
    vis_callback_instance = None
    if args.visualize:
        print("Live visualization enabled. Initializing VisualizationCallback...")
        vis_callback_instance = VisualizationCallback(server_host="localhost", server_port=8765, verbose=1)
        active_callbacks.append(vis_callback_instance)
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
    vis_callback_instance_for_shutdown = None
    if args.visualize:
        for cb in active_callbacks:
            if isinstance(cb, VisualizationCallback):
                vis_callback_instance_for_shutdown = cb
                break
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=active_callbacks, # Pass the list of active callbacks
            log_interval=1 # Log training stats every N episodes (adjust as needed for train_monitor.csv)
                           # For SAC, log_interval is usually per episode for Monitor wrapper.
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # SB3 should call _on_training_end of callbacks on normal exit or KeyboardInterrupt.
        # If vis_callback_instance_for_shutdown is not None and its server is still running,
        # it implies something went wrong with SB3's callback handling or an unhandled exception occurred before _on_training_end.
        # For extra safety, one *could* add an explicit stop here, but it's usually handled by SB3.
        if vis_callback_instance_for_shutdown and \
           hasattr(vis_callback_instance_for_shutdown, 'server') and \
           vis_callback_instance_for_shutdown.server and \
           hasattr(vis_callback_instance_for_shutdown.server, 'server_thread') and \
           vis_callback_instance_for_shutdown.server.server_thread and \
           vis_callback_instance_for_shutdown.server.server_thread.is_alive():
            print("Explicitly stopping visualization server in finally block (might be redundant if SB3 handled it).")
            vis_callback_instance_for_shutdown.server.stop()
        # Save the Final Model
        final_model_path = os.path.join(MODEL_SAVE_PATH, "sac_trader_final.zip")
        print(f"Saving final model to {final_model_path}")
        model.save(final_model_path)
        print("Model saved.")
        print("Training complete.")

if __name__ == '__main__':
    main()