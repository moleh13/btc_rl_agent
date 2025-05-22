# /Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/visualization_callback.py

import logging
from stable_baselines3.common.callbacks import BaseCallback
from live_training_server import LiveTrainingServer # Import your server

# Configure basic logging for the callback (optional)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - CALLBACK - %(message)s')

class VisualizationCallback(BaseCallback):
    """
    A custom callback to send training data to a WebSocket server for live visualization.

    :param server_host: Host for the WebSocket server.
    :param server_port: Port for the WebSocket server.
    :param verbose: Verbosity level.
    """
    def __init__(self, server_host: str = "localhost", server_port: int = 8765, verbose: int = 0):
        super().__init__(verbose)
        self.server_host = server_host
        self.server_port = server_port
        self.server = None
        self.current_episode_num = 0 # Start at 0, increment when a new one begins

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        Initialize and start the WebSocket server.
        """
        if self.server is None:
            self.server = LiveTrainingServer(host=self.server_host, port=self.server_port)
            logging.info("Starting WebSocket server from VisualizationCallback...")
            if not self.server.start():
                logging.error("Failed to start WebSocket server. Visualization will not work.")
                self.server = None
            else:
                logging.info("WebSocket server started successfully by VisualizationCallback.")
        self.current_episode_num = 1 # First episode is starting
        # Send initial episode reset so frontend knows episode 1 has begun
        if self.server:
            self.server.broadcast_data({
                "type": "episode_reset",
                "episode_number": self.current_episode_num,
                "total_timesteps": self.num_timesteps # Should be 0 here
            })

    def _on_rollout_start(self) -> None:
        """
        This method is called when a new rollout is initiated.
        We will not use this for sending episode_reset for SAC,
        as 'dones' in _on_step is more reliable for actual episode ends.
        """
        pass # Do nothing here for episode reset for now

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if not self.server:
            return True # Continue training even if server isn't up

        infos = self.locals.get("infos", [{}])[0] 
        actions = self.locals.get("actions") # The action taken by the agent
        rewards = self.locals.get("rewards") # Reward received
        dones = self.locals.get("dones", [False])[0] # Whether the episode ended

        timestamp_obj = infos.get("timestamp_dt")
        # Ensure timestamp_obj is not None and is a pandas Timestamp before calling isoformat
        timestamp_str_for_js = "N/A"
        if timestamp_obj is not None and hasattr(timestamp_obj, 'isoformat'):
            try:
                timestamp_str_for_js = timestamp_obj.isoformat()
            except Exception as e:
                logging.warning(f"Could not format timestamp: {timestamp_obj}, error: {e}")
        elif timestamp_obj is not None: # If it's not None but not a Timestamp (e.g., already a string)
            timestamp_str_for_js = str(timestamp_obj)

        step_data = {
            "type": "step_data",
            "total_timesteps": self.num_timesteps,
            "episode_num": self.current_episode_num,
            "episode_step": infos.get("episode_step", 0), # Get from env info
            "timestamp_str": timestamp_str_for_js,
            "open_price": infos.get("open", None),     # From TradingEnv info for current bar
            "high_price": infos.get("high", None),     # From TradingEnv info
            "low_price": infos.get("low", None),       # From TradingEnv info
            "close_price": infos.get("close", None),   # From TradingEnv info (usually previous close if action at open)
                                                       # Or current close if env gives current bar's full data
            
            "action": actions[0].tolist() if actions is not None else None, # Assuming single continuous action
            "reward": float(rewards[0]) if rewards is not None else None,
            
            "total_equity": infos.get("total_equity", None),
            "balance": infos.get("balance", None),
            "btc_held": infos.get("btc_held", None),
            "current_position_value_btc": infos.get("current_position_value_btc", None),
            "current_position_ratio": infos.get("current_position_ratio", None),
            "hodl_equity": infos.get("hodl_equity", None), # Explicitly include hodl_equity
            "fees_paid_this_step": infos.get("fees_paid_this_step", 0.0), # Add this to TradingEnv info
            "trade_type": infos.get("trade_type_this_step", "NONE"),             # <<< NEW FIELD
            "trade_amount_btc": infos.get("trade_amount_btc_this_step", 0.0),   # <<< NEW FIELD
            "dones": bool(dones) # <--- Add this flag
        }
        
        self.server.broadcast_data(step_data)

        if dones:
            logging.info(f"Episode {self.current_episode_num} ended at total_timesteps: {self.num_timesteps}. Sending reset signal for next.")
            self.current_episode_num += 1 # Increment for the *next* episode
            self.server.broadcast_data({
                "type": "episode_reset",
                "episode_number": self.current_episode_num, # This is the number of the NEW episode
                "total_timesteps": self.num_timesteps # Timesteps at the point of reset
            })

        return True # Continue training

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        Stop the WebSocket server.
        """
        if self.server:
            logging.info("Stopping WebSocket server from VisualizationCallback...")
            self.server.stop()
            logging.info("WebSocket server stopped by VisualizationCallback.")
            self.server = None