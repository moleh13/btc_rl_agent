# /Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/custom_cnn.py

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the trading environment.
    It processes the 'market_data' (100x33) with 1D CNNs and concatenates
    the 'position_ratio' (1x1) to the flattened CNN output.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted. This corresponds to the dimension
        of the output of this extractor.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        # features_dim is the number of features output by this extractor.
        # We will ensure our network outputs this dimension.
        super().__init__(observation_space, features_dim)

        market_obs_shape = observation_space["market_data"].shape # (window_size, num_market_features)
        # market_obs_shape[1] is num_market_features (33 in our case) - this is in_channels for Conv1d
        # market_obs_shape[0] is window_size (100 in our case) - this is sequence_length

        # CNN for market data
        # Input shape for Conv1d: (batch_size, in_channels, sequence_length)
        # Our market_data comes in as (batch_size, sequence_length, in_channels)
        # So, we'll need to permute it in the forward pass.
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=market_obs_shape[1], # num_market_features (33)
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding="same" # Ensures output has same length as input for stride=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Optional: Add more Conv1D layers or pooling
            # nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding="same"
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),  # (reduces seq-length by ½)
            nn.Flatten(),  # Flattens the output of CNN (batch_size, num_filters * seq_len_after_conv)
        )

        # Compute the shape of the CNN output dynamically
        # To do this, pass a dummy tensor through the CNN part
        with torch.no_grad():
            # Ensure the dummy lives on the same device as the module’s parameters
            device = next(self.parameters()).device
            # Dummy input: (batch_size=1, seq_len=window_size, channels=num_features)
            dummy_market_data = torch.as_tensor(
                observation_space["market_data"].sample()[None]
            ).float().to(device)
            # Permute to (batch_size, channels, seq_len) for Conv1d
            dummy_market_data = dummy_market_data.permute(0, 2, 1)
            cnn_output_dim = self.cnn(dummy_market_data).shape[1]

        # The total number of features before the final linear layer will be
        # cnn_output_dim + shape of position_ratio (which is 1)
        combined_features_dim = cnn_output_dim + observation_space["position_ratio"].shape[0]

        # Linear layer to project combined features to the desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(combined_features_dim, features_dim),
            nn.ReLU() # Or Tanh, or no activation depending on preference
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        market_data_tensor = observations["market_data"] # (batch_size, window_size, num_market_features)
        position_ratio_tensor = observations["position_ratio"] # (batch_size, 1)

        # Permute market_data for Conv1d: (batch_size, num_market_features, window_size)
        cnn_input = market_data_tensor.permute(0, 2, 1)
        
        cnn_features = self.cnn(cnn_input) # (batch_size, cnn_output_dim_calculated_above)
        
        # Concatenate CNN features with position_ratio
        # position_ratio_tensor might need unsqueezing if it's (batch_size) instead of (batch_size, 1)
        # but our obs space defines it as shape (1,), so SB3 should make it (batch_size, 1)
        combined_features = torch.cat([cnn_features, position_ratio_tensor], dim=1)
        
        # Pass through the final linear layer to get the desired features_dim
        extracted_features = self.linear(combined_features)
        
        return extracted_features

if __name__ == '__main__':
    # Example usage:
    from src.trading_env import TradingEnv # Assuming trading_env.py is in the same directory or accessible
    from src.data_manager import load_and_preprocess_data, EXPECTED_FEATURE_COLUMNS
    import os

    print("Testing CustomCNN Feature Extractor...")

    # 1. Load data
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_script_dir, '..', '..', 'BTC_hourly_with_features.csv')
    csv_path = os.path.normpath(csv_path)
    date_col_in_csv = 'timestamp'
    
    try:
        train_df, _, feature_cols = load_and_preprocess_data(
            csv_path, 
            date_column_name=date_col_in_csv,
            expected_features=EXPECTED_FEATURE_COLUMNS
        )

        # 2. Create a sample environment
        sample_env = TradingEnv(
            df=train_df,
            feature_columns=feature_cols,
            window_size=100, # Use the actual window_size
            episode_length=30, # Doesn't matter much for this test
            is_training=True
        )

        # 3. Instantiate the feature extractor
        # Let's use a features_dim of 256 for this example
        desired_features_dim = 256 
        custom_extractor = CustomCNN(sample_env.observation_space, features_dim=desired_features_dim)
        print(f"CustomCNN instantiated with features_dim={desired_features_dim}")
        print(custom_extractor) # Print the model structure

        # 4. Get a sample observation from the environment
        obs, _ = sample_env.reset()

        # 5. Convert observation to PyTorch tensors (mimicking SB3 behavior)
        # SB3 usually handles the batching and conversion. For a single obs:
        obs_tensor = {}
        for key, value in obs.items():
            # Add batch dimension and convert to tensor
            obs_tensor[key] = torch.as_tensor(value[None]).float() 
            # For 'position_ratio', ensure it's (batch_size, 1)
            if key == "position_ratio" and obs_tensor[key].ndim == 1:
                obs_tensor[key] = obs_tensor[key].unsqueeze(-1)


        # 6. Pass observation through the extractor
        extracted_features = custom_extractor(obs_tensor)

        print(f"\nSample observation 'market_data' shape: {obs_tensor['market_data'].shape}")
        print(f"Sample observation 'position_ratio' shape: {obs_tensor['position_ratio'].shape}")
        print(f"Shape of extracted features: {extracted_features.shape}")

        # Check if the output dimension matches features_dim
        assert extracted_features.shape[0] == 1 # Batch size
        assert extracted_features.shape[1] == desired_features_dim, \
            f"Output features_dim mismatch! Expected {desired_features_dim}, Got {extracted_features.shape[1]}"
        
        print("\nCustomCNN test passed successfully!")

    except Exception as e:
        print(f"Error during CustomCNN test: {e}")
        import traceback
        traceback.print_exc()