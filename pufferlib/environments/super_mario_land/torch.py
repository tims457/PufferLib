from functools import partial
import torch
import torch.nn as nn
import pufferlib.models

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


class Policy(nn.Module):
    def __init__(
        self,
        env,
        input_size=512,
        hidden_size=512,
        output_size=512,
        framestack=1,
        flat_size=12288,
        channels_last=True,
        
    ):
        super().__init__()

        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.channels_last = channels_last

        self.screen_network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.feature_network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, flat_size//2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size//2, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init( nn.Linear(hidden_size, env.single_action_space.n), std=0.01)

        # critic
        self.value_fn = pufferlib.pytorch.layer_init( nn.Linear(output_size, 1), std=1)


    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        # Nativize the tensor
        observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

        # Pop out the screen observations, we need to extract features first
        screen_obs = observations.pop("screen")

        if self.channels_last:
            screen_obs = screen_obs.permute(0, 3, 1, 2)
        screen_features = self.screen_network(screen_obs.float()/255.0)


        all_features = torch.cat((screen_features,), dim=1)
        hidden = self.feature_network(all_features)

        return hidden, None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class Policy2(pufferlib.models.Convolutional):
    def __init__(self, env, input_size=512, hidden_size=512, output_size=512,
            framestack=1, flat_size=12288):
        super().__init__(
            env=env,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            framestack=framestack,
            flat_size=flat_size,
        )
