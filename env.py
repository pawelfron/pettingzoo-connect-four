import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pettingzoo.classic import connect_four_v3
import random

class Connect4SingleAgentWrapper(gym.Env):
    def __init__(self, opponent_policy="random", render_mode=None):
        super().__init__()
        self.env = connect_four_v3.env(render_mode=render_mode)
        self.opponent_policy = opponent_policy
        self.opponent_model = None
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(7)  # 7 columns
        self.observation_space = gym.spaces.Box(
            low=0, high=2, shape=(6, 7, 2), dtype=np.int8
        )
        
        self.agents = None
        self.agent_selection = None
        
    def set_opponent_model(self, model):
        self.opponent_model = model
        self.opponent_policy = "model"
    
    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents.copy()
        self.agent_selection = self.env.agent_selection
        

        self.our_agent = random.choice(self.agents)
        

        if self.agent_selection != self.our_agent:
            self._opponent_move()
            
        observation = self._get_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        if self.env.terminations[self.agent_selection] or self.env.truncations[self.agent_selection]:
            return self._get_observation(), 0, True, False, {}
            
        self.env.step(action)
        

        if self._is_game_over():
            reward = self._get_reward()
            return self._get_observation(), reward, True, False, {}
        

        self._opponent_move()
        

        if self._is_game_over():
            reward = self._get_reward()
            return self._get_observation(), reward, True, False, {}

        observation = self._get_observation()
        reward = 0
        return observation, reward, False, False, {}
    
    def _opponent_move(self):
        if self._is_game_over():
            return
            
        if self.opponent_policy == "random":
            valid_actions = [i for i in range(7) if self.env.observe(self.agent_selection)['action_mask'][i]]
            if valid_actions:
                action = random.choice(valid_actions)
                self.env.step(action)
        elif self.opponent_policy == "model" and self.opponent_model:
            obs = self._get_observation()
            action, _ = self.opponent_model.predict(obs, deterministic=False)
            if self.env.observe(self.agent_selection)['action_mask'][action]:
                self.env.step(action)
            else:
                valid_actions = [i for i in range(7) if self.env.observe(self.agent_selection)['action_mask'][i]]
                if valid_actions:
                    self.env.step(random.choice(valid_actions))
    
    def _get_observation(self):
        if self._is_game_over():
            return np.zeros((6, 7, 2), dtype=np.int8)
        return self.env.observe(self.our_agent)['observation']
    
    def _get_reward(self):
        if not self._is_game_over():
            return 0

        our_reward = self.env.rewards[self.our_agent]
        
        if our_reward > 0:
            return 1  # Win
        elif our_reward < 0:
            return -1  # Loss
        else:
            return 0  # Draw
    
    def _is_game_over(self):
        return all(self.env.terminations.values()) or all(self.env.truncations.values())

class SelfPlayCallback(BaseCallback):
    def __init__(self, update_frequency=10000, verbose=0):
        super().__init__(verbose)
        self.update_frequency = update_frequency
        self.last_update = 0
        
    def _on_step(self) -> bool:
        # Update opponent model
        if self.num_timesteps - self.last_update >= self.update_frequency:
            if hasattr(self.training_env.envs[0], 'set_opponent_model'):
                self.training_env.envs[0].set_opponent_model(self.model)
                print(f"Updated opponent model at step {self.num_timesteps}")
                self.last_update = self.num_timesteps
        return True