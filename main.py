import numpy as np
import os
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from pettingzoo.classic import connect_four_v3
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import gymnasium as gym

LOG_DIR = 'training'
MODEL_DIR = 'models'

def make_env():
    """Create and wrap PettingZoo environment for single agent training"""
    
    # Create the base environment
    env = connect_four_v3.env(render_mode=None)
    
    # Apply PettingZoo wrappers
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    
    # Convert to single-agent environment
    env = SingleAgentWrapper(env)
    
    return env

class SingleAgentWrapper(gym.Env):
    """
    Wrapper to convert PettingZoo environment to single-agent for SB3
    The agent controls player_0, and player_1 acts randomly
    """
    
    def __init__(self, env):
        self.env = env
        self.env.reset()
        
        # Get spaces from the first agent
        first_agent = self.env.agents[0]
        self.observation_space = self.env.observation_space(first_agent)
        self.action_space = self.env.action_space(first_agent)
        
        self.agent_name = self.env.agents[0]  # Agent we're training
        self.opponent_name = self.env.agents[1]  # Opponent (random)
        
        # Episode tracking for proper logging
        self.episode_reward = 0
        self.episode_length = 0
        
    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        
        # Reset episode tracking
        self.episode_reward = 0
        self.episode_length = 0
        
        # Step to our agent's turn if needed
        while self.env.agent_selection != self.agent_name:
            if self.env.terminations[self.env.agent_selection] or self.env.truncations[self.env.agent_selection]:
                break
            # Random action for opponent
            action = self.env.action_space(self.env.agent_selection).sample()
            self.env.step(action)
        
        obs, _, _, _, info = self.env.last()
        return obs, info
    
    def step(self, action):
        self.episode_length += 1
        
        # Our agent takes action
        self.env.step(action)
        obs, reward, terminated, truncated, info = self.env.last()
        
        # Track cumulative reward
        self.episode_reward += reward
        done = terminated or truncated
        
        # If game not ended, let opponent play
        if not done and self.env.agent_selection == self.opponent_name:
            if not (self.env.terminations[self.opponent_name] or self.env.truncations[self.opponent_name]):
                opponent_action = self.env.action_space(self.opponent_name).sample()
                self.env.step(opponent_action)
                obs, opponent_reward, terminated, truncated, info = self.env.last()
                done = terminated or truncated
                
                # Add opponent's reward (negated) to our tracking
                self.episode_reward += -opponent_reward
        
        # Add episode info for SB3 logging
        if done:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length
            }
        
        return obs, reward, done, False, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()

class OpponentWrapper(gym.Env):
    """
    Alternative wrapper where we can specify opponent behavior
    """
    
    def __init__(self, env, opponent_policy=None):
        self.env = env
        self.opponent_policy = opponent_policy  # Can be 'random', 'model', or a function
        self.env.reset()
        
        first_agent = self.env.agents[0]
        self.observation_space = self.env.observation_space(first_agent)
        self.action_space = self.env.action_space(first_agent)
        
        self.player_agent = self.env.agents[0]
        self.opponent_agent = self.env.agents[1]
        
    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        
        # Step to our agent's turn if needed
        while self.env.agent_selection != self.player_agent:
            if self.env.terminations[self.env.agent_selection] or self.env.truncations[self.env.agent_selection]:
                break
            action = self._get_opponent_action()
            self.env.step(action)
        
        obs, _, _, _, info = self.env.last()
        return obs, info
    
    def step(self, action):
        # Player action
        self.env.step(action)
        obs, reward, terminated, truncated, info = self.env.last()
        
        done = terminated or truncated
        if done:
            return obs, reward, done, False, info
        
        # Opponent action
        if self.env.agent_selection == self.opponent_agent:
            if not (self.env.terminations[self.opponent_agent] or self.env.truncations[self.opponent_agent]):
                opponent_action = self._get_opponent_action()
                self.env.step(opponent_action)
                obs, opp_reward, terminated, truncated, info = self.env.last()
                done = terminated or truncated
                
                # Reward shaping: negative opponent reward when game continues
                if not done:
                    reward = -opp_reward
        
        return obs, reward, done, False, info
    
    def _get_opponent_action(self):
        """Get opponent action based on policy"""
        if self.opponent_policy == 'random' or self.opponent_policy is None:
            return self.env.action_space(self.opponent_agent).sample()
        elif callable(self.opponent_policy):
            obs, _, _, _, _ = self.env.last()
            return self.opponent_policy(obs)
        else:
            # Assume it's a trained model
            obs, _, _, _, _ = self.env.last()
            action, _ = self.opponent_policy.predict(obs, deterministic=False)
            return action
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()

def train_agent(name: str):
    """Train a single agent against random opponent"""
    
    print("Creating environment...")
    env = DummyVecEnv([make_env])
    
    print("Initializing PPO model...")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=os.path.join(LOG_DIR, name)
    )
    
    print("Starting training...")
    model.learn(total_timesteps=100000)
    
    print("Saving model...")
    model.save(os.path.join(MODEL_DIR, name))
    
    return model

def evaluate_trained_agent(model_name: str, num_episodes=10):
    """Evaluate the trained agent"""
    
    model_path = os.path.join(MODEL_DIR, model_name)
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    env = make_env()
    
    wins = 0
    draws = 0
    losses = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total reward = {total_reward}")
        
        if total_reward > 0:
            wins += 1
        elif total_reward == 0:
            draws += 1
        else:
            losses += 1
    
    print(f"\nResults over {num_episodes} episodes:")
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print(f"Win rate: {wins/num_episodes*100:.1f}%")

def display_game(model_name: str, human_vs_ai=False):
    """Display a visual game between agents or human vs AI"""
    model_path = os.path.join(MODEL_DIR, model_name)
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    if human_vs_ai:
        env = create_human_vs_ai_env()
        play_human_vs_ai(model, env)
    else:
        env = create_visual_env()
        play_ai_vs_ai(model, env)

def create_visual_env():
    """Create environment with visual rendering"""
    env = connect_four_v3.env(render_mode="human")
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return VisualWrapper(env)

def create_human_vs_ai_env():
    """Create environment for human vs AI play"""
    env = connect_four_v3.env(render_mode="human")
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return HumanVsAIWrapper(env)

class VisualWrapper:
    """Wrapper for displaying AI vs AI games"""
    
    def __init__(self, env):
        self.env = env
        self.player_agent = None
        self.opponent_agent = None
        
    def reset(self):
        self.env.reset()
        self.player_agent = self.env.agents[0]
        self.opponent_agent = self.env.agents[1]
        
        # Step to first agent if needed
        while self.env.agent_selection != self.player_agent:
            if self.env.terminations[self.env.agent_selection] or self.env.truncations[self.env.agent_selection]:
                break
            action = self.env.action_space(self.env.agent_selection).sample()
            self.env.step(action)
        
        obs, _, _, _, _ = self.env.last()
        return obs
    
    def step(self, action, is_ai_turn=True):
        """Step with indication of whose turn it is"""
        current_agent = self.env.agent_selection
        
        if is_ai_turn:
            print(f"AI ({current_agent}) plays column {action}")
        else:
            print(f"Random opponent ({current_agent}) plays column {action}")
            
        self.env.step(action)
        self.env.render()
        
        obs, reward, terminated, truncated, info = self.env.last()
        return obs, reward, terminated or truncated, info
    
    def render(self):
        self.env.render()

class HumanVsAIWrapper:
    """Wrapper for human vs AI games"""
    
    def __init__(self, env):
        self.env = env
        self.human_agent = None
        self.ai_agent = None
        
    def reset(self):
        self.env.reset()
        self.human_agent = self.env.agents[0]  # Human is player 0
        self.ai_agent = self.env.agents[1]     # AI is player 1
        
        obs, _, _, _, _ = self.env.last()
        return obs
    
    def step(self, action):
        current_agent = self.env.agent_selection
        
        if current_agent == self.human_agent:
            print(f"Human plays column {action}")
        else:
            print(f"AI plays column {action}")
            
        self.env.step(action)
        self.env.render()
        
        obs, reward, terminated, truncated, info = self.env.last()
        return obs, reward, terminated or truncated, info
    
    def get_human_action(self):
        """Get action from human player"""
        while True:
            try:
                print("\nYour turn! Enter column (0-6): ", end="")
                action = int(input())
                if 0 <= action <= 6:
                    return action
                else:
                    print("Invalid column! Please enter 0-6")
            except ValueError:
                print("Please enter a valid number!")
    
    def render(self):
        self.env.render()

def play_ai_vs_ai(model, env):
    """Play and display AI vs random opponent game"""
    
    print("\n" + "="*50)
    print("AI vs Random Opponent - Connect Four")
    print("="*50)
    
    obs = env.reset()
    env.render()
    done = False
    turn_count = 0
    
    import time
    
    while not done:
        turn_count += 1
        current_agent = env.env.agent_selection
        
        if current_agent == env.player_agent:
            # AI's turn
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action, is_ai_turn=True)
        else:
            # Random opponent's turn
            action = env.env.action_space(current_agent).sample()
            obs, reward, done, info = env.step(action, is_ai_turn=False)
        
        # Add delay for better visualization
        time.sleep(1)
        
        if done:
            print(f"\nGame finished after {turn_count} moves!")
            if reward > 0:
                print("🎉 AI wins!")
            elif reward < 0:
                print("😞 AI loses!")
            else:
                print("🤝 It's a draw!")

def play_human_vs_ai(model, env):
    """Play human vs AI game"""
    
    print("\n" + "="*50)
    print("Human vs AI - Connect Four")
    print("You are Red (player 0), AI is Yellow (player 1)")
    print("="*50)
    
    obs = env.reset()
    env.render()
    done = False
    turn_count = 0
    
    while not done:
        turn_count += 1
        current_agent = env.env.agent_selection
        
        if current_agent == env.human_agent:
            # Human's turn
            action = env.get_human_action()
            obs, reward, done, info = env.step(action)
        else:
            # AI's turn
            print("\nAI is thinking...")
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        if done:
            print(f"\nGame finished after {turn_count} moves!")
            if current_agent == env.human_agent:
                if reward > 0:
                    print("🎉 You win!")
                elif reward < 0:
                    print("😞 You lose!")
                else:
                    print("🤝 It's a draw!")
            else:
                if reward > 0:
                    print("🤖 AI wins!")
                elif reward < 0:
                    print("😞 AI loses!")
                else:
                    print("🤝 It's a draw!")

def quick_demo(model_name: str):
    """Quick demo function to show a single game"""
    model_path = os.path.join(MODEL_DIR, model_name)
    print("Quick Demo - AI vs Random Opponent")
    print("-" * 40)
    
    try:
        model = PPO.load(model_path)
        env = create_visual_env()
        play_ai_vs_ai(model, env)
    except FileNotFoundError:
        print(f"Model file '{model_path}.zip' not found!")
        print("Please train a model first using train_agent()")
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Note: Visual rendering requires a display. Try running in a local environment.")

def train_with_custom_opponent():
    """Example of training against a custom opponent"""
    
    def smart_opponent(obs):
        """Simple heuristic opponent - tries to block center column"""
        # This is a placeholder - implement your own strategy
        valid_actions = [i for i in range(7) if obs[0][i] == 0]  # Assuming top row indicates valid moves
        
        # Prefer center columns
        center_actions = [a for a in valid_actions if a in [2, 3, 4]]
        if center_actions:
            return np.random.choice(center_actions)
        else:
            return np.random.choice(valid_actions) if valid_actions else 0
    
    # Create environment with custom opponent
    def make_custom_env():
        base_env = connect_four_v3.env(render_mode=None)
        base_env = wrappers.TerminateIllegalWrapper(base_env, illegal_reward=-1)
        base_env = wrappers.AssertOutOfBoundsWrapper(base_env)
        base_env = wrappers.OrderEnforcingWrapper(base_env)
        return OpponentWrapper(base_env, opponent_policy=smart_opponent)
    
    env = DummyVecEnv([make_custom_env])
    
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("ppo_vs_smart_opponent")
    
    return model

if __name__ == "__main__":
    print("Training RL agent with Stable Baselines 3 + PettingZoo")
    print("=" * 50)

    try:
        # Train the agent
        trained_model = train_agent('ppo_agent')
        
        print("\nTraining completzed! Evaluating agent...")
        
        # Evaluate the trained agent
        evaluate_trained_agent('ppo_agent')
        
        print("\nRunning visual demo...")
        # Display a game
        quick_demo('ppo_agent')
        
        print("\nWant to play against the AI? Uncomment the line below:")
        print("# display_game('ppo_connect_four', human_vs_ai=True)")
        
        print("\nExample: Training against custom opponent...")
        # Uncomment to try custom opponent training
        # custom_model = train_with_custom_opponent()
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Make sure you have installed: pip install stable-baselines3[extra] pettingzoo[classic]")
        
    # Additional demo options:
    print("\n" + "="*50)
    print("DEMO OPTIONS:")
    print("="*50)
    print("1. Watch AI vs Random:     quick_demo()")
    print("2. Play vs AI:             display_game('ppo_connect_four', human_vs_ai=True)")
    print("3. Watch AI vs AI:         display_game('ppo_connect_four', human_vs_ai=False)")
    print("4. Evaluate performance:   evaluate_trained_agent()")
    print("="*50)