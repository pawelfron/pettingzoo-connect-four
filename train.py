from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import Connect4SingleAgentWrapper, SelfPlayCallback

def train():
    def make_env():
        return Connect4SingleAgentWrapper(opponent_policy="random")
    env = make_vec_env(make_env, n_envs=4, seed=0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./connect4_tensorboard/"
    )

    print("Random training")
    model.learn(total_timesteps=100000)
    model.save("connect4_vs_random")
    

    print("Self-play training...")
    def make_selfplay_env():
        return Connect4SingleAgentWrapper(opponent_policy="model")
    selfplay_env = make_vec_env(make_selfplay_env, n_envs=4, seed=0)

    selfplay_callback = SelfPlayCallback(update_frequency=20000)
    selfplay_model = PPO(
        "MlpPolicy",
        selfplay_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log="./connect4_selfplay_tensorboard/"
    )
    selfplay_model.set_parameters(model.get_parameters())
    selfplay_model.learn(
        total_timesteps=200000,
        callback=selfplay_callback
    )
    selfplay_model.save("connect4_selfplay_final")

if __name__ == "__main__":
    train()
