import os
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.hybrid_env import HybridTradingEnv

class TradingLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, log_path: str):
        super(TradingLoggerCallback, self).__init__()
        self.check_freq = check_freq
        self.log_path = log_path
        self.metrics = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            infos = self.training_env.env_method("_get_info") if hasattr(self.training_env, "env_method") else [self.training_env.envs[0]._get_info()]
            info = infos[0]
            
            self.metrics.append({
                "timesteps": self.num_timesteps,
                "balance": info["balance"],
                "equity": info["equity"],
                "position": info["position"]
            })
            
            if len(self.metrics) % 10 == 0:
                pd.DataFrame(self.metrics).to_csv(self.log_path, index=False)
        return True

def train_manager(data_path: str, oracle_path: str, total_timesteps: int = 250000, session: str = 'LONDON'):
    print(f"🎬 Initializing SAC Portfolio Manager for {session} Session...")
    
    df = pd.read_csv(data_path)
    
    def make_env():
        return HybridTradingEnv(df, session=session, window_size=30, oracle_path=oracle_path)
        
    vec_env = DummyVecEnv([make_env])
    
    # Soft Actor-Critic (SAC) - The industry standard for continuous state control
    model = SAC(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        ent_coef='auto', # Automatically tuning entropy for risk management
        gamma=0.99,
        tensorboard_log="./sac_manager_tensorboard/"
    )
    
    log_csv_path = f"./logs/manager_metrics_{session.lower()}.csv"
    os.makedirs("./logs", exist_ok=True)
    logger_callback = TradingLoggerCallback(check_freq=1000, log_path=log_csv_path)
    
    print(f"🚀 Commencing Manager SAC training loop for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=logger_callback)
    
    model_path = f"./manager_sac_{session.lower()}.zip"
    model.save(model_path)
    
    print(f"🏆 SAC Manager Weights successfully saved to {model_path}")

if __name__ == "__main__":
    train_manager(
        data_path='../data/processed/master_features_15m.csv', 
        oracle_path='./oracle_lstm.pth',
        total_timesteps=150000, 
        session='LONDON'
    )