# models/diagnostic_tester.py
import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import SAC
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.hybrid_env import HybridTradingEnv

def run_diagnostic_batch(steps_to_run=500):
    log_file_path = "master_diagnostic_log.txt"
    
    print(f"Spinning up Diagnostic Tester for {steps_to_run} steps...")
    
    # Load Data and Env
    df_test = pd.read_csv('../data/processed/master_features_15m.csv')
    env = HybridTradingEnv(
        df=df_test, 
        session='LONDON', 
        window_size=30, 
        oracle_path='./oracle_lstm.pth', 
        scaler_path='./oracle_scaler.npz'
    )
    
    model = SAC.load('./manager_sac_london.zip')
    obs, info = env.reset()
    
    with open(log_file_path, "w", encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write("🧠 MASTER DIAGNOSTIC LOG: ORACLE-MANAGER PIPELINE\n")
        f.write("=========================================================\n\n")
        
        for i in range(steps_to_run):
            # 1. Peek inside the Observation Space to see what the Oracle is saying
            # obs array = [equity_ratio, drawdown_pct, prob_hold, prob_long, prob_short]
            p_hold, p_long, p_short = obs[2], obs[3], obs[4]
            
            # 2. Get the Agent's Action
            action, _ = model.predict(obs, deterministic=True)
            dir_action, tp_action = action[0], action[1]
            
            # 3. Step the Environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- LOGGING ENGINE ---
            step_num = info['step']
            current_pos = info['position']
            equity = info['equity']
            
            # Only write to log if something interesting is happening (probs > 50% or trade active)
            if p_long > 0.5 or p_short > 0.5 or current_pos != 0 or abs(dir_action) > 0.6:
                f.write(f"--- STEP {step_num} ---\n")
                f.write(f"📊 ORACLE PROBS  | Hold: {p_hold:.2f} | Long: {p_long:.2f} | Short: {p_short:.2f}\n")
                f.write(f"🤖 MANAGER ACTION| Direction Out: {dir_action:.3f} | TP Mult Out: {tp_action:.3f}\n")
                
                if current_pos != 0:
                    pos_str = "LONG" if current_pos == 1 else "SHORT"
                    f.write(f"📈 ACTIVE TRADE  | Type: {pos_str} | Equity: ${equity:.2f}\n")
                else:
                    f.write(f"⚖️ STATUS        | Holding cash. Equity: ${equity:.2f}\n")
                f.write("\n")
                
            if terminated or truncated:
                f.write(f"\n🛑 CIRCUIT BREAKER HIT OR END OF FILE AT STEP {step_num}\n")
                break
                
        f.write("\n=========================================================\n")
        f.write(f"🏁 DIAGNOSTIC COMPLETE. Final Equity: ${info['equity']:.2f}\n")
        f.write("=========================================================\n")
        
    print(f"Diagnostic complete! Check '{log_file_path}' to see exactly what the bot was thinking.")

if __name__ == "__main__":
    run_diagnostic_batch(steps_to_run=1000)