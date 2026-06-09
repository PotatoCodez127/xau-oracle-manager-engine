import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

# Route system path to include the parent project directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.hybrid_env import HybridTradingEnv

def run_forward_test(model_path: str, test_data_path: str, oracle_path: str, session: str):
    print(f"Loading Out-of-Sample Validation Data from {test_data_path}...")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Missing test data file at {test_data_path}")
    df_test = pd.read_csv(test_data_path)
    
    print("Initializing Hybrid Environment in Forward Inference Mode...")
    # Instantiate the base environment pointing directly to our trained Oracle model weights
    env = HybridTradingEnv(df=df_test, session=session, window_size=30, oracle_path=oracle_path)
    
    print(f"Loading Trained SAC Capital Allocation Brain from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing SAC model file at {model_path}. Complete training first.")
    model = SAC.load(model_path)
    
    # Reset environment states
    obs, info = env.reset()
    initial_balance = env.initial_balance
    
    # Metric Storage Matrices
    equity_curve = [initial_balance]
    trade_journal = []
    
    winning_pnls = []
    losing_pnls = []
    holding_times = []
    
    active_trade_entry_step = None
    entry_balance = None
    previous_position = 0
    
    print("\n" + "="*50)
    print("   STARTING LIVE HYBRID MARKET SIMULATION STREAM")
    print("="*50)
    
    done = False
    
    while not done:
        # SAC outputs continuous actions: [Direction/Size, TP Multiplier]
        action, _states = model.predict(obs, deterministic=True)
        
        # Advance the simulation universe by exactly one candle
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        current_equity = info['equity']
        current_balance = info['balance']
        current_step = info['step']
        position = info['position']
        
        equity_curve.append(current_equity)
        
        # --- snipe and track live trades ---
        # --- snipe and track live trades ---
        if previous_position == 0 and position != 0:
            # Trade entry detected
            active_trade_entry_step = current_step
            entry_balance = current_balance + env.commission 
            
            direction_name = "LONG" if position == 1 else "SHORT"
            print(f"[LIVE STREAM] 🎯 Entry -> Step: {current_step:<4} | Type: {direction_name:<5} | Units: {env.position_size:.2f} oz")
            print(f"              SL Target: ${env.sl_price:.2f} | TP Target: ${env.tp_price:.2f}")
            
        elif previous_position != 0 and (position == 0 or done):
            # Trade exit triggered by bracket logic, circuit breaker, or end of file
            if active_trade_entry_step is not None:
                bars_held = current_step - active_trade_entry_step
                holding_times.append(bars_held)
                
                # Realized outcome OR unrealized outcome if forced closed
                trade_pnl = current_balance - entry_balance if position == 0 else current_equity - entry_balance
                
                if trade_pnl > 0:
                    winning_pnls.append(trade_pnl)
                    outcome_str = f"✅ TAKE PROFIT (+${trade_pnl:.2f})"
                else:
                    losing_pnls.append(trade_pnl)
                    if position != 0:
                        outcome_str = f"🛑 MARGIN CALL (-${abs(trade_pnl):.2f})"
                    else:
                        outcome_str = f"❌ STOP LOSS (-${abs(trade_pnl):.2f})"
                    
                print(f"[LIVE STREAM] 🏛️ Exit  -> Step: {current_step:<4} | Outcome: {outcome_str} | Held: {bars_held} bars | Account Value: ${current_equity:.2f}\n")
                active_trade_entry_step = None
                
        previous_position = position
        
        # Append to journal structure if a trade is active or processing actions
        if abs(action[0]) > 0.2:
            trade_journal.append({
                "step": current_step,
                "action_direction": action[0],
                "action_tp_mult": action[1],
                "position": position,
                "balance": current_balance,
                "equity": current_equity,
                "max_drawdown_pct": round(((env.peak_equity - current_equity) / env.peak_equity) * 100, 4)
            })

    # --- PERFORMANCE ANALYSIS ENGINE ---
    print("\n" + "="*60)
    print("      HYBRID ORACLE-MANAGER ENGINE SIMULATION COMPLETE")
    print("="*60)
    
    total_wins = len(winning_pnls)
    total_losses = len(losing_pnls)
    total_trades = total_wins + total_losses
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    avg_hold = sum(holding_times) / len(holding_times) if holding_times else 0.0
    net_profit = current_equity - initial_balance
    max_drawdown_pct = ((env.peak_equity - min(equity_curve)) / env.peak_equity) * 100

    print(f"Initial Starting Allocation:   ${initial_balance:,.2f}")
    print(f"Final Ending Portfolio Value:  ${current_equity:,.2f}")
    print(f"Net Strategy Realized PnL:     ${net_profit:,.2f} ({(net_profit/initial_balance)*100:.2f}%)")
    print(f"Maximum Peak-to-Valley DD:     {max_drawdown_pct:.2f}%")
    print("-" * 60)
    print(f"Total Structural Trades Executed: {total_trades}")
    print(f"Strategy Edge Win Rate:           {win_rate:.2f}% ({total_wins} Wins / {total_losses} Losses)")
    print(f"Average Position Holding Time:    {avg_hold:.1f} candles ({avg_hold * 15 / 60:.2f} hours)")
    print("-" * 60)
    
    print("      STRUCTURAL TAKE PROFIT (WIN) METRICS")
    if total_wins > 0:
        print(f"🏆 Highest Realized TP:         +${max(winning_pnls):.2f}")
        print(f"📈 Average Realized TP:         +${(sum(winning_pnls) / total_wins):.2f}")
        print(f"📉 Lowest Realized TP:          +${min(winning_pnls):.2f}")
    else:
        print("No winning trades printed to validation set.")
        
    print("\n      STRUCTURAL STOP LOSS (LOSS) METRICS")
    if total_losses > 0:
        print(f"💀 Worst Realized SL:           -${abs(min(losing_pnls)):.2f}")
        print(f"📉 Average Realized SL:         -${abs(sum(losing_pnls) / total_losses):.2f}")
        print(f"🛡️ Best Controlled SL:          -${abs(max(losing_pnls)):.2f}")
    else:
        print("No losing trades printed to validation set.")
        
    avg_win = sum(winning_pnls) / total_wins if total_wins > 0 else 0
    avg_loss = abs(sum(losing_pnls) / total_losses) if total_losses > 0 else 0
    rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else float('inf')
    
    print("\n      MATHEMATICAL SYSTEM EXPECTANCY")
    print(f"True Managed Risk/Reward Ratio:  1 : {rr_ratio:.2f}")
    print("=" * 60)
    
    # Save the execution sequence to drive
    journal_df = pd.DataFrame(trade_journal)
    journal_path = "hybrid_forward_journal.csv"
    journal_df.to_csv(journal_path, index=False)
    print(f"\nSaved master trading journal logs to: {journal_path}")
    
    # Render Strategy Equity Performance Curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='Hybrid System Account Equity', color='darkcyan', linewidth=1.8)
    plt.axhline(y=initial_balance, color='crimson', linestyle='--', label='Initial Capital Anchor')
    plt.title(f"Aurelius Engine Out-of-Sample Forward Test\nSupervised Oracle + SAC Portfolio Manager Integration")
    plt.xlabel("Timeline Steps (15m Sequential Candles)")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_forward_test(
        model_path='./manager_sac_london.zip',
        test_data_path='../data/processed/test_features_15m.csv',
        oracle_path='./oracle_lstm.pth',
        session='LONDON'
    )