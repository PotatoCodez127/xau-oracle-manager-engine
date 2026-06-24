import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import sys
import os

# Import the Oracle architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from oracle_lstm import OracleLSTM

class HybridTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, session: str = 'ALL', window_size: int = 30, initial_balance: float = 10000.0, oracle_path: str = '../models/oracle_lstm.pth', scaler_path: str = '../models/oracle_scaler.npz'):
        super(HybridTradingEnv, self).__init__()
        
        self.window_size = window_size
        self.df = self._filter_session(df.copy(), session).reset_index(drop=True) 
        self._calculate_atr()
        
        # Isolate features
        self.feature_cols = [
            c for c in self.df.columns 
            if self.df[c].dtype in [np.float64, np.float32, np.int64, np.int32] 
            and not c.startswith('env_') and c != 'target' and c != 'unnamed: 0'
        ]
        self.data = self.df[self.feature_cols].values.astype(np.float32)
        
        # --- PRE-TRADE OUT-OF-SAMPLE FIREWALL VERIFICATION ---
        if os.path.exists(scaler_path):
            scaler_data = np.load(scaler_path)
            self.feature_mean = scaler_data['mean']
            self.feature_std = scaler_data['std']
        else:
            # Enforce strict frozen parameter propagation to block lookahead validation leakage
            raise FileNotFoundError(
                f"Frozen scale boundaries missing at '{scaler_path}'. "
                "Calculated parameters must be derived exclusively from the active training partition "
                "split and passed forward to transform evaluation tensors."
            )
        
        # --- LOAD THE ORACLE ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_dim = self.data.shape[1] if self.data.shape[1] > 0 else 10 
        self.oracle = OracleLSTM(input_dim=input_dim).to(self.device)
        
        if os.path.exists(oracle_path):
            self.oracle.load_state_dict(torch.load(oracle_path, map_location=self.device, weights_only=True))
        self.oracle.eval()
        
        # --- CONTINUOUS ACTION SPACE ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # --- MODERNIZED 9D OBSERVATION SPACE CONSTRAINTS ---
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(9,), dtype=np.float32)
        
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.peak_equity = self.initial_balance
        self.current_step = self.window_size
        
        # Trade State
        self.position = 0 
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.position_size = 0.0
        self.bars_held = 0 
        
        self.spread = 0.15 
        self.commission = 0.10 
        self.max_hold = 32 
        
    def _calculate_atr(self):
        high_low = self.df['env_high'] - self.df['env_low']
        high_close = np.abs(self.df['env_high'] - self.df['env_close'].shift())
        low_close = np.abs(self.df['env_low'] - self.df['env_close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        self.df['env_atr'] = true_range.rolling(14).mean().bfill(limit=14).fillna(0.5)
        
    def _filter_session(self, df: pd.DataFrame, session: str) -> pd.DataFrame:
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        if session == 'LONDON':
            return df.between_time('09:00', '14:00')
        elif session == 'NY':
            return df.between_time('14:00', '18:00')
        return df

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.peak_equity = self.initial_balance
        self.current_step = self.window_size
        self.position = 0
        self.bars_held = 0
        return self._next_observation(), self._get_info()

    def _next_observation(self):
        raw_obs = self.data[self.current_step - self.window_size : self.current_step]
        norm_obs = (raw_obs - self.feature_mean) / self.feature_std
        
        with torch.no_grad():
            tensor_obs = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            if hasattr(self.oracle, 'fc'):
                logits = self.oracle(tensor_obs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            else:
                probs = [0.33, 0.33, 0.33]
            
        info = self._get_info()
        equity_ratio = info["equity"] / self.initial_balance
        drawdown_pct = (self.peak_equity - info["equity"]) / self.peak_equity
        
        current_atr = self.df.loc[self.current_step, 'env_atr']
        
        # Calculate real-time portfolio scale factors
        current_price = self.df.loc[self.current_step, 'env_close']
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) * self.position_size
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) * self.position_size
        scaled_pnl = unrealized_pnl / self.initial_balance

        # Complete Portfolio Layering Matrix Matrix Output (9 Dimensions)
        return np.array([
            equity_ratio, 
            drawdown_pct, 
            probs[0], 
            probs[1], 
            probs[2], 
            current_atr, 
            float(self.position),
            scaled_pnl,
            float(self.bars_held)
        ], dtype=np.float32)

    def _get_info(self):
        current_price = self.df.loc[self.current_step, 'env_close']
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) * self.position_size
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) * self.position_size
            
        return {
            "balance": self.balance,
            "equity": self.balance + unrealized_pnl,
            "position": self.position,
            "step": self.current_step
        }

    def step(self, action):
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        current_price = self.df.loc[self.current_step, 'env_close']
        current_high = self.df.loc[self.current_step, 'env_high']
        current_low = self.df.loc[self.current_step, 'env_low']
        current_atr = self.df.loc[self.current_step, 'env_atr']
        
        reward = 0.0
        trade_closed_this_step = False 
        
        # ==========================================
        # 1. MANAGE ACTIVE POSITIONS (WITH INTRA-CANDLE RESOLUTION)
        # ==========================================
        if self.position != 0:
            self.bars_held += 1
            current_open = self.df.loc[self.current_step, 'env_open']
            
            dist_to_sl = abs(current_open - self.sl_price)
            dist_to_tp = abs(current_open - self.tp_price)
            
            # Check LONG Exits
            if self.position == 1:
                hit_sl = current_low <= self.sl_price
                hit_tp = current_high >= self.tp_price
                
                if hit_sl and hit_tp:
                    if dist_to_sl < dist_to_tp:
                        hit_tp = False  
                    else:
                        hit_sl = False  
                        
                if hit_sl:
                    pnl = (self.sl_price - self.entry_price) * self.position_size 
                    self.balance += pnl
                    reward += pnl  
                    trade_closed_this_step = True
                elif hit_tp:
                    pnl = (self.tp_price - self.entry_price) * self.position_size 
                    self.balance += pnl
                    reward += (pnl * 2.0)  
                    trade_closed_this_step = True

            # Check SHORT Exits
            elif self.position == -1:
                hit_sl = current_high >= self.sl_price
                hit_tp = current_low <= self.tp_price
                
                if hit_sl and hit_tp:
                    if dist_to_sl < dist_to_tp:
                        hit_tp = False  
                    else:
                        hit_sl = False  
                        
                if hit_sl:
                    pnl = (self.entry_price - self.sl_price) * self.position_size 
                    self.balance += pnl
                    reward += pnl  
                    trade_closed_this_step = True
                elif hit_tp:
                    pnl = (self.entry_price - self.tp_price) * self.position_size 
                    reward += (pnl * 2.0)  
                    self.balance += pnl
                    trade_closed_this_step = True
            
            if not trade_closed_this_step and self.bars_held >= self.max_hold:
                pnl = (current_price - self.entry_price) * self.position_size if self.position == 1 else (self.entry_price - current_price) * self.position_size
                reward += pnl
                self.balance += pnl
                trade_closed_this_step = True

            if trade_closed_this_step:
                self.position = 0
                self.position_size = 0.0
                self.bars_held = 0

        # ==========================================
        # 2. EVALUATE NEW ENTRIES
        # ==========================================
        if self.position == 0 and not trade_closed_this_step:
            direction_action = action[0] 
            tp_mult_action = action[1] 
            
            if abs(direction_action) > 0.1: 
                risk_pct = np.interp(abs(direction_action), [0.1, 1.0], [0.005, 0.02])
                risk_dollar_amount = self.balance * risk_pct
                
                safe_atr = max(current_atr, 0.1)
                self.position_size = risk_dollar_amount / safe_atr 
                
                tp_multiplier = np.interp(tp_mult_action, [-1.0, 1.0], [1.0, 3.0])
                
                self.balance -= self.commission
                reward -= self.commission
                
                sl_distance = current_atr * 1.5 
                
                if direction_action > 0: 
                    self.position = 1
                    self.entry_price = current_price + self.spread
                    self.sl_price = self.entry_price - sl_distance
                    self.tp_price = self.entry_price + (current_atr * tp_multiplier)
                else: 
                    self.position = -1
                    self.entry_price = current_price - self.spread
                    self.sl_price = self.entry_price + sl_distance
                    self.tp_price = self.entry_price - (current_atr * tp_multiplier)
                self.bars_held = 0

        # ==========================================
        # 3. GLOBAL STATE & METRICS
        # ==========================================
        info = self._get_info()
        
        if info["equity"] > self.peak_equity:
            self.peak_equity = info["equity"]
            
        if info["equity"] <= (self.initial_balance * 0.90):
            terminated = True
            reward -= (self.initial_balance * 0.10) 

        scaled_reward = reward / self.initial_balance
        scaled_reward = scaled_reward * 10.0
        scaled_reward = np.clip(scaled_reward, -1.0, 1.0)

        return self._next_observation(), scaled_reward, terminated, truncated, info