# 🧠 SYSTEM PROMPT & PROJECT MEMORY: HYBRID ORACLE-MANAGER ENGINE

**Core Objective:** Develop, train, and forward-test a hybrid Algorithmic Trading Engine for Gold (XAUUSD) that decouples Pattern Recognition from Risk Management. 

## 🏛️ The "Oracle-Manager" Architecture

### 1. The Oracle (Supervised Deep Sequence Model)
* **Goal:** Eliminate the RL "Credit Assignment Problem" by predicting deterministic mathematical outcomes based on 30m/4H structural zones and DXY momentum.
* **Network:** PyTorch LSTM (`models/oracle_lstm.py`).
* **Input Space:** A rolling 30-candle window of strictly stationary features (Z-score scaled).
* **Target Output:** Multiclass Softmax `[P_Hold, P_Long, P_Short]`. 

### 2. The Manager (Continuous RL Agent)
* **Goal:** Optimize the portfolio equity curve by managing capital deployment based on the Oracle's confidence.
* **Network:** Stable-Baselines3 `SAC` (Soft Actor-Critic) (`models/manager_sac.py`).
* **Observation Space (7D):** Current Account Balance, Drawdown %, Oracle Probabilities (x3), Current 15m ATR, and Bars Held.
* **Action Space:** Continuous `Box` space outputting two values: `Direction` (-1.0 to 1.0) and `TP_Multiplier` (-1.0 to 1.0).

---

## 📜 HISTORY OF ITERATIONS & FIXES (STEPS WE'VE TRIED)

**Phase 1: The "Infinite Hold & Deadzone Trap"**
* **Attempt:** Trained agent on standard step rewards. 
* **Result:** The SAC agent executed 0 trades over 1,000+ candles, despite 90%+ confidence setups, because the environment rewarded it `+0.01` for sitting in cash. 
* **Fix Applied:** Removed idle cash reward, lowered entry threshold. Agent started trading but became too risk-averse.

**Phase 2: Asymmetric Reward Shaping**
* **Attempt:** Tried to break the "Risk-Averse Equilibrium."
* **Fix Applied:** Introduced a `2.0x` multiplier for winning trades and a "Missed Opportunity Sting" (penalty) for holding cash when the Oracle screamed >85% confidence.
* **Result:** Agent's fear was cured. Achieved a 1.62 R:R.

**Phase 3: The "Forced Trading Paradox"**
* **Attempt:** Forward-testing the aggressive model.
* **Result:** Account bled out rapidly (15% win rate). The Oracle was over-regularized (`dropout=0.4`) and went blind. Due to the Phase 2 cash penalty, the Manager was being punished for ignoring erratic signals, forcing it to take terrible setups. 

**Phase 4: Structural Relief & The "Goldilocks" Fix**
* **Attempt:** Re-balancing the SAC mathematical reward space and environment physics to secure the final 1.2% win-rate edge.
* **Fix Applied:** 1. Eradicated the cash penalty entirely, restoring the Manager's right to veto bad setups without financial damage.
    2. Smoothed the continuous action space by removing the hard `0.25` execution cliff. 
    3. Widened Stop Losses to `1.5 * ATR` to prevent higher timeframe (4H) structural trades from being prematurely closed by lower timeframe (15m) noise.
    4. Implemented Distance-to-Open evaluation to cure "intra-candle ambiguity" where SL and TP hits inside the same candle were being recorded as losses due to logical ordering.

---

## 📍 EXACT CURRENT STATUS & ACTIVE GOAL

**Current Status:** The structural physics of the engine are mathematically bulletproof. The intra-candle ambiguity bug suppressing the backtest win rate has been resolved. The Manager is fully empowered to size risk and cut losses dynamically without forced action parameters. 

## 🎯 IMMEDIATE NEXT ACTIONS 
1. **Restore Oracle Vision:** Retrain `oracle_lstm.py` with dialed-back regularization (`dropout=0.3`, `weight_decay=5e-5`) to export optimal weights and a new `oracle_scaler.npz`.
2. **Final Cloud Run:** Execute a 1,000,000 timestep SAC training run on Google Colab with the newly sighted Oracle and the newly un-forced Manager environment (`hybrid_env.py`).