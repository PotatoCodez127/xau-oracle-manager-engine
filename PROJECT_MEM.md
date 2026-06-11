# 🧠 SYSTEM PROMPT & PROJECT MEMORY: HYBRID ORACLE-MANAGER ENGINE

**Core Objective:** Develop, train, and forward-test a hybrid Algorithmic Trading Engine for Gold (XAUUSD) that decouples Pattern Recognition from Risk Management. 

## 🏛️ The "Oracle-Manager" Architecture

### 1. The Oracle (Supervised Deep Sequence Model)
* **Goal:** Eliminate the RL "Credit Assignment Problem" by predicting deterministic mathematical outcomes based on 30m/4H structural zones and US Dollar Index (DXY) momentum correlations.
* **Network:** PyTorch LSTM (`models/oracle_lstm.py`).
* **Input Space:** A rolling 30-candle window of strictly stationary features (Z-score scaled via `oracle_scaler.npz` to prevent distribution shifts).
* **Target Output:** Multiclass Softmax `[P_Hold, P_Long, P_Short]`. 
* **Regularization (The "Goldilocks" Zone):** `dropout=0.3` and `weight_decay=5e-5` using the Adam optimizer to balance pattern recognition without memorization. Evaluated at ~69.61% directional accuracy.

### 2. The Manager (Continuous RL Agent)
* **Goal:** Optimize the portfolio equity curve by managing capital deployment based on the Oracle's confidence, effectively acting as a stochastic risk allocator and defensive filter.
* **Network:** Stable-Baselines3 `SAC` (Soft Actor-Critic) (`models/manager_sac.py`). Maximum entropy framework utilized to encourage boundary exploration before exploiting the Oracle's edge.
* **Observation Space (7D):** Current Account Balance, Drawdown %, Oracle Probabilities (x3), Current 15m ATR, and Bars Held.
* **Action Space:** Continuous `Box` space outputting two values: 
    * `Direction` (-1.0 to 1.0): Mapped smoothly via linear interpolation to risk scale without hard logic gates.
    * `TP_Multiplier` (-1.0 to 1.0): Interpolated to target `[1.0, 3.0]` multipliers.

---

## 📜 HISTORY OF ITERATIONS & MATHEMATICAL FIXES 

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
* **Result:** Account bled out rapidly (15% win rate). The Oracle was over-regularized (`dropout=0.4`) and went blind. Due to the Phase 2 cash penalty, the Manager was being punished for ignoring erratic signals, forcing it to take mathematically terrible setups. 

**Phase 4: Structural Relief & Environment Physics**
* **Attempt:** Re-balancing the SAC mathematical reward space and environment physics to secure the final statistical edge.
* **Fixes Applied:** 1. **Eradicated Cash Penalty:** Restored the Manager's right to veto bad setups and hold cash (`0.0` neutral reward) without financial damage.
    2. **Smoothed Action Space:** Removed the hard `> 0.25` execution cliff. Continuous actions now smoothly scale down to $0 position sizing.
    3. **Widened Stop Losses:** Scaled SL to `1.5 * current_atr` to prevent higher timeframe (4H) structural trades from being prematurely closed by lower timeframe (15m) noise.
    4. **Intra-Candle Ambiguity Resolution:** Implemented Distance-to-Open evaluation (`dist_to_sl` vs `dist_to_tp`) to accurately estimate which target was hit first during volatile 15m candles, curing artificially suppressed win rates.

**Phase 5: London Session Cloud Optimization**
* **Attempt:** 1,000,000 timestep SAC training run on Google Colab with the sighted Oracle and un-forced Manager environment.
* **Result:** The maximum entropy objective functioned perfectly. The agent explored aggressive margin boundaries (reaching ~$40k equity early on) before converging. The entropy coefficient successfully stabilized at `0.10`, shifting the agent from exploration to exploitation.

---

## 📍 EXACT CURRENT STATUS

**Current Status:** The engine has successfully crossed the threshold into out-of-sample profitability. During the strict out-of-sample forward test (`hybrid_forward_journal_london.csv`), the decoupled architecture achieved a net realized return of **+17.6%** (Initial: $10,000.00 | Final: $11,761.03). The structural physics of the environment are mathematically robust, and the Manager correctly utilizes the continuous action space to scale risk dynamically rather than forcing trades.