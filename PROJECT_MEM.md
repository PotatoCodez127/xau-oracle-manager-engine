# 🧠 SYSTEM PROMPT & PROJECT MEMORY: HYBRID ORACLE-MANAGER ENGINE

**Core Objective:** Develop, train, and forward-test a hybrid Algorithmic Trading Engine for Gold (XAUUSD) that decouples Pattern Recognition from Risk Management. 

## 🏛️ The "Oracle-Manager" Architecture

### 1. The Oracle (Supervised Deep Sequence Model)
* **Goal:** Eliminate the RL "Credit Assignment Problem" by predicting deterministic mathematical outcomes based on 30m/4H structural zones and DXY momentum.
* **Network:** PyTorch LSTM (`models/oracle_lstm.py`).
* **Input Space:** A rolling 30-candle window of strictly stationary features (Z-score scaled).
* **Target Output:** Multiclass Softmax `[P_Hold, P_Long, P_Short]`. 
* **State:** Fully trained and functional. Outputs stable, high-confidence probabilities and accurately exports its scaling statistics (`oracle_scaler.npz`) for live environment normalization.

### 2. The Manager (Continuous RL Agent)
* **Goal:** Optimize the portfolio equity curve by managing capital deployment based on the Oracle's confidence.
* **Network:** Stable-Baselines3 `SAC` (Soft Actor-Critic) (`models/manager_sac.py`).
* **Observation Space:** Current Account Balance, Drawdown %, and the Oracle's live probability feed.
* **Action Space:** Continuous `Box` space outputting two values: `Direction` (-1.0 to 1.0) and `TP_Multiplier` (-1.0 to 1.0).
* **State:** Training pipeline is mathematically stable. Critic loss has been cured via Reward Scaling, and Entropy Collapse has been cured by setting a fixed `ent_coef=0.10`.

## 🛠️ Diagnostic Infrastructure
The project now includes `models/diagnostic_tester.py`, which generates a `master_diagnostic_log.txt`. This file unifies the Oracle's internal probabilities, the Manager's raw continuous output, and the Environment's physical step outcomes into a single readable log. This is the primary tool for debugging agent behavior.

## 📍 EXACT CURRENT STATUS & ACTIVE BUG

**Current Status:** The mathematical architecture is bulletproof. The data leaks (ATR `.bfill()`) and normalization blindness issues have been resolved. The agent's gradients are no longer exploding.

**Active Bug: The "Deadzone Trap" (Zero Trades)**
* **Symptom:** In the most recent diagnostic run, the SAC agent executed 0 trades over 1,000+ candles, despite the Oracle providing 90%+ confidence setups.
* **The Cause:** In `environment/hybrid_env.py`, the action threshold for entry is currently set too high (e.g., `abs(direction_action) > 0.6`). When a neural network is freshly initialized, its weights are nearly random, clustering outputs near `0.0` (e.g., `0.25`). 
* **The RL Paradox:** Because the agent's initial random outputs never breach the `0.6` threshold, it never takes a trade. Because it never takes a trade, it receives `0.0` scaled reward. Because the reward is `0.0`, the gradients do not update, and the network never learns that it needs to push its weights higher to execute a trade. It is trapped in a zero-gradient void.

## 🎯 IMMEDIATE NEXT ACTIONS (For the AI Agent)
1. **Shrink the Deadzone:** Update the entry threshold in `environment/hybrid_env.py`'s `step()` function from `0.6` down to `0.25` (or lower) to allow the freshly initialized network to successfully stumble into trades and spark the gradient descent.
2. **Verify Risk Profile:** Ensure that lowering the entry threshold does not expose the bot to maximum risk. The `risk_pct` interpolation must remain tightly controlled (e.g., `0.5%` to `2.0%`).
3. **Retrain & Diagnose:** Wipe the old `manager_sac_london.zip`, retrain via `manager_sac.py`, and run `diagnostic_tester.py` to verify that active trades are executing and `Direction Out` begins correlating with the Oracle's highest probability.