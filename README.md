# XAU/USD Hybrid Oracle-Manager Deep Reinforcement Learning Engine

An enterprise-grade, volatility-resilient algorithmic trading architecture for Spot Gold (XAUUSD) designed to decouple **Pattern Recognition** from **Stochastic Risk Management**.

## 🔬 Architectural Topology

The core system architecture explicitly decouples prediction mechanics from capital management to eliminate the Reinforcement Learning "Credit Assignment Problem" over non-stationary financial structures.
```
[ Feature Feed ] ──> ( 30-Candle Window ) ──> [ Supervised Oracle LSTM ]
│
(3D Softmax Probabilities)
│
▼
[ Account State ] ──────────────────────────> [ Stochastic Manager SAC ]
│
(Continuous Sizing & Multipliers)
│
▼
[ Simulated XAU Spot Market ]
```
### 1. The Oracle (Supervised Deep Sequence Model)
* **Objective:** Projects high-probability structural trajectories based on 30-minute and 4-hour wick-based Support & Resistance zones combined with US Dollar Index (DXY) momentum indicators.
* **Network Topology:** Sequential PyTorch LSTM processing a rolling 30-candle sequence of strictly stationary features.
* **Output Vector:** Multiclass Softmax: `[P_Hold, P_Long, P_Short]`. Directional accuracy registers at **~69.61%**.

### 2. The Manager (Continuous Deep RL Agent)
* **Objective:** Acts as a defensive capital allocator and risk gatekeeper, optimizing the portfolio equity curve based on the Oracle's classification confidence.
* **Network Topology:** Stable-Baselines3 Soft Actor-Critic (SAC) utilizing a continuous Maximum Entropy framework to guide portfolio optimization.
* **Observation Vector (9D Portfolio Layering Matrix):**
  1. Equity Ratio ($Equity_t / Balance_0$)
  2. Maximum Drawdown Percentage ($\%$ Peak-to-Trough)
  3. Oracle $P(Hold)$ Probability
  4. Oracle $P(Long)$ Probability
  5. Oracle $P(Short)$ Probability
  6. Current 15-Minute Average True Range (ATR)
  7. Current Position Vector ($-1.0$ Short, $0.0$ Cash, $1.0$ Long)
  8. Scaled Unrealized PnL ($PnL / Balance_0$)
  9. Sequential Bars Held Count

---

## ⚡ Engineering Guardrails & Physics

* **Pre-Trade Out-of-Sample Firewall:** Strict sequential index boundaries (Train 70%, Validation 15%, Test 15%) are maintained without shuffling. Scaling bounds (`oracle_scaler.npz`) are derived *exclusively* from the training split and frozen to prevent forward lookahead data leakage.
* **Intra-Candle Ambiguity Resolution:** Environment evaluations process Distance-to-Open metrics (`dist_to_sl` vs `dist_to_tp`) to accurately resolve target interaction order within a single 15-minute candle execution sequence.
* **Temporal Synchronization:** All historical datasets, backtesting layers, and container execution runners are synchronized at the machine layer to UTC timezone parameters.

---

## ⚙️ Deterministic Installation & Tooling

Dependency management is consolidated under PEP 517 build standards using `pyproject.toml` as the single source of truth.

### Prerequisites
* Python 3.10+
* CUDA 12.1+ compatible GPU environment (Optional for local training, recommended)

### Workspace Initialization
```bash
# Clone the repository workspace
git clone [https://github.com/potatocodez127/xau_dl_engine.git](https://github.com/potatocodez127/xau_dl_engine.git)
cd xau_dl_engine

# Install the repository as a local editable module with strict tool locks
pip install --upgrade pip
pip install -e .
```

## 🐳 Containerized Production Deployment

Production deployments enforce immutable layer isolation via a multi-stage Docker architecture to prevent C-extension and compiler bloat.

### Execute Production Environment Container
```bash
# Compile and package the isolated runtime image
docker build -t xau-dl-engine:latest .

# Launch automated testing verification loops
docker run --rm xau-dl-engine:latest -m pytest
```
