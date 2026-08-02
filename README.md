# EEG-Based Depression Detection using Spiking Neural Networks

> Classifies depression vs. healthy individuals from EEG signals using Spiking Neural Networks (SNN) with patient-adaptive fine-tuning.

---



## 🏗️ Architecture

```
EEG Signals (raw)
      │
      ▼
┌─────────────────┐
│  Preprocessing  │  Bandpass filter (1–40 Hz) + Z-score normalization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feature Extraction│  Time-domain + Frequency-domain (Delta/Theta/Alpha/Beta)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌──────────────────────┐
│  RF   │  │    Spike Encoding    │
│Baseline  │  (Rate Coding)       │
└───────┘  └──────────┬───────────┘
                      │
                      ▼
               ┌───────────────┐
               │  SNN Model    │
               │  FC → LIF1    │
               │  FC → LIF2    │
               │  FC → Output  │
               └───────┬───────┘
                       │
                       ▼
               ┌───────────────┐
               │Patient Adapt  │  Fine-tune 2–3 epochs
               └───────┬───────┘
                       │
                       ▼
               ┌───────────────┐
               │  Evaluation   │  CM, ROC, Comparison
               └───────────────┘
```

---

## 📁 Project Structure

```
eeg-depression-snn/
│
├── data/
│   ├── raw/                   # Raw EEG CSVs
│   └── processed/             # Preprocessed arrays
│
├── notebooks/
│   └── exploration.ipynb      # EDA 
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading & synthetic generation
│   ├── preprocessing.py       # Bandpass filter, normalization, windowing
│   ├── feature_extraction.py  # Time + frequency domain features
│   ├── spike_encoding.py      # Rate coding encoder
│   ├── snn_model.py           # SNN architecture (snnTorch + PyTorch)
│   ├── baseline_model.py      # Random Forest / Logistic Regression
│   ├── train.py               # SNN training & patient adaptation
│   ├── evaluate.py            # Metrics & all visualizations
│   └── utils.py               # Utilities 
│
├── models/
│   ├── random_forest.joblib   # Trained RF model
│   ├── snn_model.pt           # Trained SNN weights
│   └── snn_adapted.pt         # Patient-adapted SNN weights
│
├── results/
│
├── requirements.txt
├── README.md
└── main.py                   
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11+ | Core language |
| NumPy / Pandas | Data processing |
| SciPy | Signal filtering (Butterworth) |
| Scikit-learn | Baseline models, metrics |
| PyTorch | Deep learning backend |
| snnTorch | Spiking Neural Network layers |
| Matplotlib / Seaborn | Visualizations |
| Jupyter | Exploratory notebook |

---

## 📊 Dataset

The project auto-generates a synthetic EEG dataset if no real data is provided. The synthetic generator mimics known EEG depression biomarkers:

- **Depressed** patients: elevated delta/theta power, suppressed alpha
- **Healthy** patients: balanced frequency profile with dominant alpha

To use a real dataset:
1. Place your CSV in `data/raw/`
2. Ensure columns: `ch_0 … ch_N`, `label` (0=healthy, 1=depressed), optional `subject_id`
3. Run with `--no-synthetic --data-path data/raw/your_file.csv`

---

## 🚀 How to Run

### 1. Create a virtual environment

**Windows (PowerShell)**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python main.py
```

### 4. Run with custom options
```bash
python main.py --n-subjects 80 --epochs 30 --T 50 --seed 42
python main.py --no-synthetic --data-path data/raw/real_eeg.csv
```

### 5. Deactivate the virtual environment 
```bash
deactivate
```

---

## 🧪 Feature Extraction

**Time-domain** (per channel):
- Mean, Standard Deviation, Variance
- Skewness, Kurtosis, Peak-to-peak amplitude

**Frequency-domain** (per channel):
- Delta power (1–4 Hz)
- Theta power (4–8 Hz)
- Alpha power (8–13 Hz)
- Beta power  (13–30 Hz)
- Relative band powers (ratio to total)

---

## ⚡ Spike Encoding (Rate Coding)

Features are min-max scaled to `[0, 1]`, then each value is treated as a **Bernoulli firing probability** over `T` discrete time steps. This produces binary spike trains `(T × N × F)` as input to the SNN.

---

## 🧠 SNN Architecture

```
Input (n_features)  →  FC(128)  →  LIF1  →  FC(64)  →  LIF2  →  FC(2)
                                   Leaky Integrate-and-Fire neurons
                                   with surrogate gradient (fast sigmoid)
```

- Trained with **cross-entropy loss** on summed membrane potentials
- **Patient adaptation**: 3-epoch fine-tuning at lower LR (5e-4)

---




## 🔮 Future Improvements (Real-World Deployment)

While the current pipeline uses a highly separable synthetic dataset (often yielding ~100% accuracy), deploying this on real-world clinical data (like the [MODMA dataset](http://modma.lzu.edu.cn/data/index/)) will introduce significant noise (e.g., eye blinks, muscle artifacts). 

To adapt this project for a robust real-world scenario, consider the following upgrades:
1. **K-Fold Cross-Validation**: Replace the single `train_test_split` with 5-fold or 10-fold cross-validation to ensure the model's performance is stable across different patient subsets.
2. **Hyperparameter Tuning**: Use a library like [Optuna](https://optuna.org/) or [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to dynamically search for the optimal SNN parameters (learning rate, batch size, LIF `beta` decay).
3. **Advanced Artifact Removal**: Implement Independent Component Analysis (ICA) in the preprocessing step to remove physiological noise from real EEG signals before feature extraction.

---

## 👨‍💻 Coding Standards

- Modular functions with docstrings
- Type hints throughout
- Reproducible random seeds (`--seed 42`)
- Error handling and logging
- Runs on CPU (no GPU required)

---

## 📄 License

MIT License - free to use and modify for academic and portfolio purposes.
