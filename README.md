# **Explainable AI for Smoking Behaviour Detection: A Danish Medical Records Study**

This project implements a modular and reproducible pipeline for smoking status and pack-year classification from Danish clinical notes using a CNN-LSTM architecture with LIME-based interpretability.

---

## 🧠 What the Project Does

The methodology of this study builds upon our previous work [4], aiming to validate and extend the best-performing model. Specifically, we classify patients into:

- **Smoking categories**:  
  `Never Smoker`, `Current Smoker`, `Former Smoker`, `E-Cigarette`, `Passive Smoker`, `Unknown`
  
- **Pack-per-year categories**:  
  `Low`, `Medium`, or `High`

The model is trained using data from `pop1_merged.csv` and evaluated on `pop2_merged.csv`. It uses a hybrid **Conv1D + LSTM** architecture for sequence classification and LIME for explainable AI.

---

## 📦 Structure

```
.
├── data/                         # Contains pop1_merged.csv and pop2_merged.csv
├── src/
│   ├── config.py                 # Set task and model type
│   ├── data_loader.py           # Preprocessing and tokenization
│   ├── models.py                # CNN-LSTM model
│   ├── train.py                 # Training and evaluation logic
│   └── explain.py               # LIME explanation with save as PNG/HTML
├── run_experiments.ipynb        # Unified experiment notebook
└── requirements.txt             # Dependencies
```

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your data:
```
project-root/
└── data/
    ├── pop1_merged.csv
    └── pop2_merged.csv
```

3. Launch the notebook:
```bash
jupyter notebook run_experiments.ipynb
```

4. To use LIME:
```python
from src.explain import lime_explanation
lime_explanation(model, X_test, sentence_index=0)
```

---

## 📊 Results & Interpretability

- Classification report and confusion matrix are generated after evaluation
- LIME explanations are visualized in the notebook
- Exported to both `.html` and `.png` for easy sharing

---

## 📄 Citation

If you use this code or adapt it for your own research, please cite:

> Ebrahimi, A., Wiil, U.K. (2024).  
> Deep learning architectures for identifying smoking status and pack-per-year from clinical notes using Danish electronic health records.  
> *BMC Medical Research Methodology* 24, 231.  
> [https://doi.org/10.1186/s12874-024-02231-4](https://doi.org/10.1186/s12874-024-02231-4)

---

## 🔖 License

This project is licensed under the MIT License.
