# Matrix Estimation Project Documentation

## Overview

This project predicts the **next quarter's (63 trading days) variance-covariance matrix** for a set of sector ETFs using a deep learning model (CNN-LSTM). The workflow includes data collection, preprocessing, model training, hyperparameter tuning, evaluation, and visualization.

---

## What the Code Does

### 1. **Data Collection & Preparation**
- **Tickers:** Uses 11 SPDR sector ETFs.
- **Data Download:** Fetches adjusted close prices for training (2018-07-01 to 2021-12-31) and testing (2022-01-01 to 2024-12-31) periods using `yfinance`.
- **Covariance Matrices:** Computes rolling 63-day covariance matrices of log returns for each ETF pair, resulting in a sequence of 11x11 matrices.
- **Prediction Horizon:** The model is trained to predict the covariance matrix 63 days (one quarter) ahead.

### 2. **Data Normalization & Sequencing**
- **Normalization:** Each covariance matrix is divided by its maximum value for numerical stability.
- **Sequencing:** For each sample, a sequence of 5 consecutive normalized matrices is used as input to predict the next matrix.

### 3. **Model Architecture**
- **CNN-LSTM Model:**
  - **Input:** Sequence of 5 matrices (shape: 5, 11, 11, 1).
  - **CNN Layers:** Extract spatial features from each matrix.
  - **LSTM Layer:** Learns temporal dependencies across the sequence.
  - **Dense + Reshape:** Outputs a single predicted 11x11 matrix.
- **Loss Function:** Custom Euclidean distance loss between predicted and true matrices.

### 4. **Training & Evaluation**
- **Training:** Model is trained on the training set for 30 epochs.
- **Evaluation:** Evaluates performance on both training and test data, denormalizes predictions, and visualizes results with heatmaps.

### 5. **Hyperparameter Tuning**
- **Grid Search:** Explores combinations of CNN filters, LSTM units, optimizers, learning rates, and batch sizes to find the best configuration.

### 6. **Visualization**
- **Heatmaps:** Plots true and predicted covariance matrices for qualitative assessment.

---

## How to Run the Code

### **Requirements**
- Python 3.8+
- Jupyter Notebook
- Packages: `yfinance`, `numpy`, `pandas`, `matplotlib`, `tensorflow`, `scikit-learn`

Install requirements (if needed):
```bash
pip install yfinance numpy pandas matplotlib tensorflow scikit-learn
```

### **Steps**

1. **Open the Notebook**
   - Open `matrix_estimation.ipynb` in Jupyter Notebook or VS Code.

2. **Run All Cells**
   - Execute each cell in order. The notebook will:
     - Download and preprocess data
     - Visualize sample matrices
     - Build and train the CNN-LSTM model
     - Perform hyperparameter tuning (optional, can be time-consuming)
     - Retrain the best model
     - Evaluate and visualize predictions

3. **Saving the Best Model**
   - After training the best model, you can save it for future use:
     ```python
     best_model.save("best_cnn_lstm_model.h5")
     ```
   - To load the model later:
     ```python
     from tensorflow.keras.models import load_model
     model = load_model("best_cnn_lstm_model.h5", compile=False)
     ```

4. **Predict on New Data**
   - Use the trained model to predict future covariance matrices by preparing new data in the same format and calling `model.predict()`.

---

## File Structure

```
matrix-estimation/
│
├── matrix_estimation.ipynb      # Main notebook (run this)
├── best_cnn_lstm_model.h5       # (Optional) Saved trained model
├── requirements.txt             # (Optional) List of dependencies
└── README.md                    # This documentation
```

---

## Notes

- **Hyperparameter Tuning:** The grid search section can be slow. You can skip or limit the grid for faster runs.
- **Visualization:** The notebook includes functions to plot and compare true vs. predicted covariance matrices.
- **Customization:** You can adjust tickers, date ranges, and model parameters as needed.

---

## Contact

For questions or issues, please contact the project maintainer.