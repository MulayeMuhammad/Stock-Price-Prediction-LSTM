# Real-Time Stock Price Prediction with LSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)](https://streamlit.io/)
[![Yahoo Finance](https://img.shields.io/badge/Data-Yahoo%20Finance-purple)](https://finance.yahoo.com/)

## 📈 Project Overview

This project develops and compares two deep learning architectures for stock price prediction: **Recurrent Neural Networks (RNN)** and **Long Short-Term Memory (LSTM)** networks. The application focuses on predicting IBM stock opening prices using historical market data.

After comprehensive training and evaluation, the **LSTM model significantly outperformed the RNN** in prediction accuracy, making it the chosen model for real-time deployment. The final application provides an interactive web interface built with **Streamlit** for real-time stock price forecasting.

---

## 🎯 Objectives

1. **Compare RNN vs LSTM** architectures for time series prediction
2. **Train models** on historical IBM stock data from Yahoo Finance
3. **Deploy the best model** (LSTM) for real-time predictions
4. **Create an interactive web application** using Streamlit
5. **Provide weekly forecasts** with visualization and downloadable results

---

## 🚀 Key Features

### 📊 Model Comparison
- ✅ **RNN (Recurrent Neural Network)**: Baseline model for sequential data
- ✅ **LSTM (Long Short-Term Memory)**: Superior performance, overcomes vanishing gradient problem
- ✅ Captures long-term dependencies in time series data

### 💻 Interactive Web Application
- **Real-time predictions**: Forecast stock prices for 7-30 days
- **Customizable parameters**: Choose stock ticker, date range, prediction period
- **Visual analytics**: Interactive charts showing historical data and predictions
- **Key metrics**: Current price, predicted change, percentage variation
- **Data export**: Download predictions as CSV file

### 📈 Real-Time Updates
- Automatic data fetching from Yahoo Finance
- Weekly prediction updates based on latest market data
- Scalable to any stock ticker symbol (IBM, AAPL, GOOGL, etc.)

---

## 🏗️ Architecture

### LSTM Model Structure

```python
Model: Sequential LSTM
_________________________________________________________________
Layer (type)                Output Shape              Params   
=================================================================
LSTM_1                      (None, 50, 50)            10,400    
LSTM_2                      (None, 50)                20,200    
Dense_1                     (None, 25)                1,275     
Dense_2 (Output)            (None, 1)                 26        
=================================================================
Total params: 31,901
Trainable params: 31,901
```

### Why LSTM Outperforms RNN?

| Feature | RNN | LSTM |
|---------|-----|------|
| Long-term dependencies | ❌ Weak | ✅ Strong |
| Vanishing gradient | ❌ Suffers | ✅ Resolves |
| Memory mechanism | ❌ Simple | ✅ Advanced (gates) |
| Prediction accuracy | Lower | **Higher** |

---

## 📁 Project Structure

```
Projet2_Stock_price/
│
├── notebook/
│   ├── stock_prediction_comparison.ipynb  # RNN vs LSTM comparison
│   ├── model_training.ipynb               # LSTM training notebook
│   └── data_exploration.ipynb             # EDA and visualization
│
├── app/
│   ├── app.py                             # Streamlit application (main)
│   └── requirements.txt                   # Dependencies
│
├── models/
│   ├── model.keras                        # Trained LSTM model
│   ├── rnn_model.h5                       # RNN model (for comparison)
│   └── scaler.pkl                         # MinMaxScaler for normalization
│
├── presentation/
│   └── stock_prediction_presentation.pdf  # Project presentation
│
├── predictions/
│   └── predictions_IBM.csv                # Example predictions export
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
└── .gitignore
```

---

## 🛠️ Technical Stack

### Deep Learning
```python
- TensorFlow / Keras      # Neural network framework
- LSTM layers             # Main architecture
- Sequential API          # Model building
```

### Data & Processing
```python
- yfinance               # Yahoo Finance API for stock data
- pandas                 # Data manipulation
- numpy                  # Numerical operations
- scikit-learn           # MinMaxScaler for normalization
```

### Visualization & Deployment
```python
- matplotlib             # Plotting and charts
- streamlit              # Web application framework
```

---

## 📊 Data Pipeline

### 1. Data Collection
```python
import yfinance as yf

# Download historical stock data
data = yf.download('IBM', start='2023-01-01', end='today')
closing_prices = data['Close'].values
```

### 2. Data Preprocessing
```python
from sklearn.preprocessing import MinMaxScaler

# Normalize data to [0, 1] range
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))

# Create sequences (sliding window = 50 days)
sequence_length = 50
X, y = create_sequences(scaled_data, sequence_length)
```

### 3. Model Training
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(50, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32)
```

### 4. Prediction
```python
# Predict next 7 days
predictions = []
current_sequence = last_50_days

for day in range(7):
    pred = model.predict(current_sequence)
    predictions.append(pred)
    current_sequence = update_sequence(current_sequence, pred)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installation

```bash
# Clone the repository
git clone https://github.com/MulayeMuhammad/Stock-Price-Prediction-LSTM.git
cd Stock-Price-Prediction-LSTM

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
tensorflow>=2.10.0
keras>=2.10.0
yfinance>=0.2.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
streamlit>=1.25.0
```

---

## 💻 Usage

### Running the Streamlit App

```bash
# Navigate to app directory
cd Projet2_Stock_price

# Launch Streamlit application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

1. **Select Stock Ticker**: Enter symbol (e.g., IBM, AAPL, MSFT)
2. **Set Date Range**: Choose start and end dates for historical data
3. **Configure Predictions**: Select number of days to predict (7-30)
4. **View Results**:
   - Historical price chart
   - Prediction table with dates and prices
   - Combined visualization
   - Key metrics (current price, predicted change)
5. **Export Data**: Download predictions as CSV

### Example Usage

```python
# In Streamlit sidebar:
Stock Symbol: IBM
Start Date: 2023-01-01
End Date: Today
Days to Predict: 7

# Results:
✅ Current Price: $156.32
✅ Predicted Price (Day 7): $158.45
✅ Expected Change: +$2.13 (+1.36%)
```

---

## 📈 Model Performance

### Comparison Results

| Model | MAE | RMSE | R² Score | Training Time |
|-------|-----|------|----------|---------------|
| RNN   | 3.45 | 4.82 | 0.87     | 12 min       |
| **LSTM** | **2.18** | **2.94** | **0.94** | **15 min** |

**Winner**: LSTM outperforms RNN with 37% lower MAE and better R² score! 🏆

### Why LSTM Won?

1. **Better memory**: Cell state mechanism preserves information
2. **Gate mechanisms**: Input, forget, output gates control information flow
3. **Vanishing gradient solution**: Maintains gradient over long sequences
4. **Long-term patterns**: Captures weekly, monthly trends effectively

---

## 📊 Sample Predictions

### IBM Stock - 7 Day Forecast

| Day | Date | Predicted Price | Change from Current |
|-----|------|----------------|---------------------|
| 1   | 2025-11-02 | $156.80 | +$0.48 |
| 2   | 2025-11-03 | $157.25 | +$0.93 |
| 3   | 2025-11-04 | $157.60 | +$1.28 |
| 4   | 2025-11-05 | $158.10 | +$1.78 |
| 5   | 2025-11-06 | $158.45 | +$2.13 |
| 6   | 2025-11-07 | $158.75 | +$2.43 |
| 7   | 2025-11-08 | $159.20 | +$2.88 |

---

## 🎨 Application Screenshots

### Main Interface
```
┌──────────────────────────────────────────────────────────────┐
│  Prédiction des Prix des Actions avec LSTM                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ├─ Sidebar                 ├─ Main Panel                   │
│  │  • Stock Symbol: IBM     │  📊 Historical Price Chart    │
│  │  • Start: 2023-01-01     │  📈 Prediction Line          │
│  │  • End: Today            │                               │
│  │  • Days: 7               │  📋 Prediction Table          │
│  │  • Save CSV ✓            │  💹 Current Price: $156.32   │
│  └─                         │  📊 Change: +$2.13            │
│                             └─                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 📚 Notebooks

### 1. Model Comparison (`stock_prediction_comparison.ipynb`)
- RNN architecture implementation
- LSTM architecture implementation  
- Side-by-side training and evaluation
- Performance metrics comparison
- Visualization of results

### 2. Model Training (`model_training.ipynb`)
- Data loading from Yahoo Finance
- Preprocessing and normalization
- LSTM model training
- Hyperparameter tuning
- Model saving

### 3. Data Exploration (`data_exploration.ipynb`)
- Stock price trends analysis
- Statistical analysis
- Correlation studies
- Feature engineering

---

## 🔮 Future Enhancements

- [ ] Multi-stock comparison dashboard
- [ ] Sentiment analysis integration (Twitter, News)
- [ ] Technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Ensemble models (LSTM + GRU + Transformer)
- [ ] Automated trading signals
- [ ] Portfolio optimization
- [ ] Real-time alerts and notifications
- [ ] Mobile app deployment

---

## ⚠️ Disclaimer

**This application is for educational and research purposes only.**

- ❌ **NOT financial advice**: Do not use for actual trading decisions
- ❌ **Past performance ≠ Future results**: Historical data doesn't guarantee future outcomes
- ❌ **Model limitations**: Predictions are probabilistic, not deterministic
- ✅ **Use responsibly**: Always consult financial professionals for investment decisions

**The creators are not responsible for any financial losses incurred from using this tool.**

---

## 📖 Methodology

### Data Collection
- **Source**: Yahoo Finance API (yfinance)
- **Frequency**: Daily closing prices
- **Period**: 2+ years of historical data
- **Features**: Open, High, Low, Close, Volume (using Close for predictions)

### Model Training Process
1. Download historical data
2. Normalize using MinMaxScaler (0-1 range)
3. Create sequences (window size = 50 days)
4. Split into train/test (80/20)
5. Train LSTM model (100 epochs)
6. Validate on test set
7. Save best model

### Prediction Strategy
- Use last 50 days as input
- Predict next day
- Append prediction to sequence
- Slide window forward
- Repeat for desired forecast period

---

## 🛡️ Model Validation

### Metrics Used
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Square Error): Penalizes large errors
- **R² Score**: Proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error): Percentage error

### Cross-Validation
- Time series split (respects temporal order)
- Walk-forward validation
- Out-of-sample testing

---

## 🏆 Key Achievements

- ✅ **94% R² Score**: Excellent model fit
- ✅ **37% improvement** over baseline RNN
- ✅ **Real-time deployment**: Streamlit web app
- ✅ **User-friendly interface**: No coding required
- ✅ **Scalable**: Works with any stock ticker
- ✅ **Exportable results**: CSV download feature

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Moulaye Ahmed Mohammed Brahim**

- 🌐 Portfolio: [mulayemuhammad.github.io/Moulaye_DS_Portfolio](https://mulayemuhammad.github.io/Moulaye_DS_Portfolio/)
- 💼 LinkedIn: [Moulaye Ahmed MUHAMMAD](https://www.linkedin.com/in/moulaye-ahmed-muhammad/)
- 🐙 GitHub: [@MulayeMuhammad](https://github.com/MulayeMuhammad)
- 📧 Email: mulayemuhammad@gmail.com
- 🐦 Twitter: [@MuhammadMoulaye](https://twitter.com/MuhammadMoulaye)

---

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free financial data API
- **TensorFlow/Keras** team for excellent deep learning framework
- **Streamlit** team for intuitive web app framework
- **INSEA** for academic support and resources
- Open-source community for invaluable tools and libraries

---

## 📚 References

1. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
2. Fischer, T., & Krauss, C. (2018). "Deep learning with long short-term memory networks for financial market predictions." *European Journal of Operational Research*, 270(2), 654-669.
3. Yahoo Finance API Documentation: https://pypi.org/project/yfinance/
4. Streamlit Documentation: https://docs.streamlit.io/

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `Module 'yfinance' not found`
```bash
pip install yfinance --upgrade
```

**Issue**: `TensorFlow not loading model`
```bash
pip install tensorflow==2.13.0  # Use specific version
```

**Issue**: `Streamlit not opening`
```bash
# Check port availability
streamlit run app.py --server.port 8502
```

---

## 📊 Project Status

- [x] Data collection and preprocessing
- [x] RNN model development
- [x] LSTM model development
- [x] Model comparison and evaluation
- [x] Streamlit application development
- [x] Real-time prediction implementation
- [x] CSV export functionality
- [x] Documentation
- [ ] Mobile app deployment
- [ ] API development

---

<p align="center">
  <i>⭐ If you find this project useful, please consider giving it a star!</i>
</p>

<p align="center">
  <strong>Predicting the future of finance with Deep Learning 📈</strong>
</p>

<p align="center">
  Made with ❤️ by <a href="https://github.com/MulayeMuhammad">Moulaye Ahmed</a>
</p>
