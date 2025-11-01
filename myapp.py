import yfinance as yf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st

# Titre de l'application
st.title("Prédiction des Prix des Actions avec LSTM")

# Charger les données historiques depuis Yahoo Finance
st.sidebar.header("Paramètres de l'application")
ticker_symbol = st.sidebar.text_input("Symbol de l'action", value='IBM')
start_date = st.sidebar.date_input("Date de début", pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime('today'))
days_to_predict = st.sidebar.slider("Nombre de jours à prédire", 7, 30, 7)

data = yf.download(ticker_symbol, start=start_date, end=end_date)['Close'].values
data = data.reshape(-1, 1)

# Afficher les données historiques dans un tableau
st.subheader(f"Données historiques pour {ticker_symbol}")
st.line_chart(data)

# Normalisation
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Charger le modèle LSTM pré-entraîné
model = load_model('model.keras')

# Initialisation des séquences avec des données historiques
sequence_length = 50
initial_sequence = scaled_data[-sequence_length:]  # Derniers 50 prix
current_sequence = initial_sequence.reshape(1, sequence_length, 1)

# Fonction pour prédire les prochains jours
def predict_next_week(model, sequence, scaler, days=7):
    predictions = []
    current_seq = sequence.copy()
    for _ in range(days):
        predicted_scaled = model.predict(current_seq, verbose=0)[0, 0]
        predictions.append(predicted_scaled)
        # Ajouter la prédiction à la séquence et faire glisser
        current_seq = np.append(current_seq[0, 1:, :], [[predicted_scaled]], axis=0).reshape(1, -1, 1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Prédire les prochains jours
predicted_week = predict_next_week(model, current_sequence, scaler, days=days_to_predict)

# Statistiques des actions
current_price = data[-1].item()  # Extraire le dernier prix actuel
st.metric(label="Prix actuel", value=f"${current_price:.2f}")
st.metric(label=f"Changement sur {days_to_predict} jours", value=f"${(predicted_week[-1] - current_price):.2f}")

# Afficher les prédictions dans un tableau
st.subheader(f"Prédictions pour les {days_to_predict} prochains jours")
last_date = pd.to_datetime('today')
prediction_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_to_predict + 1)]
prediction_table = pd.DataFrame({
    "Jour": prediction_dates,
    "Prix Prédit ($)": predicted_week
})
st.table(prediction_table)

# Afficher les prédictions sur un graphique
st.subheader(f"Prédictions et historique des prix pour {ticker_symbol}")
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(data.flatten(), label="Prix historiques", color="blue")
ax.plot(range(len(data), len(data) + days_to_predict), predicted_week, label=f"Prédictions ({days_to_predict} jours)", color="red")
ax.set_title(f"Prédiction des prix de {ticker_symbol}")
ax.set_xlabel("Temps (jours)")
ax.set_ylabel("Prix ($)")
ax.legend()

st.pyplot(fig)

# Sauvegarder les prédictions dans un fichier CSV (optionnel)
save_csv = st.sidebar.checkbox("Sauvegarder les prédictions en CSV")
if save_csv:
    prediction_table.to_csv(f"predictions_{ticker_symbol}.csv", index=False)
    st.sidebar.success(f"Fichier CSV des prédictions sauvegardé sous 'predictions_{ticker_symbol}.csv'")
