import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

hisse = "THYAO.IS" 
print(f"{hisse} verileri çekiliyor, lütfen bekleyin...")
data = yf.download(hisse, start="2025-02-10", end="2026-02-10")

data['Gun'] = np.arange(len(data))
X = data[['Gun']].values
y = data['Close'].values

model = LinearRegression()
model.fit(X, y) # AI burada geçmiş veriden trendi öğreniyor

tahmin_gunu = np.array([[len(data)]])
gelecek_fiyat = model.predict(tahmin_gunu)

print("-" * 30)
print(f"Yapay zekaya göre bir sonraki tahmin edilen fiyat: {gelecek_fiyat.item():.2f} TL")
print("-" * 30)


plt.figure(figsize=(10,5))
plt.plot(X, y, color='blue', label='Gerçek Veri')
plt.plot(X, model.predict(X), color='red', label='AI Trend Çizgisi')
plt.title(f"{hisse} Hisse Senedi AI Trend Analizi")
plt.xlabel("Gün Sayısı")
plt.ylabel("Fiyat (TL)")
plt.legend()
plt.grid(True)
plt.show()
