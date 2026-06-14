import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def min_max_scaler(X_train, X_test):
    """Normalisasi Min-Max berdasarkan data train."""
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1          # hindari pembagian nol
    X_train_scaled = (X_train - X_min) / denom
    X_test_scaled  = (X_test  - X_min) / denom
    return X_train_scaled, X_test_scaled


def chebyshev_distance(a, b):
    """Jarak Chebyshev: max(|a_i - b_i|) untuk semua dimensi i."""
    return np.max(np.abs(a - b))


def knn_predict(X_train, y_train, X_query, k):
    """
    Prediksi nilai target untuk satu titik query menggunakan KNN.
    Kembalikan rata-rata y dari K tetangga terdekat (jarak Chebyshev).
    """
    n_train = len(X_train)
    distances = np.zeros(n_train)
    for i in range(n_train):
        distances[i] = chebyshev_distance(X_train[i], X_query)

    # Ambil indeks K tetangga dengan jarak terkecil
    k_indices = np.argsort(distances)[:k]
    return np.mean(y_train[k_indices])


def knn_predict_all(X_train, y_train, X_test, k):
    """Prediksi untuk seluruh data test."""
    predictions = np.zeros(len(X_test))
    for i in range(len(X_test)):
        predictions[i] = knn_predict(X_train, y_train, X_test[i], k)
    return predictions


def hitung_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def hitung_rmse(y_true, y_pred):
    return np.sqrt(hitung_mse(y_true, y_pred))

def hitung_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def hitung_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def hitung_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def cross_val_mse(X, y, k, n_folds=5):
    """
    K-Fold Cross Validation manual untuk memilih K optimal.
    Kembalikan rata-rata MSE lintas fold.
    """
    n = len(X)
    fold_size = n // n_folds
    mse_list  = []

    for f in range(n_folds):
        val_start = f * fold_size
        val_end   = val_start + fold_size if f < n_folds - 1 else n

        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_tr  = np.concatenate([X[:val_start], X[val_end:]], axis=0)
        y_tr  = np.concatenate([y[:val_start], y[val_end:]], axis=0)

        if len(X_tr) == 0 or len(X_val) == 0:
            continue

        y_pred_val = knn_predict_all(X_tr, y_tr, X_val, k)
        mse_list.append(hitung_mse(y_val, y_pred_val))

    return np.mean(mse_list)

df = pd.read_excel(
    "/Users/halim/Desktop/python/buellypt2/Dataset Remidi UTS KDKA Prodi AI Unesa.xlsx",
    sheet_name="Sheet1"
)
df = df[pd.to_numeric(df["Tahun"], errors="coerce").notna()].copy()
df["Tahun"] = df["Tahun"].astype(int)
df = df.sort_values("Tahun").reset_index(drop=True)

print("=" * 62)
print("  PREDIKSI NILAI TUKAR IDR/USD – KNN CHEBYSHEV (FROM SCRATCH)")
print("=" * 62)
print(f"\n[INFO] Jumlah data total  : {len(df)} baris")
print(f"[INFO] Rentang tahun      : {df['Tahun'].min()} – {df['Tahun'].max()}")

fitur_kolom = [
    "Tahun",
    "FDI Net Inflows (US$)",
    "Inflasi (%)",
    "GDP Growth (%)",
    "GDP (Current US$)",
    "Current Account Balance (US$)",
    "Trade (% GDP)",
    "Exports (US$)",
    "Imports (US$)",
    "Total Reserves (includes gold, current US$)",
    "Broad Money (% of GDP)"
]
target_kolom = "Kurs Resmi (IDR per USD)"

X = df[fitur_kolom].values.astype(float)
y = df[target_kolom].values.astype(float)

n_total = len(X)
n_train = int(np.floor(n_total * 0.80))
n_test  = n_total - n_train

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
tahun_train = df["Tahun"].values[:n_train]
tahun_test  = df["Tahun"].values[n_train:]

print(f"\n[SPLIT] Total data : {n_total}")
print(f"        Train      : {n_train} baris (tahun {tahun_train[0]}–{tahun_train[-1]})")
print(f"        Test       : {n_test}  baris (tahun {tahun_test[0]}–{tahun_test[-1]})")

X_train_s, X_test_s = min_max_scaler(X_train, X_test)

print("\n[PROSES] Cross-validation manual mencari K optimal (K = 3–10)...")
k_values = list(range(3, 11))
cv_mse   = []

for k in k_values:
    mse_k = cross_val_mse(X_train_s, y_train, k, n_folds=5)
    cv_mse.append(mse_k)
    print(f"          K={k}  MSE-CV = {mse_k:>15,.2f}")

k_optimal = k_values[int(np.argmin(cv_mse))]
print(f"\n        K optimal : {k_optimal}  (MSE CV = {cv_mse[k_values.index(k_optimal)]:,.2f})")

print(f"\n[MODEL] Prediksi KNN Chebyshev dengan K = {k_optimal}...")
y_pred_train = knn_predict_all(X_train_s, y_train, X_train_s, k_optimal)
y_pred_test  = knn_predict_all(X_train_s, y_train, X_test_s,  k_optimal)

mse  = hitung_mse(y_test,  y_pred_test)
rmse = hitung_rmse(y_test, y_pred_test)
mae  = hitung_mae(y_test,  y_pred_test)
r2   = hitung_r2(y_test,   y_pred_test)
mape = hitung_mape(y_test, y_pred_test)

print("\n" + "=" * 62)
print("  HASIL EVALUASI MODEL (DATA TEST)")
print("=" * 62)
print(f"  Algoritma        : KNN From Scratch")
print(f"  Metrik Jarak     : Chebyshev (max|a_i - b_i|)")
print(f"  K Tetangga       : {k_optimal}")
print(f"  Jumlah Fitur     : {len(fitur_kolom)}")
print(f"  MSE  (Mean Squared Error)       : {mse:>18,.4f}")
print(f"  RMSE (Root Mean Squared Error)  : {rmse:>15,.4f}  IDR")
print(f"  MAE  (Mean Absolute Error)      : {mae:>15,.4f}  IDR")
print(f"  R²   (Koefisien Determinasi)    : {r2:>18,.4f}")
print(f"  Error Rate / MAPE               : {mape:>14,.4f} %")
print("=" * 62)

print("\n[DETAIL] Perbandingan Nilai Aktual vs Prediksi (Data Test):")
print(f"  {'Tahun':<8} {'Aktual (IDR)':>14} {'Prediksi (IDR)':>15} {'Error (IDR)':>13} {'Error (%)':>10}")
print("  " + "-" * 62)
for i in range(n_test):
    err     = y_test[i] - y_pred_test[i]
    err_pct = abs(err / y_test[i]) * 100
    print(f"  {tahun_test[i]:<8} {y_test[i]:>14,.2f} {y_pred_test[i]:>15,.2f} {err:>+13,.2f} {err_pct:>9.2f}%")

fig = plt.figure(figsize=(16, 12), facecolor="#0f1117")
fig.suptitle(
    "Prediksi Nilai Tukar IDR/USD – KNN Chebyshev (From Scratch)\n"
    "Dataset: World Bank Indonesia (1990–2024)  |  Split: 80% Train / 20% Test",
    fontsize=13, fontweight="bold", color="white", y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

def style_ax(ax):
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(alpha=0.13, color="gray")

ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1)
ax1.plot(df["Tahun"], y, "o-", color="#4fc3f7", lw=2, ms=5, label="Aktual")
ax1.plot(tahun_train, y_pred_train, "s--", color="#aed581", lw=1.8, ms=5,
         alpha=0.85, label=f"Prediksi Train ({tahun_train[0]}–{tahun_train[-1]})")
ax1.plot(tahun_test, y_pred_test, "^-", color="#ff8a65", lw=2.2, ms=7,
         zorder=5, label=f"Prediksi Test ({tahun_test[0]}–{tahun_test[-1]})")
ax1.axvline(tahun_test[0] - 0.5, color="#ffeb3b", ls=":", lw=1.5, alpha=0.7)
ax1.text(tahun_test[0] - 0.3, ax1.get_ylim()[0] + 200,
         "← Train | Test →", color="#ffeb3b", fontsize=8, va="bottom")
ax1.set_title("Nilai Tukar IDR/USD: Aktual vs Prediksi (Seluruh Periode)",
              color="white", fontsize=11, pad=8)
ax1.set_xlabel("Tahun", color="white")
ax1.set_ylabel("Kurs (IDR/USD)", color="white")
ax1.legend(facecolor="#2a2d3a", labelcolor="white", fontsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2)
bw, xi = 0.35, np.arange(n_test)
ax2.bar(xi - bw/2, y_test,      bw, label="Aktual",   color="#4fc3f7", alpha=0.85)
ax2.bar(xi + bw/2, y_pred_test, bw, label="Prediksi", color="#ff8a65", alpha=0.85)
ax2.set_xticks(xi)
ax2.set_xticklabels([str(t) for t in tahun_test], rotation=30, color="white", fontsize=8)
ax2.set_title("Aktual vs Prediksi – Data Test", color="white", fontsize=10)
ax2.set_ylabel("Kurs (IDR/USD)", color="white")
ax2.legend(facecolor="#2a2d3a", labelcolor="white", fontsize=9)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3)
col_k = ["#ff8a65" if k == k_optimal else "#4fc3f7" for k in k_values]
ax3.bar(k_values, cv_mse, color=col_k, alpha=0.85, edgecolor="#555")
ax3.set_xticks(k_values)
ax3.set_title(f"MSE Cross-Validation per Nilai K\n(K optimal = {k_optimal}, ditandai jingga)",
              color="white", fontsize=10)
ax3.set_xlabel("Nilai K", color="white")
ax3.set_ylabel("MSE (CV)", color="white")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

summary_txt = (
    f"MSE  : {mse:>12,.2f}\n"
    f"RMSE : {rmse:>10,.2f} IDR\n"
    f"MAE  : {mae:>10,.2f} IDR\n"
    f"R²   : {r2:>14.4f}\n"
    f"MAPE : {mape:>12.2f} %"
)
fig.text(0.995, 0.01, summary_txt, ha="right", va="bottom", fontsize=9,
         color="white", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#2a2d3a",
                   edgecolor="#4fc3f7", alpha=0.92))

plt.savefig("/Users/halim/Desktop/python/buellypt2/hasil_prediksi_kurs_knn_chebyshev.png",
            dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()

print("\n[OUTPUT] Grafik disimpan.")
print("[SELESAI] Program berjalan sukses.\n")
