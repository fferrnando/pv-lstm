# ==============================================================================
# PROGRAM MODEL A - DATA DENGAN NILAI 0 (PEMBANDING)
# Judul: Desain dan Implementasi Long Short-Term Memory (LSTM) pada Sistem
# Monitoring dan Data Logger Photovoltaic (PV) untuk Prediksi Keluaran Daya
# Terhadap Fluktuasi Cahaya
#
# CATATAN:
# Program ini MENYERTAKAN data nilai 0 (kondisi PWM OFF / SCC memotong arus)
# Digunakan sebagai MODEL PEMBANDING terhadap Model B (tanpa nilai 0)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from google.colab import drive
import os
import warnings
import tensorflow as tf
import random

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings('ignore')

# ==============================================================================
# SETUP GOOGLE DRIVE
# ==============================================================================
print("="*60)
print("  MODEL A - DATA DENGAN NILAI 0 (PEMBANDING)")
print("="*60)

drive.mount('/content/drive')

folder        = '/content/drive/MyDrive/Data Final PLTS/'
folder_data   = folder + 'Data_CSV/'
folder_model  = folder + 'Model/'
folder_hasil  = folder + 'Hasil/'
folder_grafik = folder + 'Grafik/'

for f in [folder, folder_data, folder_model, folder_hasil, folder_grafik]:
    os.makedirs(f, exist_ok=True)

print("✅ Google Drive terhubung!")

# ==============================================================================
# TAHAP 1: MEMBACA & MENGGABUNGKAN 5 FILE CSV
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 1: MEMBACA DATA CSV")
print("="*60)

df_light = pd.read_csv(folder_data + 'Grafik_Intesitas Cahaya.csv', sep=';')
df_volt  = pd.read_csv(folder_data + 'Grafik_Voltage.csv', sep=';')
df_temp  = pd.read_csv(folder_data + 'Grafik_Suhu.csv', sep=';')
df_curr  = pd.read_csv(folder_data + 'Grafik_Arus.csv', sep=';')
df_power = pd.read_csv(folder_data + 'Grafik_Daya.csv', sep=';')

for d in [df_light, df_volt, df_temp, df_curr, df_power]:
    d['Timestamp'] = pd.to_datetime(d['Timestamp'])

df = pd.merge(df_light, df_volt, on='Timestamp', how='inner')
df = pd.merge(df, df_temp, on='Timestamp', how='inner')
df = pd.merge(df, df_curr, on='Timestamp', how='inner')
df = pd.merge(df, df_power, on='Timestamp', how='inner')

df = df.sort_values('Timestamp').reset_index(drop=True)
total_hari = len(df['Timestamp'].dt.date.unique())

print(f"✅ Data berhasil digabungkan!")
print(f"   Total data  : {len(df)} sampel")
print(f"   Total hari  : {total_hari} Hari")
print(f"   Kolom       : {df.columns.tolist()}")
print(f"   Mulai       : {df['Timestamp'].min()}")
print(f"   Selesai     : {df['Timestamp'].max()}")

# ==============================================================================
# TAHAP 2: CLEANING DATA (MENYERTAKAN NILAI 0)
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 2: CLEANING DATA (TERMASUK NILAI 0)")
print("="*60)

data_awal = len(df)

# Model A: menyertakan nilai 0 - hanya buang outlier ekstrem
df = df[
    (df['Intesitas Cahaya'] >= 0)   & (df['Intesitas Cahaya'] <= 150000) &
    (df['Voltage']          >= 10)  & (df['Voltage']          <= 20)     &
    (df['Current']          >= 0)   & (df['Current']          <= 5)      &
    (df['Power']            >= 0)   & (df['Power']            <= 100)    &
    (df['Temperature']      >= 15)  & (df['Temperature']      <= 85)
].reset_index(drop=True)

print(f"   Data awal   : {data_awal} sampel")
print(f"   Data bersih : {len(df)} sampel (termasuk nilai 0)")
print(f"   Dihapus     : {data_awal - len(df)} sampel (outlier ekstrem saja)")
print(f"   ✅ Data nilai 0 (PWM OFF) DIPERTAHANKAN untuk analisis pembanding")

# Cek distribusi nilai 0
pwm_off = df[df['Current'] <= 0.1]
pwm_on  = df[df['Current'] > 0.1]
pct_off = len(pwm_off) / len(df) * 100
pct_on  = len(pwm_on)  / len(df) * 100
print(f"\n   Distribusi data:")
print(f"   PWM ON  (arus > 0.1A) : {len(pwm_on):,} sampel ({pct_on:.1f}%)")
print(f"   PWM OFF (arus ≤ 0.1A) : {len(pwm_off):,} sampel ({pct_off:.1f}%)")

# ==============================================================================
# TAHAP 3: STATISTIK DESKRIPTIF
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 3: STATISTIK DESKRIPTIF")
print("="*60)

kolom_params = {
    'Intesitas Cahaya' : 'lux',
    'Voltage'          : 'V',
    'Current'          : 'A',
    'Power'            : 'W',
    'Temperature'      : '°C'
}

statistik_list = []
for kolom, satuan in kolom_params.items():
    if kolom in df.columns:
        stat = {
            'Parameter' : kolom,
            'Satuan'    : satuan,
            'Min'       : df[kolom].min(),
            'Max'       : df[kolom].max(),
            'Mean'      : df[kolom].mean(),
            'Median'    : df[kolom].median(),
            'Std'       : df[kolom].std()
        }
        statistik_list.append(stat)
        print(f"\n{kolom} ({satuan}):")
        print(f"   Min     : {stat['Min']:.4f}")
        print(f"   Max     : {stat['Max']:.4f}")
        print(f"   Mean    : {stat['Mean']:.4f}")
        print(f"   Median  : {stat['Median']:.4f}")
        print(f"   Std Dev : {stat['Std']:.4f}")

df_statistik = pd.DataFrame(statistik_list)
df_statistik.to_csv(folder_hasil + 'statistik_model_A.csv', index=False)
print("\n✅ Statistik deskriptif tersimpan!")

# ==============================================================================
# TAHAP 4: GRAFIK TIME SERIES SEMUA PARAMETER
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 4: GRAFIK TIME SERIES")
print("="*60)

fig, axes = plt.subplots(5, 1, figsize=(16, 20))
fig.suptitle('Grafik Time Series Seluruh Parameter (Model A - Dengan Nilai 0)\nSistem Monitoring PLTS 50WP',
             fontsize=14, fontweight='bold', y=0.98)

params_plot = [
    ('Intesitas Cahaya', 'Intensitas Cahaya (lux)', '#FF9800'),
    ('Voltage',          'Tegangan (V)',            '#2196F3'),
    ('Current',          'Arus (A)',                '#4CAF50'),
    ('Power',            'Daya (W)',                '#F44336'),
    ('Temperature',      'Suhu Panel (°C)',         '#9C27B0'),
]

for ax, (kolom, label, warna) in zip(axes, params_plot):
    ax.plot(df['Timestamp'], df[kolom], color=warna, linewidth=0.6, alpha=0.8, label=label)
    ax.axhline(df[kolom].mean(), color='black', linestyle='--', linewidth=1.2,
               label=f"Rata-rata: {df[kolom].mean():.3f}")
    ax.fill_between(df['Timestamp'], df[kolom], alpha=0.1, color=warna)
    ax.set_ylabel(label, fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Waktu', fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(folder_grafik + 'model_A_time_series.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Grafik time series tersimpan!")

# ==============================================================================
# TAHAP 5: ANALISIS KORELASI PEARSON
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 5: ANALISIS KORELASI PEARSON")
print("="*60)

def interpretasi_korelasi(r):
    r_abs = abs(r)
    if r_abs >= 0.8:   return "Sangat Kuat"
    elif r_abs >= 0.6: return "Kuat"
    elif r_abs >= 0.4: return "Sedang"
    else:              return "Lemah"

print(f"\n{'Parameter':<20} {'r':>8} {'p-value':>12} {'Interpretasi'}")
print("-"*60)

korelasi_list = []
params_korelasi = ['Intesitas Cahaya', 'Voltage', 'Current', 'Temperature']

for kolom in params_korelasi:
    if kolom in df.columns:
        r, p = stats.pearsonr(df[kolom], df['Power'])
        interp = interpretasi_korelasi(r)
        sig    = "Signifikan ✅" if p < 0.05 else "Tidak Signifikan ⚠️"
        p_str  = f"{p:.2e}" if p < 0.0001 else f"{p:.6f}"
        print(f"{kolom:<20} {r:>8.4f} {p_str:>12} {interp}")
        korelasi_list.append({
            'Parameter'    : kolom,
            'r_pearson'    : round(r, 4),
            'p_value'      : p_str,
            'Interpretasi' : interp,
            'Signifikansi' : sig
        })

df_korelasi = pd.DataFrame(korelasi_list)
df_korelasi.to_csv(folder_hasil + 'korelasi_model_A.csv', index=False)
print("\n✅ Hasil korelasi tersimpan!")

# ==============================================================================
# TAHAP 6: HEATMAP KORELASI
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 6: HEATMAP KORELASI")
print("="*60)

kolom_heatmap = ['Intesitas Cahaya', 'Voltage', 'Current', 'Temperature', 'Power']
kolom_ada     = [k for k in kolom_heatmap if k in df.columns]
corr_matrix   = df[kolom_ada].corr()

plt.figure(figsize=(9, 7))
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask, k=1)] = True

sns.heatmap(
    corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, square=True,
    mask=mask, linewidths=0.8, annot_kws={'size': 11, 'weight': 'bold'}, vmin=-1, vmax=1
)
plt.title('Heatmap Korelasi Antar Parameter (Model A - Dengan Nilai 0)\nSistem Monitoring PLTS 50WP',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(folder_grafik + 'model_A_heatmap_korelasi.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Heatmap korelasi tersimpan!")

# ==============================================================================
# TAHAP 7: SCATTER PLOT KORELASI
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 7: SCATTER PLOT KORELASI")
print("="*60)

scatter_params = [
    ('Intesitas Cahaya', 'Intensitas Cahaya (lux)', '#FF9800'),
    ('Temperature',      'Suhu Panel (°C)',         '#9C27B0'),
    ('Voltage',          'Tegangan (V)',            '#2196F3'),
    ('Current',          'Arus (A)',                '#4CAF50'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Scatter Plot Korelasi Parameter vs Daya Output Panel Surya\n(Model A - Dengan Nilai 0)',
             fontsize=13, fontweight='bold')

for ax, (kolom, label, warna) in zip(axes.flatten(), scatter_params):
    if kolom in df.columns:
        r, p = stats.pearsonr(df[kolom], df['Power'])
        ax.scatter(df[kolom], df['Power'], alpha=0.15, color=warna, s=8, label='Data')
        z     = np.polyfit(df[kolom], df['Power'], 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(df[kolom].min(), df[kolom].max(), 100)
        ax.plot(x_line, p_fit(x_line), color='black', linewidth=2,
                linestyle='--', label='Tren Linear')
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel('Daya (W)', fontsize=9)
        ax.set_title(f'{kolom} vs Daya\nr = {r:.4f} | {interpretasi_korelasi(r)}',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(folder_grafik + 'model_A_scatter_korelasi.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Scatter plot tersimpan!")

# ==============================================================================
# TAHAP 8: ANALISIS PWM SCC
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 8: ANALISIS KARAKTERISTIK SCC PWM")
print("="*60)

PWM_THRESHOLD = 0.1
pwm_on_vis  = df[df['Current'] > PWM_THRESHOLD]
pwm_off_vis = df[df['Current'] <= PWM_THRESHOLD]

avg_current_on  = pwm_on_vis['Current'].mean()
avg_current_off = pwm_off_vis['Current'].mean()
avg_power_on    = pwm_on_vis['Power'].mean()
avg_power_off   = pwm_off_vis['Power'].mean()

print(f"   PWM ON  : {len(pwm_on_vis):,} data ({pct_on:.1f}%)")
print(f"   PWM OFF : {len(pwm_off_vis):,} data ({pct_off:.1f}%)")
print(f"   Rata-rata arus PWM ON  : {avg_current_on:.3f} A")
print(f"   Rata-rata arus PWM OFF : {avg_current_off:.3f} A")
print(f"   Rata-rata daya PWM ON  : {avg_power_on:.3f} W")
print(f"   Rata-rata daya PWM OFF : {avg_power_off:.3f} W")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Analisis Karakteristik SCC PWM (Model A - Dengan Nilai 0)\nPengaruh Switching terhadap Arus dan Daya',
             fontsize=12, fontweight='bold')

x     = np.arange(2)
width = 0.35
axes[0].bar(x - width/2, [avg_current_on, avg_current_off], width,
            label='Arus (A)', color='#4CAF50', alpha=0.8)
axes[0].bar(x + width/2, [avg_power_on, avg_power_off], width,
            label='Daya (W)', color='#F44336', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['PWM ON', 'PWM OFF'])
axes[0].set_ylabel('Nilai')
axes[0].set_title(f'Perbandingan Nilai Rata-rata\nPWM ON ({pct_on:.1f}%) vs PWM OFF ({pct_off:.1f}%)',
                  fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

sample_size = min(1000, len(df))
sample      = df.iloc[:sample_size].copy()
drop_points = sample[sample['Current'] <= PWM_THRESHOLD]

axes[1].plot(sample['Timestamp'], sample['Power'],
             linewidth=1.5, color='#1565C0', label='Daya (W)')
axes[1].scatter(drop_points['Timestamp'], drop_points['Power'],
                s=20, color='#F44336', label='PWM OFF (Nilai 0)', zorder=5)
axes[1].set_title('Fenomena Zero-Watt Drop Akibat Intervensi SCC PWM', fontweight='bold')
axes[1].set_xlabel('Waktu')
axes[1].set_ylabel('Daya (W)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(folder_grafik + 'model_A_analisis_pwm.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Grafik analisis PWM tersimpan!")

# ==============================================================================
# TAHAP 9: SMOOTHING DATA
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 9: SMOOTHING DATA")
print("="*60)

df['Power_Smoothed']   = df['Power'].rolling(window=20, min_periods=1).mean()
df['Current_Smoothed'] = df['Current'].rolling(window=20, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Perbandingan Data Mentah vs Smoothed (Model A - Dengan Nilai 0)\nRolling Average Window=20',
             fontsize=12, fontweight='bold')

axes[0].plot(df['Timestamp'], df['Current'],          color='#4CAF50', linewidth=0.4, alpha=0.6, label='Arus Mentah')
axes[0].plot(df['Timestamp'], df['Current_Smoothed'], color='#1B5E20', linewidth=1.5,            label='Arus Smoothed')
axes[0].set_title('Arus (A) - Mentah vs Smoothed', fontweight='bold')
axes[0].set_ylabel('Arus (A)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df['Timestamp'], df['Power'],          color='#F44336', linewidth=0.4, alpha=0.6, label='Daya Mentah')
axes[1].plot(df['Timestamp'], df['Power_Smoothed'], color='#B71C1C', linewidth=1.5,            label='Daya Smoothed')
axes[1].set_title('Daya (W) - Mentah vs Smoothed', fontweight='bold')
axes[1].set_ylabel('Daya (W)')
axes[1].set_xlabel('Waktu')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(folder_grafik + 'model_A_smoothing.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Grafik smoothing tersimpan!")

# ==============================================================================
# TAHAP 10: NORMALISASI DAN PEMBENTUKAN SEQUENCE
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 10: NORMALISASI DAN PEMBENTUKAN SEQUENCE")
print("="*60)

features_scale = [
    'Intesitas Cahaya',
    'Temperature',
    'Voltage',
    'Current_Smoothed',
    'Power_Smoothed'
]

features_input = [
    'Intesitas Cahaya',
    'Temperature',
    'Voltage',
    'Current_Smoothed'
]

target_col = 'Power_Smoothed'
target_idx_all = features_scale.index(target_col)

WINDOW = 20

# ==================================================
# NORMALISASI DATA
# ==================================================

data_all = df[features_scale].values

# Split index terlebih dahulu
total_data = len(data_all)

train_end = int(total_data * 0.70)
val_end   = int(total_data * 0.80)

train_raw = data_all[:train_end]
val_raw   = data_all[train_end:val_end]
test_raw  = data_all[val_end:]

# Fit scaler HANYA pada training
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_raw)

scaled_train = scaler.transform(train_raw)
scaled_val   = scaler.transform(val_raw)
scaled_test  = scaler.transform(test_raw)

print(f"Training Data   : {len(train_raw)}")
print(f"Validation Data : {len(val_raw)}")
print(f"Testing Data    : {len(test_raw)}")

# ==================================================
# MEMBUAT SEQUENCE
# ==================================================

def create_sequence(data_scaled, window=20):

    X = []
    y = []

    for i in range(len(data_scaled) - window):

        X.append(
            data_scaled[i:i+window, :4]
        )

        y.append(
            data_scaled[i+window, 4]
        )

    return np.array(X), np.array(y)

X_train, y_train = create_sequence(
    scaled_train,
    WINDOW
)

X_val, y_val = create_sequence(
    scaled_val,
    WINDOW
)

X_test, y_test = create_sequence(
    scaled_test,
    WINDOW
)

n_fitur = X_train.shape[2]

total_sequence = (
    len(X_train)
    + len(X_val)
    + len(X_test)
)

print("\n" + "="*50)
print(f"Total Sequence : {total_sequence}")
print(
    f"Training       : {len(X_train)} "
    f"({len(X_train)/total_sequence*100:.1f}%)"
)
print(
    f"Validation     : {len(X_val)} "
    f"({len(X_val)/total_sequence*100:.1f}%)"
)
print(
    f"Testing        : {len(X_test)} "
    f"({len(X_test)/total_sequence*100:.1f}%)"
)
print("="*50)

# ==================================================
# DATA UNTUK VISUALISASI NORMALISASI
# ==================================================

scaled_all = np.vstack([
    scaled_train,
    scaled_val,
    scaled_test
])

print("\n✅ Sequence berhasil dibuat")
print("✅ Split 70/10/20 berhasil diterapkan")
print("✅ scaled_all berhasil dibuat")

# ==================================================
# VISUALISASI PEMBAGIAN DATASET
# ==================================================

# Grafik 1: Bar Chart Horizontal
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle('Pembagian Dataset Training, Validation, dan Testing',
             fontsize=13, fontweight='bold')

# --- Bar Chart Horizontal ---
labels  = ['Dataset']
sizes   = [len(X_train), len(X_val), len(X_test)]
colors  = ['#4CAF50', '#2196F3', '#F44336']
lefts   = [0, len(X_train), len(X_train) + len(X_val)]
slabels = [
    f'Training\n{len(X_train)} seq\n({len(X_train)/total_sequence*100:.1f}%)',
    f'Validation\n{len(X_val)} seq\n({len(X_val)/total_sequence*100:.1f}%)',
    f'Testing\n{len(X_test)} seq\n({len(X_test)/total_sequence*100:.1f}%)'
]

for i in range(3):
    axes[0].barh(
        labels,
        sizes[i],
        left=lefts[i],
        color=colors[i],
        alpha=0.85,
        label=slabels[i],
        edgecolor='white',
        linewidth=1.5
    )

axes[0].set_xlabel('Jumlah Sequence', fontsize=10)
axes[0].set_title('Distribusi Jumlah Sequence', fontweight='bold')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].set_xlim(0, total_sequence * 1.05)

# --- Pie Chart ---
pie_labels = [
    f'Training\n{len(X_train)/total_sequence*100:.1f}%',
    f'Validation\n{len(X_val)/total_sequence*100:.1f}%',
    f'Testing\n{len(X_test)/total_sequence*100:.1f}%'
]

axes[1].pie(
    sizes,
    labels=pie_labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 10},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[1].set_title('Proporsi Pembagian Dataset', fontweight='bold')

plt.tight_layout()
plt.savefig(
    folder_grafik + 'split_dataset.png',
    dpi=150,
    bbox_inches='tight'
)
plt.show()
print("✅ Grafik pembagian dataset tersimpan!")

# ==============================================================================
# TAHAP 11: ARSITEKTUR & TRAINING LSTM
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 11: TRAINING DEEP LSTM (MODEL A - DENGAN NILAI 0)")
print("="*60)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW, n_fitur)),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(folder_model + 'model_A_lstm.h5', monitor='val_loss',
                    save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train, epochs=100, batch_size=64,
    validation_data=(X_val, y_val), callbacks=callbacks, verbose=1
)
print("\n✅ Training selesai!")

# ==============================================================================
# TAHAP 12: GRAFIK LOSS TRAINING
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 12: GRAFIK LOSS TRAINING")
print("="*60)

epochs_range = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, history.history['loss'],
         label='Training Loss', color='#F44336', linewidth=2)
plt.plot(epochs_range, history.history['val_loss'],
         label='Validation Loss', color='#FF9800', linewidth=2)
plt.fill_between(epochs_range,
                 history.history['loss'],
                 history.history['val_loss'],
                 alpha=0.1, color='red')
plt.title('Grafik Loss Training Model LSTM (Model A - Dengan Nilai 0)\nPrediksi Daya Output Panel Surya PLTS 50WP',
          fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(folder_grafik + 'model_A_loss_training.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Grafik loss training tersimpan!")

# ==============================================================================
# TAHAP 13: PREDIKSI & EVALUASI
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 13: PREDIKSI DAN EVALUASI")
print("="*60)

predicted_scaled = model.predict(X_test)

dummy_pred = np.zeros((len(predicted_scaled), len(features_scale)))
dummy_pred[:, target_idx_all] = predicted_scaled[:, 0]
y_pred = scaler.inverse_transform(dummy_pred)[:, target_idx_all]

dummy_actual = np.zeros((len(y_test), len(features_scale)))
dummy_actual[:, target_idx_all] = y_test
y_actual = scaler.inverse_transform(dummy_actual)[:, target_idx_all]

rmse = np.sqrt(np.mean((y_actual - y_pred)**2))
mae  = np.mean(np.abs(y_actual - y_pred))
r2   = 1 - (np.sum((y_actual - y_pred)**2) /
            np.sum((y_actual - y_actual.mean())**2))

mask_valid = y_actual > 0.5
if mask_valid.sum() == 0:
    mape = float('nan')
else:
    mape = np.mean(np.abs((y_actual[mask_valid] - y_pred[mask_valid]) /
                           y_actual[mask_valid])) * 100

print(f"\n{'='*50}")
print(f"  HASIL EVALUASI MODEL A (DENGAN NILAI 0)")
print(f"{'='*50}")
print(f"RMSE  : {rmse:.4f} Watt")
print(f"MAE   : {mae:.4f} Watt")
print(f"MAPE  : {mape:.2f} %")
print(f"R²    : {r2:.4f}")

pd.DataFrame({
    'Metric'     : ['RMSE', 'MAE', 'MAPE', 'R2'],
    'Value'      : [rmse, mae, mape, r2],
    'Keterangan' : [
        'Root Mean Square Error (W)',
        'Mean Absolute Error (W)',
        'Mean Absolute Percentage Error (%)',
        'Koefisien Determinasi'
    ]
}).to_csv(folder_hasil + 'metrics_model_A.csv', index=False)

# ==============================================================================
# TAHAP 14: GRAFIK PREDIKSI VS AKTUAL
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 14: GRAFIK PREDIKSI VS AKTUAL")
print("="*60)

plt.figure(figsize=(14, 6))
plt.plot(y_actual, color='#1565C0', linewidth=1.5, label='Daya Aktual (Smoothed)')
plt.plot(y_pred,   color='#F44336', linestyle='--', linewidth=1.5, label='Daya Prediksi LSTM')
plt.fill_between(range(len(y_actual)), y_actual.flatten(), y_pred.flatten(),
                 alpha=0.12, color='red', label='Error Area')
plt.title(f'Perbandingan Daya Aktual vs Prediksi LSTM (Data Mentah)\n'
          f'RMSE={rmse:.4f}W | MAPE={mape:.2f}% | R²={r2:.4f}',
          fontsize=12, fontweight='bold')
plt.xlabel('Waktu (Sampel Testing)')
plt.ylabel('Daya (Watt)')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(folder_grafik + 'model_A_prediksi_vs_aktual.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Grafik prediksi vs aktual tersimpan!")

# ==============================================================================
# TAHAP 15: ANALISIS SKENARIO CAHAYA
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 15: ANALISIS SKENARIO CAHAYA")
print("="*60)

rata_temp = df['Temperature'].mean()

skenario_cahaya = {
    'Sangat Mendung' : 2000,
    'Mendung'        : 8000,
    'Berawan'        : 20000,
    'Cerah Berawan'  : 35000,
    'Cerah'          : 50000,
    'Sangat Terik'   : 70000
}

hasil_skenario = []

for kondisi, lux in skenario_cahaya.items():

    # Cari 100 data aktual yang lux-nya paling dekat
    data_dekat = df.iloc[
        (df['Intesitas Cahaya'] - lux).abs().argsort()[:100]
    ]

    sim_temp = data_dekat['Temperature'].mean()
    sim_volt = data_dekat['Voltage'].mean()
    sim_curr = data_dekat['Current_Smoothed'].mean()

    # Bentuk sequence 20 timestep
    input_skenario = np.zeros((WINDOW, len(features_input)))
    input_skenario[:, 0] = lux
    input_skenario[:, 1] = sim_temp
    input_skenario[:, 2] = sim_volt
    input_skenario[:, 3] = sim_curr

    # Scaling — isi kolom Power_Smoothed dengan nilai minimum
    # scaler agar tidak terjadi distorsi saat transform
    dummy_scale = np.zeros((WINDOW, len(features_scale)))
    dummy_scale[:, 0] = input_skenario[:, 0]
    dummy_scale[:, 1] = input_skenario[:, 1]
    dummy_scale[:, 2] = input_skenario[:, 2]
    dummy_scale[:, 3] = input_skenario[:, 3]
    dummy_scale[:, 4] = scaler.data_min_[4]  # FIX: isi dengan min scaler

    input_scaled = scaler.transform(dummy_scale)[:, :4]

    # Prediksi
    window_input = input_scaled.reshape(1, WINDOW, n_fitur)
    pred_scaled  = model.predict(window_input, verbose=0)

    # Inverse Transform Target
    dummy_inv = np.zeros((1, len(features_scale)))
    dummy_inv[0, target_idx_all] = pred_scaled[0, 0]
    pred_watt = scaler.inverse_transform(dummy_inv)[0, target_idx_all]
    pred_watt = max(0, pred_watt)

    hasil_skenario.append({
        'Kondisi Cuaca'    : kondisi,
        'Intensitas (lux)' : lux,
        'Suhu (°C)'        : round(sim_temp, 2),
        'Tegangan (V)'     : round(sim_volt, 2),
        'Arus (A)'         : round(sim_curr, 3),
        'Prediksi Daya (W)': round(pred_watt, 4)
    })

    print(
        f"{kondisi:15} | "
        f"{lux:6} lux | "
        f"{sim_curr:.3f} A | "
        f"{pred_watt:.3f} W"
    )

df_skenario = pd.DataFrame(hasil_skenario)
print("\nData Skenario:")
print(df_skenario)

df_skenario.to_csv(
    folder_hasil + 'prediksi_skenario.csv',
    index=False
)

print(f"\nJumlah skenario : {len(df_skenario)}")
print("File CSV berhasil disimpan")

# ==============================================================================
# TAHAP 16: GRAFIK SKENARIO CAHAYA
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 16: GRAFIK SKENARIO CAHAYA")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

fig.suptitle(
    'Analisis Pengaruh Fluktuasi Cahaya\n'
    'Terhadap Prediksi Daya Output Panel Surya 50WP (LSTM)',
    fontsize=13,
    fontweight='bold'
)

colors = [
    '#0D47A1', '#1976D2', '#64B5F6',
    '#FF9800', '#F57C00', '#BF360C'
]

# FIX: set y_max eksplisit agar skala konsisten
y_max = df_skenario['Prediksi Daya (W)'].max() * 1.25

# Bar Chart
bars = axes[0].bar(
    df_skenario['Kondisi Cuaca'],
    df_skenario['Prediksi Daya (W)'],
    color=colors,
    edgecolor='white',
    linewidth=1.5,
    width=0.6
)

axes[0].set_title('Prediksi Daya per Kondisi Cuaca',
                  fontweight='bold', fontsize=11)
axes[0].set_ylabel('Prediksi Daya (W)', fontsize=10)
axes[0].tick_params(axis='x', rotation=30)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(0, y_max)  # FIX: skala eksplisit

for bar, (_, row) in zip(bars, df_skenario.iterrows()):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + y_max * 0.02,
        f"{row['Prediksi Daya (W)']:.2f}W",
        ha='center', va='bottom',
        fontsize=9, fontweight='bold'
    )

# Line Chart
axes[1].plot(
    df_skenario['Intensitas (lux)'],
    df_skenario['Prediksi Daya (W)'],
    color='#F44336',
    linewidth=2.5,
    marker='o',
    markersize=10,
    markerfacecolor='white',
    markeredgecolor='#F44336',
    markeredgewidth=2.5,
    label='Prediksi Daya'
)

axes[1].fill_between(
    df_skenario['Intensitas (lux)'],
    df_skenario['Prediksi Daya (W)'],
    alpha=0.15,
    color='#F44336'
)

axes[1].set_title('Tren Intensitas Cahaya vs Prediksi Daya',
                  fontweight='bold', fontsize=11)
axes[1].set_xlabel('Intensitas Cahaya (lux)', fontsize=10)
axes[1].set_ylabel('Prediksi Daya (W)', fontsize=10)
axes[1].set_ylim(0, y_max)  # FIX: skala eksplisit
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    folder_grafik + 'grafik_skenario_cahaya.png',
    dpi=200,
    bbox_inches='tight'
)
plt.show()
print("✅ Grafik analisis skenario cahaya berhasil dibuat!")

# ==============================================================================
# TAHAP 17: ANALISIS RESIDUAL
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 17: ANALISIS DISTRIBUSI RESIDUAL")
print("="*60)

residuals = y_actual - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Analisis Distribusi Residual (Data Mentah)\nModel LSTM Prediksi Daya PLTS',
             fontsize=12, fontweight='bold')

axes[0].hist(residuals, bins=40, color='#F44336', alpha=0.75, edgecolor='white')
axes[0].axvline(0,                color='black',  linestyle='--', linewidth=1.5, label='Nol')
axes[0].axvline(residuals.mean(), color='orange', linestyle='--', linewidth=1.5,
                label=f'Mean Error: {residuals.mean():.4f}W')
axes[0].set_title('Histogram Distribusi Residual', fontweight='bold')
axes[0].set_xlabel('Residual (W)')
axes[0].set_ylabel('Frekuensi')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_pred, residuals, alpha=0.2, color='#FF9800', s=8)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_title('Residual vs Nilai Prediksi', fontweight='bold')
axes[1].set_xlabel('Prediksi Daya (W)')
axes[1].set_ylabel('Residual (W)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(folder_grafik + 'model_A_residual.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Grafik residual tersimpan!")

# ==============================================================================
# TAHAP 18: KALKULASI TOTAL ENERGI
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 18: ANALISIS TOTAL ENERGI DAN BEBAN PENGGUNAAN")
print("="*60)

df['Energy_Wh']  = df['Power'] * (15 / 3600)
total_energi_wh  = df['Energy_Wh'].sum()
rata_rata_harian = total_energi_wh / total_hari

print(f"  Total Hari Operasi   : {total_hari} Hari")
print(f"  Total Produksi Energi: {total_energi_wh:.2f} Wh ({total_energi_wh/1000:.4f} kWh)")
print(f"  Rata-rata per Hari   : {rata_rata_harian:.2f} Wh per hari\n")
print(f"  ESTIMASI DURASI BEBAN LISTRIK MIKRO:")
print(f"  1. Lampu LED 10W   : {rata_rata_harian / 10:.1f} Jam")
print(f"  2. Charger HP 15W  : {rata_rata_harian / 15:.1f} Jam")
print(f"  3. Kipas Angin 25W : {rata_rata_harian / 25:.1f} Jam")
print(f"  4. TV LED 50W      : {rata_rata_harian / 50:.1f} Jam")

print("\n" + "="*60)
print("  MODEL A SELESAI DIOLAH!")
print("  File tersimpan dengan prefix 'model_A_' di folder Grafik dan Hasil")
print("="*60)