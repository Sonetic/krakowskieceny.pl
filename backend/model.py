import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


# ===== 1. wczytanie =====
df = pd.read_csv(os.path.join(BASE_DIR, "transactions_ready.csv"))
df["inv_powierzchnia"] = 1/df["powierzchnia_uzyt"]
df["relacja_ulica"] = df["srednia_budynek"] / df["srednia_cena_dzielnica"]
df["pokoje_na_m2"] = df["liczba_pokoi"] / df["powierzchnia_uzyt"]
df["near_diff"] = df["near_300"] - df["srednia_budynek"]
df["ulica_vs_budynek"] = df["srednia_cena_ulica"] - df["srednia_budynek"]
df["rodzaj_rynku"] = df["rodzaj_rynku"].apply(lambda x: 0 if x=="wtorny" else 1)


features = [
    "inv_powierzchnia",
    "liczba_pokoi",
    "piętro",
    "srednia_cena_dzielnica",
    #"srednia_cena_ulica",
    "ulica_vs_budynek",
    #"near_300",
    "near_diff",
    "pokoje_na_m2",
    "srednia_budynek",
   # "relacja_ulica",
    "dist_centrum",
    #"rodzaj_rynku"
]



target = "cena_za_m2"

X = df[features]
y = df[target]

# ===== 2. podział na percentyle =====
threshold = np.percentile(y, 98)

mask_normal = y <= threshold
mask_outliers = y > threshold

X_normal, y_normal = X[mask_normal], y[mask_normal]
X_out, y_out = X[mask_outliers], y[mask_outliers]

#print(f"Normal: {len(X_normal)}, Outliers: {len(X_out)}")

# ===== 3. split =====
Xn_train, Xn_test, yn_train, yn_test = train_test_split(
    X_normal, y_normal, test_size=0.2, random_state=42
)

Xo_train, Xo_test, yo_train, yo_test = train_test_split(
    X_out, y_out, test_size=0.2, random_state=42
)

# ===== 4. modele =====
model_normal = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

model_out = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", GradientBoostingRegressor(n_estimators=600, max_depth=3, learning_rate=0.01))
])

# ===== 5. trening =====
model_normal.fit(Xn_train, yn_train)
model_out.fit(Xo_train, yo_train)

# ===== 6. predykcja =====
yn_pred = model_normal.predict(Xn_test)
yo_pred = model_out.predict(Xo_test)

# train predictions
yn_pred_train = model_normal.predict(Xn_train)
yo_pred_train = model_out.predict(Xo_train)
'''
# ===== 7. metryki =====
def mape(y_true, y_pred):
    errors = np.abs(y_true - y_pred) / y_pred * 100

    print("<5%:", (errors < 5).mean())
    print("<10%:", (errors < 10).mean())
    print("<15%:", (errors < 15).mean())
    mdape = np.median(errors)

    print("MdAPE:", mdape)


    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape_val:.2f}%")



# policz korelacje między cechami
corr_matrix = df[features].corr()

# wyświetl w formie mapy cieplnej
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Macierz korelacji cech")
plt.show()




# ===== 8. filtr 2025 =====
df_2025 = df[df['rok'] == 2025]

X_2025 = df_2025[features]
y_2025 = df_2025[target]

# podział na percentyle tylko w 2025
threshold_2025 = np.percentile(y_2025, 98)
mask_normal_2025 = y_2025 <= threshold_2025
mask_out_2025 = y_2025 > threshold_2025

X_normal_2025 = X_2025[mask_normal_2025]
y_normal_2025 = y_2025[mask_normal_2025]

X_out_2025 = X_2025[mask_out_2025]
y_out_2025 = y_2025[mask_out_2025]

# predykcja
yn_pred_2025 = model_normal.predict(X_normal_2025)
yo_pred_2025 = model_out.predict(X_out_2025)


# metryki tylko dla 2025
print_metrics("NORMAL TEST 2025 (Ridge)", y_normal_2025, yn_pred_2025)
print_metrics("OUTLIERS TEST 2025 (GBR)", y_out_2025, yo_pred_2025)

print_metrics("NORMAL TRAIN (Ridge)", yn_train, yn_pred_train)
print_metrics("NORMAL TEST (Ridge)", yn_test, yn_pred)

print_metrics("OUTLIERS TRAIN (GBR)", yo_train, yo_pred_train)
print_metrics("OUTLIERS TEST (GBR)", yo_test, yo_pred)


coefs = model_normal.named_steps["model"].coef_

for f, c in zip(features, coefs):
    print(f"{f}: {c:.2f}")


'''