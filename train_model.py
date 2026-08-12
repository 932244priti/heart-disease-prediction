import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("heart_disease_risk_2026 (2).csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. REMOVE ID
# ==========================================

df = df.drop(columns=["patient_id"])


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["has_heart_disease"])
y = df["has_heart_disease"]


# ==========================================
# 4. CONVERT BOOLEAN COLUMNS
# ==========================================

boolean_columns = [
    "exercise_induced_angina",
    "family_history",
    "wearable_owner"
]

for col in boolean_columns:
    X[col] = X[col].astype(int)


# ==========================================
# 5. COLUMN TYPES
# ==========================================

categorical_columns = [
    "sex",
    "chest_pain_type",
    "smoker_status"
]

numeric_columns = [
    col for col in X.columns
    if col not in categorical_columns
]


# ==========================================
# 6. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_columns
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ]
)


# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# 8. PREPROCESS
# ==========================================

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print("Processed training shape:", X_train.shape)


# ==========================================
# 9. CREATE MODEL
# ==========================================

model = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(128, activation="relu"),
    Dropout(0.30),

    Dense(64, activation="relu"),
    Dropout(0.20),

    Dense(32, activation="relu"),

    Dense(1, activation="sigmoid")
])


# ==========================================
# 10. COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 11. EARLY STOPPING
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ==========================================
# 12. TRAIN MODEL
# ==========================================

print("\nTraining model...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)


# ==========================================
# 13. EVALUATE
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n================================")
print("MODEL RESULTS")
print("================================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")


# ==========================================
# 14. CLASSIFICATION REPORT
# ==========================================

y_probability = model.predict(
    X_test,
    verbose=0
).ravel()

y_prediction = (y_probability >= 0.5).astype(int)

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_test,
        y_prediction,
        target_names=[
            "No Heart Disease",
            "Heart Disease"
        ]
    )
)


# ==========================================
# 15. SAVE MODEL
# ==========================================

model.save("heart_disease.keras")

joblib.dump(
    preprocessor,
    "preprocessor.pkl"
)


print("\n================================")
print("FILES SAVED")
print("================================")
print("heart_disease.keras")
print("preprocessor.pkl")
print("\nTraining completed successfully!")