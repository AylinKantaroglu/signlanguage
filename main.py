
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('C:/Users/aylen/signlanguage/Arabic Sign Language Letters Dataset.csv')
print("DAtensatzgröße: ", df.shape)
print(df.head()) # Erste Zeilen anzeigen
# 1. Features (X): Alle Spalten außer der Label-Spalte
X = df.drop(columns=['letter']).values 

# 2. Labels (y): Nur die Label-Spalte
y = df['letter'].values

print("Shape von X (Koordinaten):", X.shape)
print("Shape von y (Labels):", y.shape)

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# a) Buchstaben in Zahlen umwandeln (z.B. 'Alif' -> 0, 'Ba' -> 1)p
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
# Zeigt die Mapping an: z.B. (0: 'Alif', 1: 'Ba', ...)
print("Anzahl der Klassen:", len(label_encoder.classes_))

# b) Zahlen in One-Hot-Vektoren umwandeln (z.B. 0 -> [1, 0, 0, ...])
y_encoded = to_categorical(integer_encoded) 

NUM_CLASSES = y_encoded.shape[1]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y_encoded, 
    test_size=0.2, # 20% für die Validierung
    random_state=42, # Sorgt für reproduzierbare Ergebnisse
    stratify=y # Stellt sicher, dass alle Klassen gleichmäßig aufgeteilt werden
)

print("Trainings-Set Größe:", X_train.shape, y_train.shape)
print("Validierungs-Set Größe:", X_val.shape, y_val.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Die Anzahl der Input-Features (Koordinaten-Spalten)
INPUT_SHAPE = X_train.shape[1] 

model = Sequential([
    # Input Layer (mit erster Hidden Layer)
    Dense(128, activation='relu', input_shape=(INPUT_SHAPE,)),
    # Dropout hilft gegen Overfitting
    Dropout(0.2), 
    
    # Zweite Hidden Layer
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    # Output Layer
    # NUM_CLASSES muss der Anzahl der zu erkennenden Buchstaben entsprechen
    Dense(NUM_CLASSES, activation='softmax') 
])

# Zeigt die Struktur des Modells
model.summary()

# Definieren des Optimierers, der Verlustfunktion und der Metriken
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', # Standard für Multi-Klassen-Klassifikation
    metrics=['accuracy']
)

# Training des Modells
history = model.fit(
    X_train, 
    y_train, 
    epochs=30, # Starten Sie mit z.B. 30 Epochen
    batch_size=32, 
    validation_data=(X_val, y_val),
    verbose=1
)