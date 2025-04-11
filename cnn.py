import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_csv('/content/Mobile_Price_Classification-220531-204702.csv')#dataset file

print(df.head())
X=df.drop('price_range', axis=1)#using other columns for x expect price range
y=df['price_range']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)# preprocessing normalization of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)#75% and 25% training and testing data

model=Sequential([
    Dense(8, activation='relu', input_shape=(X.shape[1],)),
    Dense(4, activation='relu'),#2 hidden layers
    Dense(1, activation='sigmoid')  #binary class clasification
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test)) #batch size for training

