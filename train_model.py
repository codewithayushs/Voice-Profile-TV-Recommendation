import pandas as pd
import numpy as np
import librosa
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

BASE_DIR = r"C:\Users\Ayush\OneDrive\Desktop\final dl project"

DATA_CSV = os.path.join(BASE_DIR,"data","metadata","cv-valid-train.csv")
AUDIO_DIR = os.path.join(BASE_DIR,"data","clips")

df = pd.read_csv(DATA_CSV)

df = df[['filename','age','gender','accent']]
df = df.dropna()

print("Dataset size:",len(df))

def extract_features(file):

    try:

        audio_path = os.path.join(AUDIO_DIR,file)

        y,sr = librosa.load(audio_path,sr=16000)

        mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40)

        return np.mean(mfcc.T,axis=0)

    except:
        return None


features=[]
ages=[]
genders=[]
accents=[]

for i,row in df.iterrows():

    feat = extract_features(row['filename'])

    if feat is not None:

        features.append(feat)

        ages.append(row['age'])

        genders.append(row['gender'])

        accents.append(row['accent'])


X=np.array(features)

age_encoder=LabelEncoder()
gender_encoder=LabelEncoder()
accent_encoder=LabelEncoder()

y_age=age_encoder.fit_transform(ages)
y_gender=gender_encoder.fit_transform(genders)
y_accent=accent_encoder.fit_transform(accents)

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y_gender,test_size=0.2)

model=tf.keras.Sequential([

    tf.keras.layers.Dense(256,activation='relu',input_shape=(40,)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(64,activation='relu'),

    tf.keras.layers.Dense(len(gender_encoder.classes_),activation='softmax')
])

model.compile(

optimizer='adam',

loss='sparse_categorical_crossentropy',

metrics=['accuracy']

)

model.fit(X_train,y_train,epochs=20,batch_size=32,validation_data=(X_test,y_test))

os.makedirs("model",exist_ok=True)

model.save("model/voice_model.h5")

joblib.dump(age_encoder,"model/age_encoder.pkl")
joblib.dump(gender_encoder,"model/gender_encoder.pkl")
joblib.dump(accent_encoder,"model/accent_encoder.pkl")
joblib.dump(scaler,"model/scaler.pkl")

print("Model Training Completed")