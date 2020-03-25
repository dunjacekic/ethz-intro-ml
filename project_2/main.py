import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np

# 1. Import data
train_data = pd.read_csv('train_features.csv')
df = pd.DataFrame(train_data)

train_labels = pd.read_csv('train_labels.csv') # Last 4 columns are real-valued, everything else is boolean
df = pd.DataFrame(train_labels)
df.drop("pid", axis=1)

# 2.  Pre-processing
# 2a. Handle missing data
df = df.fillna(df.mean())

# 2b. Concatenate rows from a single patient, do not duplicate age
df = df.drop("Time", axis=1)
num_patients = int(df.shape[0]/12)
num_feats = int(df.shape[1]*12 - 12 - 11) # Remove patient id, extra age entries

df_np = df.to_numpy()
df_np_new = np.zeros((num_patients, num_feats))

for patient_idx in range(num_patients):
    df_np_new[0,:] = np.hstack((df_np[patient_idx,1:], df_np[patient_idx+1,2:], df_np[patient_idx+2,2:], 
                                df_np[patient_idx+3,2::], df_np[patient_idx+4,2:], df_np[patient_idx+5,2:], 
                                df_np[patient_idx+6,2::], df_np[patient_idx+7,2:], df_np[patient_idx+8,2:], 
                                df_np[patient_idx+9,2::], df_np[patient_idx+10,2:], df_np[patient_idx+11,2:]))
                                
# 2c. Normalize
scaler = StandardScaler()
scaler.fit(df_np_new)
df_np_new = scaler.transform(df_np_new)

# 3.  Model architecture
#model = keras.Sequential()
#model.add ...

# 4.  Train the model using SGD

print(df.info())