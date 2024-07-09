#!/usr/bin/env python
# coding: utf-8

# # Proyek Machine Learning Terapan : Klasifikasi Bintang, Galaksi dan Quasar
# - **Nama:** Irma Indriana Hariawang
# - **Email:** irma_loh@yahoo.com
# - **ID Dicoding:** irma_h
# - **Sumber Dataset:** https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17?resource=download

# ### Import semua packages/library yang digunakan

# In[1]:


import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import plotly.express as px
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from astropy.coordinates import SkyCoord
import astropy.units as u
import plotly.graph_objs as go

warnings.filterwarnings('ignore')


# ## 1. Data Understanding

# ### 1.1. Data Loading

# Dataset : star_classification 2.csv

# In[2]:


# Membaca data dari file .csv 

data_df = pd.read_csv("star_classification 2.csv")
data_df.head(10)


# In[3]:


data_df.info()


# In[4]:


data_df.describe()


# ### 1.2. Missing Values

# In[5]:


# Menghitung jumlah missing values

print("Jumlah missing values : ", data_df.isna().sum())


# ### 1.3. Duplikasi

# In[6]:


# Menghitung jumlah data duplikasi

print("Jumlah duplikasi : ", data_df.duplicated().sum())


# ### 1.4. Outliers

# #### 1.4.1. Detect Outliers

# In[7]:


# Mendeteksi outliers

sns.boxplot(data_df['i'])


# In[8]:


sns.boxplot(data_df['redshift'])


# #### 1.4.2. Remove Outliers

# In[9]:


# Fungsi menghapus outliers

def rem_outliers() :
    s1 = data_df.shape

    for i in data_df.select_dtypes(include = 'number').columns :
        qt1 = data_df[i].quantile(0.25)
        qt3 = data_df[i].quantile(0.75)
        iqr = qt3 - qt1
        lower = qt1-(1.5*iqr)
        upper = qt3+(1.5*iqr)
        min_in = data_df[data_df[i]<lower].index
        max_in = data_df[data_df[i]>upper].index
        data_df.drop(min_in, inplace = True)
        data_df.drop(max_in, inplace = True)

    s2 = data_df.shape
    outliers = s1[0] - s2[0]
    return outliers


# In[10]:


print ("Number of outliers deleted : ", rem_outliers())


# ### 1.5. EDA - Univariate Analysis

# #### 1.5.1. Numerical Features

# Grafik penyebaran obyek langit berdasarkan class yang di plot terhadap posisi

# In[11]:


# Grafik alpha, delta terhadap class

for i in ['alpha', 'delta']:
    plt.figure(figsize=(13,7))
    sns.histplot(data=data_df, x=i, kde=True, hue="class")
    plt.title(i)
    plt.show()
    


# In[12]:


# Penyebaran posisi obyek langit yang di plot terhadap koordinat langit

coords = SkyCoord(ra=data_df['alpha']*u.degree, dec=data_df['delta']*u.degree, frame='icrs', unit='deg')

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='mollweide')
ax.scatter(coords.ra.wrap_at(180*u.degree).radian, coords.dec.radian, s=1)
ax.grid()
plt.show()


# <div style="border-radius:10px; border:#3943B7 solid; padding: 15px; background-color: #C9D8FF; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#3943B7'>Color Magnitude Diagram</font></h3>
# Filter fotometri digunakan dalam pengamatan astronomi untuk memisahkan dan menganalisis cahaya objek-objek langit dalam berbagai panjang gelombang. Hasil dari pengamatan mengunakan filter fotometri ini akan di plot dalam Diagram Warna Mangnitudo (Color Magnitude Diagram (CMD)). Dengan menggunakan CMD kita dapat memvisualisasikan dan menganalisis properti-properti dari kelompok bintang tersebut dalam sebuah galaksi. CMD menggambarkan hubungan antara magnitudo absolut (kecerahan intrinsik bintang) dan indeks warna (perbedaan kecerahan antara dua panjang gelombang cahaya yang berbeda). CMD merupakan alat penting dalam memahami sifat-sifat bintang dalam galaksi, evolusi galaksi, dan pembentukan struktur kosmos secara umum.

# In[13]:


# Grafik color magnitude diagram
color = data_df['u'] - data_df['g']
# Define magnitude
mag = data_df['r']

# Plot CMD
fig = px.scatter(x=color, y=mag, color=color, opacity=0.5)
fig.update_layout(xaxis_title='u - g', yaxis_title='magnitude')
fig.show()


# <div style="border-radius:10px; border:#3943B7 solid; padding: 15px; background-color: #C9D8FF; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#3943B7'>Pergeseran Merah (Redshift)</font></h3>
# Redshift adalah pergeseran cahaya yang dihasilkan oleh benda-benda langit ke panjang gelombang merah (lebih besar). Apabila sebuah benda langit diamati mengalami redshift hal itu berarti posisinya semakin menjauhi pengamat. Begitu pula sebaliknya, bila benda langit tersebut mengalami blueshift itu berarti posisinya mendekati pengamat. Pengamatan redshift ini penting dalam astronomi yaitu untuk mempelajari sifat-sifat alam semesta, seperti jarak, kecepatan dan evolusi benda-benda langit.

# In[14]:


# Grafik Redshift Distribution
fig = go.Figure(data=go.Histogram(x=data_df['redshift'], nbinsx=50))

fig.update_layout(title='Redshift Distribution',
                  xaxis_title='Redshift',
                  yaxis_title='Number of Objects')

fig.show()


# <div style="border-radius:10px; border:#3943B7 solid; padding: 15px; background-color: #C9D8FF; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#3943B7'>Pengamatan Spektroskopi</font></h3>
# Pengamatan spektroskopi adalah pengamatan yang mengamati spektrum cahaya dari berbagai obyek langit, seperti : bintang, kuasar dan galaksi. Alat pengamatannya disebut spektrograph. Beberapa kegunaan dari pengamatan spektroskopi adalah untuk identifikasi dan klasifikasi objek, penelitian galaksi, kuasar dan objek aktif lainnya serta studi pergeseran merah (redshift).

# In[15]:


# Grafik Spektroskopi
size = [s if s >= 0 else 0 for s in data_df['redshift']]
fig = px.scatter(data_df, x='alpha', y='delta', color='class', size= size)
fig.update_layout(title='Spectroscopic Observations',
                  xaxis_title='Alpha (deg)',
                  yaxis_title='Delta (deg)')

fig.show()


# #### 1.5.2. Categorical Features

# In[16]:


# Menghitung jumlah data dalam tiap kategori class

a, b, c = data_df["class"].value_counts()/len(data_df)
print(f"Total percentage of Galaxies : {round(a*100,1)}%")
print(f"Total percentage of Stars : {round(b*100,1)}%")
print(f"Total percentage of Quasars : {round(c*100,1)}%")


# In[17]:


# Menggambarkan grafiknya

sns.set(rc = {'figure.figsize' : (12,7)})
count_plot = sns.countplot(x = data_df["class"], palette="Set3")
count_fig = count_plot.get_figure()
count_fig.savefig("class_count.png")


# ### 1.6. EDA - Multivariate Analysis

# In[18]:


# Menggambarkan matriks korelasi

plt.figure(figsize=(10,8))
correlation_matrix = data_df.corr(numeric_only=True)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidth=0.5)
plt.title("Correlation Matrix untuk Fitur Numerik", size=20)
plt.savefig("correlation_matrix.png")


# -----

# ## 2. Data Preparation

# ### 2.1. Encoding Fitur Kategori

# In[19]:


# Encoding fitur kategori yaitu kolom class

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data_df['class'] = le.fit_transform(data_df['class'])
data_df['class']= data_df['class'].astype(int)
data_df.head()


# In[20]:


# Mengurutkan tingkat korelasi antar kolom dari correlation matrix

corr = data_df.corr()
corr["class"].sort_values() 


# In[21]:


# Menghapus kolom yang tidak digunakan

data_df.drop(['obj_ID','run_ID','rerun_ID','field_ID','spec_obj_ID','plate','MJD','fiber_ID','cam_col','delta','alpha'], axis=1, inplace = True)
data_df


# ### 2.2. Imbalance Data

# In[22]:


# Mengatasi Imbalance data dengan teknik resample

X = data_df.drop(["class"],axis = 1)
y = data_df["class"]


# In[23]:


# Jumlah data awal

y.value_counts()


# In[24]:


# Proses resample

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 30, k_neighbors = 5)
X_resample, y_resample = sm.fit_resample(X, y)


# In[25]:


# Jumlah data setelah proses resample

y_resample.value_counts()


# ### 2.4. Train-Test-Split

# In[26]:


# Membagi dataset menjadi train dan test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size = 0.2, random_state = 30)


# In[27]:


print(f'Total of sample in whole dataset: {len(X_resample)}')
print(f'Total of sample in train dataset: {len(X_train)}')
print(f'Total of sample in test dataset: {len(X_test)}')


# ### 2.3. Standarisasi

# In[28]:


# Standarisasi fitur numerik

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----

# ## 3. Model Development

# In[29]:


# Membuat dataframe untuk menyimpan nilai akurasi Recall Score

score_df = pd.DataFrame(columns=['Algorithm', 'Accuracy'])


# ### 3.1. K-Nearest Neighbors

# Untuk menemukan nilai n_neighbors yang optimal, digunakan metode Grid Search cross-validation.

# In[30]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Membuat model KNN
knn = KNeighborsClassifier()

# Menyiapkan parameter untuk tuning
params = {'n_neighbors': list(range(1, 6)),
          'weights': ['uniform', 'distance'],
          'metric': ['euclidean', 'manhattan', 'minkowski']}

# Melakukan optimasi dengan GridSearchCV
grid_search = GridSearchCV(knn, params, cv=5, scoring='accuracy',
n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Menampilkan parameter terbaik
print("Parameter Terbaik:", grid_search.best_params_)


# In[31]:


# Memprediksi data pengujian
y_pred1 = grid_search.predict(X_test_scaled)

# Menghitung akurasi
knn_score = accuracy_score(y_test, y_pred1)
print("Akurasi:", knn_score)


# In[32]:


# Menambahkan accuracy KNN

score_df = score_df._append({'Algorithm' : 'KNN', 'Accuracy' : knn_score}, ignore_index = True)
score_df


# ### 3.2. Decision Tree

# In[33]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state = 30)
model.fit(X_train_scaled, y_train)
y_pred3 = model.predict(X_test_scaled)

dtree_score = accuracy_score(y_test, y_pred3)
print(dtree_score)


# In[34]:


score_df = score_df._append({'Algorithm':'Decision Tree', 'Accuracy': dtree_score}, ignore_index = True)
score_df


# ### 3.3. Random Forest

# In[35]:


# Dibuat sebuah dataset untuk menyimpan nilai recall dari algoritma Random Forest
# akan dipilih nilai recall terbesar untuk menentukan nilai dari n estimators

rf_df = pd.DataFrame(columns=['Estimators','Accuracy'])


# In[36]:


from sklearn.ensemble import RandomForestClassifier

for i in range(1,21):
    model = RandomForestClassifier(n_estimators = i, random_state = 30)
    model.fit(X_train_scaled, y_train)
    y_pred2 = model.predict(X_test_scaled)
    rf_df = rf_df._append({'Estimators':i, 'Accuracy':accuracy_score(y_test, y_pred2)}, ignore_index = True)

rf_df


# In[37]:


rf_df = rf_df.sort_values(by='Accuracy', ascending = False)
rf_df.head()


# Dari hasil perhitungan didapatkan nilai n_estimators terbaik adalah 20.

# In[38]:


# Random Forest dengan n_estimators = 20

model = RandomForestClassifier(n_estimators = 20, random_state = 30)
model.fit(X_train_scaled, y_train)
y_pred2 = model.predict(X_test_scaled)

rf_score = accuracy_score(y_test, y_pred2)
rf_score


# In[39]:


# Menambahkan score ke tabel

score_df = score_df._append({'Algorithm':'Random Forest','Accuracy': rf_score}, ignore_index = True)
score_df


# --------

# ## 4. Evaluation

# In[40]:


score_df = score_df.set_index('Accuracy')
score_df


# In[41]:


score_df = score_df.sort_values(by='Accuracy', ascending = False)
score_df


# -------
