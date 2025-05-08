# %% [markdown]
# # Student performance analysis

# %%
# import modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# %% [markdown]
# ## Overview Dataset

# %%
df = pd.read_csv('student_habits_performance.csv')
df.head()

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# ## Data cleaning

# %%
df.isna().sum()

# %%
category_cols = df.select_dtypes(include="object").columns
category_cols = [cat for cat in category_cols]
category_cols.remove('student_id')
category_cols

# %% [markdown]
# ## Visualize the data

# %%
plt.figure(figsize=(9,6))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.hist(df[category_cols[i]], edgecolor='black')
    plt.title(category_cols[i])

plt.tight_layout()
plt.show()

# %%
num_col = df.select_dtypes(exclude='object')
num_col = [nums for nums in num_col]
num_col

# %%
plt.figure(figsize=(12,12))
for i in range(len(num_col)):
    plt.subplot(3, 3, i+1)
    plt.hist(df[num_col[i]], bins=20, edgecolor='black')
    plt.title(num_col[i])

plt.tight_layout()
plt.show()

# %%
# df without student_id
df2 = df.drop(columns='student_id')
df2.head()

# %% [markdown]
# ## Encode features

# %%
category_cols

# %% [markdown]
# labelencoding = diet_quality, parental_education_level, internet_quality
# 
# OHE = gender, part_time_job, extracurricular_participation

# %%
diet_quality = {'Poor': 0, 'Fair': 1, 'Good': 2}
parental_education_level = {'High School' : 0, 'Bachelor' : 1, 'Master' : 2, 'None': 3}
internet_quality  = {'Poor' : 0, 'Average' : 1, 'Good' : 2}

# %%
df2['diet_encoded'] = df2['diet_quality'].map(diet_quality)
df2['parental_encoded'] = df['parental_education_level'].map(parental_education_level)
df2['internet_encoded'] = df['internet_quality'].map(internet_quality)
df2.head()

# %%
dummies = pd.get_dummies(df[['gender', 'part_time_job', 'extracurricular_participation']],drop_first=True)
dummies.head()

# %%
df3 = pd.concat([df2,dummies], axis=1)

# %%
df3 = df3.drop(columns=['gender', 'part_time_job', 'extracurricular_participation', 'diet_quality', 'parental_education_level', 'internet_quality'])

# %% [markdown]
# ## Correlation & Heat Map

# %%
correlation = df3.corr()
sns.heatmap(correlation, annot=True)

# %%
X = df3.drop(columns='exam_score')
y = df3['exam_score']
X.shape, y.shape

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_shape = X_scaled.shape
X_scaled

# %% [markdown]
# ## Train Test

# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_model.score(X_test, y_test)

# %% [markdown]
# ## Results

# %%
# Compare Linear regression with other models

models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} → R2: {r2_score(y_test, y_pred):.3f}, RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")


