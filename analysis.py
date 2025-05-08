# %% [markdown]
# # Student performance analysis

# %%
# import modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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



