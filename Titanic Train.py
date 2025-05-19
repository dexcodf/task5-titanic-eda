# %%
mkdir Titanic_EDA
cd Titanic_EDA
python -m venv venv
# Activate it
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
pip install pandas matplotlib seaborn notebook

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
df = pd.read_csv("titanic.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()
df['Survived'].value_counts()
df['Age'].hist(bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Survival vs Age")
plt.show()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Gender")
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival Count by Passenger Class")
plt.show()
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived')
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# %% [markdown]
# ### Key Insights from Titanic Data Exploration
# 
# - **Age Distribution:** The majority of passengers were young adults, with a notable number of children and elderly.
# - **Survival by Age:** Younger passengers, especially children, had a higher survival rate.
# - **Gender and Survival:** Females had a significantly higher chance of survival compared to males.
# - **Class and Survival:** Passengers in 1st and 2nd class were more likely to survive than those in 3rd class.
# - **Correlation Analysis:** Survival is positively correlated with being female and being in a higher class, and negatively correlated with age and fare for some groups.
# 
# These findings highlight the impact of age, gender, and class on survival rates during the Titanic disaster.


