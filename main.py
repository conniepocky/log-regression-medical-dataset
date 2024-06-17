from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# use logistic regression - binary outcome
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Liver_disease_data.csv")

df = df.drop('LiverFunctionTest', axis=1)

y = df.Diagnosis
X = df.drop('Diagnosis', axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train, y_train)

y_pred = log_model.predict(x_test)

score = log_model.score(x_test, y_test)
print("Accuracy:", np.round(score, 3))

# plot factors for liver disease, which one gives greatest risk
 
coefs = log_model.coef_[0]
coefs = np.abs(coefs)

factors = x_train.columns

df_factors = pd.DataFrame({'factors': factors, 'coefs': coefs})

df_factors = df_factors.sort_values(by='coefs', ascending=False)

plt.figure(figsize=(9, 6))

sns.barplot(x='coefs', y='factors', data=df_factors, palette='coolwarm')

plt.title('Factors and their coefficients for liver disease')

plt.show()