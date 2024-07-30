import numpy as np
import seaborn as sns
from kunikchon import get_df
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


df = get_df()


df['CLASS'] = df['CLASS'].map({'Employable':1, 'LessEmployable':2})


X = df.drop({'Name of Student', 'CLASS'}, axis=1)
y = df['CLASS']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
X_train_b, y_train_b = smote.fit_resample(X_train, y_train)


rf_class_model = RandomForestClassifier()
rf_class_model.fit(X_train_b, y_train_b)
y_pred = rf_class_model.predict(X_test)


cr = classification_report(y_test, y_pred); print(cr)
conf = confusion_matrix(y_test, y_pred)


index =  np.unique(y)

sns.heatmap(conf, annot=True, xticklabels=index, yticklabels=index, fmt='d')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('1-employable, 2-lessEmployable ph students')
plt.show()


