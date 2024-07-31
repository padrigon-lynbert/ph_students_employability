import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_validate, cross_val_predict

from connections import get_dataframe


df = get_dataframe() # database(mysql): datasets, table: `ph-student_employability-dataset`

df['CLASS'] = df['CLASS'].map({'Employable': 1, 'LessEmployable': 2})

X = df.drop(['Name of Student', 'CLASS'], axis=1)
y = df['CLASS']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ismot', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

cv_results = cross_validate(pipeline, X, y, cv=5, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])

print("Cross-validation results:", cv_results)
print("Mean accuracy:", cv_results['test_accuracy'].mean())
print("Mean precision:", cv_results['test_precision_macro'].mean())
print("Mean recall:", cv_results['test_recall_macro'].mean())
print("Mean F1-score:", cv_results['test_f1_macro'].mean())


y_pred = cross_val_predict(pipeline, X, y, cv=5)
cm = confusion_matrix(y, y_pred)

labels = np.unique(y) 
print('classification_report:\n', classification_report(y, y_pred))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cbar=0)

comment = '1: Employable\n2: Less Employable'
plt.text(0.05, 0.2, comment, wrap=True, horizontalalignment='left', fontsize=10)
plt.ylabel('actual')
plt.xlabel('predicted')
plt.show()
