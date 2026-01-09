import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
import joblib

# 加载数据
X_train = np.load('/home/ubuntu/X_train.npy')
X_test = np.load('/home/ubuntu/X_test.npy')
y_train = np.load('/home/ubuntu/y_train.npy')
y_test = np.load('/home/ubuntu/y_test.npy')

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    print(f"\n--- {name} Performance ---")
    print(classification_report(y_test, y_pred))

# 1. 决策树
print("Training Decision Tree...")
dt = DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=10)
dt.fit(X_train, y_train)
evaluate_model(dt, X_test, y_test, "Decision Tree")
joblib.dump(dt, '/home/ubuntu/dt_model.joblib')

# 2. SVM
print("Training Linear SVM...")
svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=1000, dual=False)
svm.fit(X_train, y_train)
evaluate_model(svm, X_test, y_test, "Linear SVM")
joblib.dump(svm, '/home/ubuntu/svm_model.joblib')

# 3. 神经网络 (使用下采样平衡数据)
print("Training Neural Network (MLP) with Downsampling...")
X_train_df = pd.DataFrame(X_train)
X_train_df['target'] = y_train
df_majority = X_train_df[X_train_df.target == 0]
df_minority = X_train_df[X_train_df.target == 1]

# 下采样多数类
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])

X_train_balanced = df_balanced.drop('target', axis=1).values
y_train_balanced = df_balanced['target'].values

mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
mlp.fit(X_train_balanced, y_train_balanced)
evaluate_model(mlp, X_test, y_test, "Neural Network")
joblib.dump(mlp, '/home/ubuntu/mlp_model.joblib')

print("\nModels trained and saved.")
