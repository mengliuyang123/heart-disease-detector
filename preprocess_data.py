import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib # 稍后用于保存预处理器

# 加载数据
df = pd.read_csv('/home/ubuntu/upload/heart_2022_with_nans.csv')

# 1. 处理目标变量缺失值
df = df.dropna(subset=['HadHeartAttack'])

# 2. 处理特征缺失值
# 对于数值型特征，使用中位数填充
num_cols = df.select_dtypes(include=['float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# 对于类别型特征，使用众数填充
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. 特征编码
# 目标变量编码
le_target = LabelEncoder()
df['HadHeartAttack'] = le_target.fit_transform(df['HadHeartAttack'])

# 类别特征编码 (使用 LabelEncoder 简化处理，或根据需要使用 OneHot)
# 为了方便 Streamlit 使用，我们记录每个编码器
encoders = {}
for col in cat_cols:
    if col != 'HadHeartAttack':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# 4. 划分数据集
X = df.drop('HadHeartAttack', axis=1)
y = df['HadHeartAttack']

# 考虑到数据集很大且极度不平衡，我们进行下采样以加快训练速度并平衡类别
# 或者在模型训练时使用 class_weight='balanced'
# 这里我们先保留完整数据，但在训练时注意平衡

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存处理后的数据和预处理器
import pickle
with open('/home/ubuntu/preprocessors.pkl', 'wb') as f:
    pickle.dump({'encoders': encoders, 'scaler': scaler, 'le_target': le_target, 'columns': X.columns.tolist()}, f)

np.save('/home/ubuntu/X_train.npy', X_train_scaled)
np.save('/home/ubuntu/X_test.npy', X_test_scaled)
np.save('/home/ubuntu/y_train.npy', y_train)
np.save('/home/ubuntu/y_test.npy', y_test)

print("Preprocessing complete. Data and preprocessors saved.")
