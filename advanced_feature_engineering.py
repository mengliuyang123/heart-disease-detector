import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import pickle

# 加载原始数据
df = pd.read_csv('/home/ubuntu/heart_disease_detector/heart_2022_with_nans.csv')

# 1. 基础清洗
df = df.dropna(subset=['HadHeartAttack'])
target = 'HadHeartAttack'
y = (df[target] == 'Yes').astype(int)
X = df.drop(columns=[target, 'State']) # 'State' 通常对个人风险预测意义不大，移除以减少维度

# 2. 特征构造 (Feature Construction)
print("Constructing new features...")
# BMI 分类
def get_bmi_category(bmi):
    if bmi < 18.5: return 'Underweight'
    if bmi < 25: return 'Normal'
    if bmi < 30: return 'Overweight'
    return 'Obese'

# 填充 BMI 缺失值以便构造特征
X['BMI'] = X['BMI'].fillna(X['BMI'].median())
X['BMICategory'] = X['BMI'].apply(get_bmi_category)

# 健康天数合并 (总不健康天数)
X['PhysicalHealthDays'] = X['PhysicalHealthDays'].fillna(0)
X['MentalHealthDays'] = X['MentalHealthDays'].fillna(0)
X['TotalUnhealthyDays'] = X['PhysicalHealthDays'] + X['MentalHealthDays']

# 3. 识别特征类型
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 4. 构建预处理流水线 (Advanced Pipeline)
print("Building preprocessing pipeline...")
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. 拟合预处理器
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 获取 One-Hot 编码后的特征名称
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features_transformed = cat_encoder.get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + cat_features_transformed

# 6. 保存预处理器和处理后的数据
joblib.dump(preprocessor, '/home/ubuntu/heart_disease_detector/advanced_preprocessor.joblib')
joblib.dump(all_feature_names, '/home/ubuntu/heart_disease_detector/feature_names.joblib')

np.save('/home/ubuntu/X_train_adv.npy', X_train_processed)
np.save('/home/ubuntu/X_test_adv.npy', X_test_processed)
np.save('/home/ubuntu/y_train_adv.npy', y_train)
np.save('/home/ubuntu/y_test_adv.npy', y_test)

print(f"Feature engineering complete. Total features: {len(all_feature_names)}")
print(f"Processed data saved.")
