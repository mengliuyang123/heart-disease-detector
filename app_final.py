import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# 设置页面配置
st.set_page_config(page_title="心脏病发作检测系统 (完整逻辑版)", layout="wide")

# 加载模型和缩放器
@st.cache_resource
def load_resources():
    try:
        # 加载用户脚本中训练好的三个模型
        ann = joblib.load('ann_model.joblib')
        dt = joblib.load('dt_model.joblib')
        svm = joblib.load('heart_disease_model.pkl') # 用户脚本中保存的 SVM
        scaler = joblib.load('scaler.pkl')
        # 加载特征列表
        features = joblib.load('top_features.joblib')
        return ann, dt, svm, scaler, features
    except Exception as e:
        st.error(f"加载资源失败: {e}")
        return None, None, None, None, None

ann_model, dt_model, svm_model, scaler, top_features = load_resources()

st.title("🫀 心脏病发作风险检测系统")
st.markdown("""
本系统基于您的完整 Python 脚本逻辑构建，集成了 **ANN (神经网络)**、**DT (决策树)** 和 **SVM (支持向量机)** 三种模型。
系统采用了 **SMOTENC** 平衡技术和 **特征重要性筛选**，以提供更准确的医学风险评估。
""")

# 用户脚本中的映射字典
mapping_dict = {
    'GeneralHealth': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4},
    'LastCheckupTime': {
        '5 or more years ago': 0,
        'Within past 5 years (2 years but less than 5 years ago)': 1,
        'Within past 2 years (1 year but less than 2 years ago)': 2,
        'Within past year (anytime less than 12 months ago)': 3
    },
    'RemovedTeeth': {
        'None of them': 0,
        '1 to 5': 1,
        '6 or more, but not all': 2,
        'All': 3
    },
    'HadDiabetes': {
        'No': 0,
        'No, pre-diabetes or borderline diabetes': 1,
        'Yes, but only during pregnancy (female)': 2,
        'Yes': 3
    },
    'SmokerStatus': {
        'Never smoked': 0,
        'Former smoker': 1,
        'Current smoker - now smokes some days': 2,
        'Current smoker - now smokes every day': 3
    },
    'ECigaretteUsage': {
        'Never used e-cigarettes in my entire life': 0,
        'Not at all (right now)': 1,
        'Use them some days': 2,
        'Use them every day': 3
    },
    'AgeCategory': {
        'Age 18 to 24': 0, 'Age 25 to 29': 1, 'Age 30 to 34': 2, 'Age 35 to 39': 3,
        'Age 40 to 44': 4, 'Age 45 to 49': 5, 'Age 50 to 54': 6, 'Age 55 to 59': 7,
        'Age 60 to 64': 8, 'Age 65 to 69': 9, 'Age 70 to 74': 10, 'Age 75 to 79': 11,
        'Age 80 or older': 12
    }
}

# 侧边栏：模型选择
st.sidebar.header("模型设置")
selected_model_name = st.sidebar.selectbox("选择预测模型", ["SVM (推荐 - 高召回率)", "ANN (神经网络)", "决策树"])

model_map = {
    "SVM (推荐 - 高召回率)": svm_model,
    "ANN (神经网络)": ann_model,
    "决策树": dt_model
}
model = model_map[selected_model_name]

# 主界面：输入表单
st.header("请输入个人健康指标")
cols = st.columns(3)
input_values = {}

# 原始特征列表 (用于生成输入控件)
# 注意：这里需要包含 top_features 中涉及的所有原始特征
for i, col in enumerate(top_features):
    if col == 'HadHeartAttack': continue
    with cols[i % 3]:
        if col in mapping_dict:
            options = list(mapping_dict[col].keys())
            input_values[col] = st.selectbox(f"{col}", options)
        elif col in ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']:
            input_values[col] = st.number_input(f"{col}", value=0.0)
        elif col in ['Sex']:
            input_values[col] = st.selectbox(f"{col}", ["Female", "Male"])
        else:
            # 默认为 Yes/No 二分类
            input_values[col] = st.selectbox(f"{col}", ["No", "Yes"])

# 预测按钮
if st.button("开始评估风险"):
    if model is None or scaler is None:
        st.error("资源未加载，请检查后台。")
    else:
        # 1. 构造输入 DataFrame
        input_df = pd.DataFrame([input_values])
        
        # 2. 执行特征工程逻辑
        processed_df = input_df.copy()
        
        # A. 有序映射
        for col, mapping in mapping_dict.items():
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].map(mapping)
        
        # B. 二分类映射
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                val = processed_df[col].iloc[0]
                if val in ['Yes', 'No']:
                    processed_df[col] = processed_df[col].map({'Yes': 1, 'No': 0})
                elif val in ['Female', 'Male']:
                    processed_df[col] = processed_df[col].map({'Female': 1, 'Male': 0})
        
        # C. 确保列顺序与 top_features 一致 (排除目标变量)
        final_features = [f for f in top_features if f != 'HadHeartAttack']
        processed_df = processed_df[final_features]
        
        # 3. 缩放
        scaled_data = scaler.transform(processed_df)
        
        # 4. 预测
        prediction = model.predict(scaled_data)[0]
        
        # 获取概率
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled_data)[0][1]
        elif hasattr(model, "decision_function"):
            df_val = model.decision_function(scaled_data)[0]
            prob = 1 / (1 + np.exp(-df_val))

        # 显示结果
        st.divider()
        if prediction == 1:
            st.error(f"### 评估结果：高风险")
            if prob is not None:
                st.write(f"风险概率: {prob:.2%}")
            st.warning("根据您的健康指标，系统检测到较高的心脏病发作风险。")
        else:
            st.success(f"### 评估结果：低风险")
            if prob is not None:
                st.write(f"风险概率: {prob:.2%}")
            st.info("您的健康指标显示心脏病发作风险较低。")

st.sidebar.markdown("""
### 模型性能参考 (测试集)
- **SVM**: Recall 0.72, F1 0.34
- **ANN**: F1 0.32
- **DT**: F1 0.30
""")
