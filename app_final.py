import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 设置页面配置
st.set_page_config(page_title="心脏病发作检测系统 (用户定制版)", layout="wide")

# 加载模型和缩放器
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"加载模型或缩放器失败: {e}")
        return None, None

model, scaler = load_resources()

st.title("🫀 心脏病发作风险检测系统")
st.markdown("本系统已根据您的 **自定义特征工程逻辑** 进行了重构。")

# 用户提供的映射字典
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

# 用户指定的 16 个特征顺序 (根据之前的输入)
features_order = [
    'DeafOrHardOfHearing', 'HadStroke', 'HadDiabetes', 'DifficultyWalking',
    'PneumoVaxEver', 'AgeCategory', 'GeneralHealth', 'DifficultyErrands',
    'HadArthritis', 'HadKidneyDisease', 'Sex', 'HadAngina', 'ChestScan',
    'RemovedTeeth', 'HadCOPD', 'PhysicalHealthDays'
]

# 主界面：输入表单
st.header("请输入个人健康指标")
cols = st.columns(3)
input_values = {}

for i, col in enumerate(features_order):
    with cols[i % 3]:
        if col in mapping_dict:
            # 有序变量：使用映射字典的键作为选项
            options = list(mapping_dict[col].keys())
            input_values[col] = st.selectbox(f"{col}", options)
        elif col in ['PhysicalHealthDays']:
            # 数值变量
            input_values[col] = st.number_input(f"{col}", min_value=0, max_value=30, value=0)
        elif col in ['Sex']:
            # 性别映射
            input_values[col] = st.selectbox(f"{col}", ["Female", "Male"])
        else:
            # 二分类变量 (Yes/No)
            input_values[col] = st.selectbox(f"{col}", ["No", "Yes"])

# 预测按钮
if st.button("开始评估风险"):
    if model is None or scaler is None:
        st.error("模型未加载，无法预测。")
    else:
        # 1. 构造 DataFrame 并保持顺序
        input_df = pd.DataFrame([input_values])[features_order]
        
        # 2. 执行用户自定义编码逻辑
        processed_df = input_df.copy()
        
        # A. 映射有序变量
        for col, mapping in mapping_dict.items():
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].map(mapping)
        
        # B. 映射二分类变量
        # Yes/No 映射
        binary_cols = [
            'DeafOrHardOfHearing', 'HadStroke', 'DifficultyWalking', 'PneumoVaxEver', 
            'DifficultyErrands', 'HadArthritis', 'HadKidneyDisease', 'HadAngina', 'ChestScan', 'HadCOPD'
        ]
        for col in binary_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].map({'Yes': 1, 'No': 0})
        
        # 性别映射 (用户逻辑：Female: 1, Male: 0)
        if 'Sex' in processed_df.columns:
            processed_df['Sex'] = processed_df['Sex'].map({'Female': 1, 'Male': 0})
        
        # 3. 缩放
        try:
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
        except Exception as e:
            st.error(f"预测过程中出错: {e}")

st.sidebar.markdown("""
### 编码逻辑说明
- **有序变量**：采用自定义映射字典。
- **二分类**：Yes=1, No=0。
- **性别**：Female=1, Male=0。
""")
