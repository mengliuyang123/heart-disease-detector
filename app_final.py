import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 设置页面配置
st.set_page_config(page_title="心脏病发作检测系统 (精简版)", layout="wide")

# 加载模型和缩放器
@st.cache_resource
def load_resources():
    # 加载用户指定的模型和缩放器
    # 注意：这里假设文件存在于当前目录或指定路径
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"加载模型或缩放器失败: {e}")
        return None, None

model, scaler = load_resources()

st.title("🫀 心脏病发作风险检测系统")
st.markdown("""
本系统使用精简后的 **16 个核心特征** 进行预测。这些特征被证明对心脏病发作风险具有最强的预测能力。
""")

# 用户提供的特征列表 (不含目标变量 HadHeartAttack)
features = [
    'DeafOrHardOfHearing', 'HadStroke', 'HadDiabetes', 'DifficultyWalking',
    'PneumoVaxEver', 'AgeCategory', 'GeneralHealth', 'DifficultyErrands',
    'HadArthritis', 'HadKidneyDisease', 'Sex', 'HadAngina', 'ChestScan',
    'RemovedTeeth', 'PhysicalHealthDays'
]

# 辅助函数：手动编码 (匹配原始数据集的常见编码方式)
def encode_input(df):
    # 简单的 Yes/No 映射
    binary_map = {"No": 0, "Yes": 1}
    
    # 处理二元特征
    binary_cols = [
        'DeafOrHardOfHearing', 'HadStroke', 'DifficultyWalking', 'PneumoVaxEver', 
        'DifficultyErrands', 'HadArthritis', 'HadKidneyDisease', 'HadAngina', 'ChestScan'
    ]
    for col in binary_cols:
        df[col] = df[col].map(binary_map)
    
    # 处理性别
    df['Sex'] = df['Sex'].map({"Female": 0, "Male": 1})
    
    # 处理糖尿病 (简化处理)
    df['HadDiabetes'] = df['HadDiabetes'].map({
        "No": 0, "Yes": 1, 
        "No, pre-diabetes or borderline diabetes": 0, 
        "Yes, but female told only during pregnancy": 1
    })
    
    # 处理年龄分段 (映射为数值)
    age_map = {
        "Age 18 to 24": 0, "Age 25 to 29": 1, "Age 30 to 34": 2, "Age 35 to 39": 3,
        "Age 40 to 44": 4, "Age 45 to 49": 5, "Age 50 to 54": 6, "Age 55 to 59": 7,
        "Age 60 to 64": 8, "Age 65 to 69": 9, "Age 70 to 74": 10, "Age 75 to 79": 11,
        "Age 80 or older": 12
    }
    df['AgeCategory'] = df['AgeCategory'].map(age_map)
    
    # 处理总体健康状况
    health_map = {"Excellent": 0, "Very good": 1, "Good": 2, "Fair": 3, "Poor": 4}
    df['GeneralHealth'] = df['GeneralHealth'].map(health_map)
    
    # 处理牙齿移除情况
    teeth_map = {"None of them": 0, "1 to 5": 1, "6 or more, but not all": 2, "All": 3}
    df['RemovedTeeth'] = df['RemovedTeeth'].map(teeth_map)
    
    # 处理 HadCOPD (如果模型需要，虽然不在用户列表但在逻辑中可能有用)
    if 'HadCOPD' in df.columns:
        df['HadCOPD'] = df['HadCOPD'].map(binary_map)
        
    return df

# 主界面：输入表单
st.header("请输入个人健康指标")
cols = st.columns(3)
input_values = {}

# 按照用户提供的顺序排列输入控件
for i, col in enumerate(features + ['HadCOPD']): # 加上用户列表中提到的 HadCOPD
    with cols[i % 3]:
        if col in ['PhysicalHealthDays']:
            input_values[col] = st.number_input(f"{col}", min_value=0, max_value=30, value=0)
        elif col in ['Sex']:
            input_values[col] = st.selectbox(f"{col}", ["Female", "Male"])
        elif col in ['GeneralHealth']:
            input_values[col] = st.selectbox(f"{col}", ["Excellent", "Very good", "Good", "Fair", "Poor"])
        elif col in ['AgeCategory']:
            input_values[col] = st.selectbox(f"{col}", [
                "Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
                "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
                "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
                "Age 80 or older"
            ])
        elif col in ['RemovedTeeth']:
            input_values[col] = st.selectbox(f"{col}", ["None of them", "1 to 5", "6 or more, but not all", "All"])
        elif col in ['HadDiabetes']:
            input_values[col] = st.selectbox(f"{col}", ["No", "Yes", "No, pre-diabetes or borderline diabetes", "Yes, but female told only during pregnancy"])
        else:
            input_values[col] = st.selectbox(f"{col}", ["No", "Yes"])

# 预测按钮
if st.button("开始评估风险"):
    if model is None or scaler is None:
        st.error("模型未加载，无法预测。请确保 heart_disease_model.pkl 和 scaler.pkl 在正确位置。")
    else:
        # 构造 DataFrame
        input_df = pd.DataFrame([input_values])
        
        # 确保列顺序与训练时一致 (按照用户提供的列表顺序)
        ordered_features = [
            'DeafOrHardOfHearing', 'HadStroke', 'HadDiabetes', 'DifficultyWalking',
            'PneumoVaxEver', 'AgeCategory', 'GeneralHealth', 'DifficultyErrands',
            'HadArthritis', 'HadKidneyDisease', 'Sex', 'HadAngina', 'ChestScan',
            'RemovedTeeth', 'HadCOPD', 'PhysicalHealthDays'
        ]
        input_df = input_df[ordered_features]
        
        # 编码
        encoded_df = encode_input(input_df.copy())
        
        # 缩放
        scaled_data = scaler.transform(encoded_df)
        
        # 预测
        prediction = model.predict(scaled_data)[0]
        
        # 获取概率 (SVM 如果训练时开启了 probability=True)
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
            st.warning("根据您的健康指标，系统检测到较高的心脏病发作风险。建议您咨询专业医生。")
        else:
            st.success(f"### 评估结果：低风险")
            if prob is not None:
                st.write(f"风险概率: {prob:.2%}")
            st.info("您的健康指标显示心脏病发作风险较低。请继续保持健康的生活方式！")

st.sidebar.info("该系统基于 SVM 模型开发，使用了 16 个关键健康特征。")
