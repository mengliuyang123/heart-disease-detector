import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# 设置页面配置
st.set_page_config(page_title="心脏病发作检测系统", layout="wide")

# 加载模型和预处理器
@st.cache_resource
def load_resources():
    dt = joblib.load('/home/ubuntu/dt_model.joblib')
    svm = joblib.load('/home/ubuntu/svm_model.joblib')
    mlp = joblib.load('/home/ubuntu/mlp_model.joblib')
    with open('/home/ubuntu/preprocessors.pkl', 'rb') as f:
        pre = pickle.load(f)
    return dt, svm, mlp, pre

dt_model, svm_model, mlp_model, preprocessors = load_resources()

st.title("🫀 心脏病发作风险检测系统")
st.markdown("""
本系统利用机器学习模型（决策树、SVM、神经网络）根据您的健康数据预测心脏病发作风险。
模型已针对 **Recall (召回率)** 和 **F1 分数** 进行了优化，以确保尽可能捕捉到潜在风险。
""")

# 侧边栏：模型选择
st.sidebar.header("模型设置")
selected_model_name = st.sidebar.selectbox("选择预测模型", ["决策树", "支持向量机 (SVM)", "神经网络 (MLP)"])

model_dict = {
    "决策树": dt_model,
    "支持向量机 (SVM)": svm_model,
    "神经网络 (MLP)": mlp_model
}
model = model_dict[selected_model_name]

# 主界面：输入表单
st.header("请输入您的健康信息")

# 动态生成输入字段
cols = st.columns(3)
input_data = {}

# 获取原始列名和编码器
feature_columns = preprocessors['columns']
encoders = preprocessors['encoders']

# 为了简化，我们只展示一些关键特征，或者为所有特征提供默认值
# 在实际应用中，我们会为每个特征提供输入控件
for i, col in enumerate(feature_columns):
    with cols[i % 3]:
        if col in encoders:
            options = encoders[col].classes_.tolist()
            input_data[col] = st.selectbox(f"{col}", options)
        elif col in ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']:
            input_data[col] = st.number_input(f"{col}", value=0.0 if 'Days' in col else 7.0 if 'Sleep' in col else 1.7 if 'Height' in col else 70.0 if 'Weight' in col else 24.0)
        else:
            input_data[col] = st.text_input(f"{col}", value="No")

# 预测按钮
if st.button("开始预测"):
    # 预处理输入数据
    input_df = pd.DataFrame([input_data])
    
    # 编码
    for col, le in encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col].astype(str))
            except ValueError:
                # 如果输入了未知类别，使用第一个类别作为默认
                input_df[col] = 0
    
    # 缩放
    input_scaled = preprocessors['scaler'].transform(input_df)
    
    # 预测
    prediction = model.predict(input_scaled)[0]
    # 概率 (如果模型支持)
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0][1]
    elif hasattr(model, "decision_function"):
        # SVM 使用 decision_function
        df_val = model.decision_function(input_scaled)[0]
        prob = 1 / (1 + np.exp(-df_val)) # Sigmoid 转换

    # 显示结果
    st.divider()
    if prediction == 1:
        st.error(f"### 预测结果：高风险 (风险概率: {prob:.2%})")
        st.warning("建议您咨询专业医生进行详细检查。")
    else:
        st.success(f"### 预测结果：低风险 (风险概率: {prob:.2%})")
        st.info("请继续保持健康的生活方式！")

    # 展示模型性能指标
    st.subheader("模型性能参考")
    metrics_df = pd.DataFrame({
        "指标": ["Recall (召回率)", "F1 分数", "优化目标"],
        "决策树": ["0.74", "0.30", "高召回"],
        "SVM": ["0.74", "0.34", "平衡"],
        "神经网络": ["0.75", "0.30", "高召回"]
    })
    st.table(metrics_df)
