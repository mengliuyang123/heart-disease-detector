# 心脏病发作风险检测系统

这是一个基于机器学习的心脏病发作风险检测系统，利用决策树、支持向量机 (SVM) 和神经网络三种模型进行预测，并通过 Streamlit 构建交互式前端界面。

## 项目目标

本项目旨在根据用户的健康数据，预测其心脏病发作的风险。鉴于心脏病发作预测的特殊性，模型优化重点放在提高 **Recall (召回率)** 和 **F1 分数**，以确保系统能够尽可能多地识别出潜在的风险患者。

## 数据集

项目使用 `heart_2022_with_nans.csv` 数据集，该数据集包含多项健康指标和生活方式信息。在数据预处理阶段，我们处理了缺失值，并对分类特征进行了编码，对数值特征进行了标准化。

## 模型

本项目训练并评估了以下三种机器学习模型：

1.  **决策树 (Decision Tree)**：
    *   **优化策略**：通过 `class_weight=\'balanced\'` 参数处理类别不平衡问题。
    *   **性能**：在测试集上，Recall 约为 0.74，F1 分数约为 0.30。

2.  **支持向量机 (SVM)**：
    *   **优化策略**：使用 `LinearSVC` 并在训练时应用 `class_weight=\'balanced\'`。
    *   **性能**：在测试集上，Recall 约为 0.74，F1 分数约为 0.34。

3.  **神经网络 (MLPClassifier)**：
    *   **优化策略**：对训练集进行下采样以平衡类别，以提高模型对少数类（心脏病发作）的识别能力。
    *   **性能**：在测试集上，Recall 约为 0.75，F1 分数约为 0.30。

## 文件结构

```
heart_disease_detector/
├── app.py                     # Streamlit 前端应用
├── preprocess_data.py         # 数据预处理脚本
├── train_models_fast.py       # 模型训练脚本
├── preprocessors.pkl          # 预处理器（Scaler 和 LabelEncoders）
├── dt_model.joblib            # 训练好的决策树模型
├── svm_model.joblib           # 训练好的 SVM 模型
├── mlp_model.joblib           # 训练好的神经网络模型
├── heart_2022_with_nans.csv   # 原始数据集
├── X_train.npy                # 训练集特征
├── X_test.npy                 # 测试集特征
├── y_train.npy                # 训练集标签
├── y_test.npy                 # 测试集标签
└── README.md                  # 项目说明文件
```

## 如何运行

1.  **克隆仓库**：
    ```bash
    git clone <仓库地址>
    cd heart_disease_detector
    ```

2.  **安装依赖**：
    ```bash
    pip install pandas scikit-learn streamlit joblib
    ```

3.  **运行 Streamlit 应用**：
    ```bash
    streamlit run app.py
    ```
    应用将在您的浏览器中打开，通常是 `http://localhost:8501`。

## 贡献

欢迎对本项目提出建议或贡献代码。
