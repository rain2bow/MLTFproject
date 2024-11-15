from joblib import dump, load

import gradio as gr
import pandas as pd
from joblib import load
import os

# 获取所有.joblib文件
model_files = [f for f in os.listdir() if f.endswith('.joblib')]

TRANSACTION_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
# 加载模型字典
models = {model_file: load(model_file) for model_file in model_files}


def predict_fraud(model_name, step, amount, transaction_type):
    type_features = {f'type_{t}': 1 if t == transaction_type else 0 for t in TRANSACTION_TYPES}

    if amount < 0:
        gr.Info("错误：amount不能为负数")
    if step < 0:
        gr.Info("错误：step不能为负数")
    # 创建单条数据的DataFrame
    data = pd.DataFrame([[
        float(step),
        float(amount),
        type_features['type_CASH_IN'],
        type_features['type_CASH_OUT'],
        type_features['type_DEBIT'],
        type_features['type_PAYMENT'],
        type_features['type_TRANSFER']
    ]], columns=['step', 'amount', 'type_CASH_IN', 'type_CASH_OUT',
                 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'])

    # 获取选择的模型
    model = models[model_name]

    # 预测
    prediction = model.predict(data)[0]
    try:
        probability = model.predict_proba(data)[0][1]

        # 返回结果
        result = f"预测结果: {'欺诈' if prediction == 1 else '正常'}\n"
        result += f"欺诈概率: {probability:.2%}"
    except Exception as e:
        print(e)
        gr.Info("该模型不支持预测概率")
        result = f"预测结果: {'欺诈' if prediction == 1 else '正常'}\n"

    return result


# 创建界面
demo = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Dropdown(choices=model_files, label="选择模型"),
        gr.Number(label="Step"),
        gr.Number(label="Amount"),
        gr.Dropdown(choices=TRANSACTION_TYPES, label="交易类型")
    ],
    outputs=gr.Textbox(label="预测结果"),
    title="欺诈交易检测系统",
    description="请输入交易信息进行欺诈预测",
    examples=[
        [model_files[0], 268, 1070857.97, 'TRANSFER'],
        [model_files[0], 177, 129059.09, 'TRANSFER'],
    ]
)

# 启动服务
if __name__ == "__main__":
    demo.launch(share=True)