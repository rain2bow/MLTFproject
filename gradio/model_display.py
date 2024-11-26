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
        gr.Info("amount cannot be negative!")
    if step < 0:
        gr.Info("step cannot be negative!")
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
    result = ""
    # 预测
    prediction = model.predict(data)[0]
    try:
        if prediction==1 or prediction==0:
            probability = model.predict_proba(data)[0][1]

            # 返回结果
            result = f"prediction result: {'fraud' if prediction == 1 else 'non-fraud'}\n"
            result += f"fraud probability: {probability:.2%}"
        else:
            probability = model.predict(data)[0][0]
            prediction = (probability > 0.15).astype(int)
            result = f"prediction result: {'fraud' if prediction == 1 else 'non-fraud'}\n"
            result += f"fraud probability: {probability:.2%}"
    except Exception as e:
        print(e)
        gr.Info(f"{e}")
    return result


# 创建界面
demo = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Dropdown(choices=model_files, label="choose a model"),
        gr.Number(label="Step"),
        gr.Number(label="Amount"),
        gr.Dropdown(choices=TRANSACTION_TYPES, label="transaction type"),
    ],
    outputs=gr.Textbox(label="prediction result"),
    title="fraud prediction system",
    description="type single transaction info to predict",
    examples=[
        [model_files[1], 268, 1070857.97, 'TRANSFER'],
        [model_files[3], 283, 186134.83, 'CASH_IN'],
    ]
)

# 启动服务
if __name__ == "__main__":
    demo.launch(share=True)