from flask import Flask, request, jsonify
import numpy as np
import torch
import joblib
import os

app = Flask(__name__)

# 确保模型目录存在
os.makedirs("models", exist_ok=True)

# 定义模型类
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = torch.nn.Linear(hidden_size, 32)
        self.fc2 = torch.nn.Linear(32, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class Attention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_size)
        weights = self.softmax(scores)
        context = torch.matmul(weights, V)
        return context

class RF_LSTM_Attention_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(RF_LSTM_Attention_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = torch.nn.LSTM(input_size, hidden_size, 1, batch_first=True, dropout=0.2)
        self.lstm2 = torch.nn.LSTM(hidden_size, hidden_size//2, 1, batch_first=True, dropout=0.2)
        self.lstm3 = torch.nn.LSTM(hidden_size//2, hidden_size//4, 1, batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_size//4)
        self.fc1 = torch.nn.Linear(hidden_size//4 * 2, 32)
        self.fc2 = torch.nn.Linear(32, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        context = self.attention(out)
        context = torch.sum(context, dim=1)
        lstm_out = out[:, -1, :]
        combined = torch.cat((context, lstm_out), dim=1)
        out = self.relu(self.fc1(combined))
        out = self.fc2(out)
        return out

# 加载模型
def load_models():
    models = {}
    try:
        # 加载随机森林模型
        models["rf"] = joblib.load("models/rf_model_horizon1.pkl")

        # 加载LSTM模型
        input_size = 13  # 特征数量
        hidden_size = 64
        output_size = 1
        lstm_model = LSTMModel(input_size, hidden_size, output_size)
        lstm_model.load_state_dict(torch.load("models/lstm_model_horizon1.pth", map_location=torch.device('cpu')))
        lstm_model.eval()
        models["lstm"] = lstm_model

        # 加载RF-LSTM-Attention模型
        hidden_size = 128
        hybrid_model = RF_LSTM_Attention_Model(input_size, hidden_size, output_size)
        hybrid_model.load_state_dict(torch.load("models/rf_lstm_attention_model_horizon1.pth", map_location=torch.device('cpu')))
        hybrid_model.eval()
        models["rf_lstm_attention"] = hybrid_model

    except Exception as e:
        print(f"加载模型失败: {e}")
    return models

models = load_models()

# 加载标准化器
try:
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    print(f"加载标准化器失败: {e}")
    scaler = None

# 预测API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        station_id = data.get("station_id")
        model_name = data.get("model")
        features = np.array(data.get("features")).reshape(1, 24, -1)  # 假设输入形状：(1, 24, n_features)

        # 选择模型
        model = models.get(model_name)
        if not model:
            return jsonify({"error": "Model not found"}), 404

        # 预测
        if model_name == "rf":
            # RF需要展平输入
            features_flat = features.reshape(1, -1)
            prediction = model.predict(features_flat)
        else:
            # LSTM和RF-LSTM-Attention使用PyTorch模型
            features_tensor = torch.tensor(features, dtype=torch.float32)
            with torch.no_grad():
                prediction = model(features_tensor).numpy()

        return jsonify({
            "station_id": station_id,
            "prediction": prediction.tolist()[0],
            "model": model_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 健康检查API
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": list(models.keys())})

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")