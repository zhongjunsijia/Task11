import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_size)
        weights = self.softmax(scores)
        context = torch.matmul(weights, V)
        return context

class RF_LSTM_Attention_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(RF_LSTM_Attention_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, 1, batch_first=True, dropout=0.2)
        self.lstm3 = nn.LSTM(hidden_size//2, hidden_size//4, 1, batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_size//4)
        self.fc1 = nn.Linear(hidden_size//4 * 2, 32)  # 拼接上下文向量和最后一个时间步的输出
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM层
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)

        # 注意力层
        context = self.attention(out)
        context = torch.sum(context, dim=1)  # 对时间步求和

        # 最后一个时间步的输出
        lstm_out = out[:, -1, :]

        # 拼接
        combined = torch.cat((context, lstm_out), dim=1)

        # 全连接层
        out = self.relu(self.fc1(combined))
        out = self.fc2(out)
        return out

def train_hybrid_model(train_data_dir, model_save_path):
    # 1. 加载数据
    X_train = np.load(f"{train_data_dir}/X_train_horizon1.npy")
    y_train = np.load(f"{train_data_dir}/y_train_horizon1.npy")
    X_val = np.load(f"{train_data_dir}/X_val_horizon1.npy")
    y_val = np.load(f"{train_data_dir}/y_val_horizon1.npy")
    X_test = np.load(f"{train_data_dir}/X_test_horizon1.npy")
    y_test = np.load(f"{train_data_dir}/y_test_horizon1.npy")

    # 2. 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 3. 定义模型参数
    input_size = X_train.shape[2]
    hidden_size = 128
    output_size = y_train.shape[1]
    num_layers = 3
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 64

    # 4. 初始化模型、损失函数和优化器
    model = RF_LSTM_Attention_Model(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 5. 训练模型
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0

        # 批量训练
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        # 计算平均训练损失
        train_loss /= len(X_train)

        # 验证模式
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        # 计算平均验证损失
        val_loss /= len(X_val)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1}")

    # 6. 评估模型
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()

    y_test = y_test.numpy()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RF-LSTM-Attention模型评估结果:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

if __name__ == "__main__":
    # 示例调用
    train_data_dir = "data/features"
    model_save_path = "models/rf_lstm_attention_model_horizon1.pth"
    train_hybrid_model(train_data_dir, model_save_path)