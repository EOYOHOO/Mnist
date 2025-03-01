import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 定义神经网络结构
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 第一层全连接层，输入维度为28*28，输出维度为784
        self.fc1 = torch.nn.Linear(28 * 28, 784)
        self.bn1 = torch.nn.BatchNorm1d(784)  # 批归一化
        # 第二层全连接层，输入维度为784，输出维度为512
        self.fc2 = torch.nn.Linear(784, 512)
        self.bn2 = torch.nn.BatchNorm1d(512)  # 批归一化
        # 第三层全连接层，输入维度为512，输出维度为256
        self.fc3 = torch.nn.Linear(512, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)  # 批归一化
        # 第四层全连接层，输入维度为256，输出维度为128
        self.fc4 = torch.nn.Linear(256, 128)
        self.bn4 = torch.nn.BatchNorm1d(128)  # 批归一化
        # 第五层全连接层，输入维度为128，输出维度为10
        self.fc5 = torch.nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(0.6)  # Dropout，防止过拟合

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))  # 激活函数
        x = torch.nn.functional.relu(self.bn2(self.fc2(x)))  # 激活函数
        x = torch.nn.functional.relu(self.bn3(self.fc3(x)))  # 激活函数
        x = torch.nn.functional.relu(self.bn4(self.fc4(x)))  # 激活函数
        x = self.dropout(x)  # Dropout
        x = torch.nn.functional.log_softmax(self.fc5(x), dim=1)  # 输出层，使用log_softmax
        return x

# 定义一个函数，用于获取数据加载器
def get_data_loader(is_train):
    # 图像转换为Tensor
    to_tensor = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()])
    # 加载MNIST数据集
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=128, shuffle=True)

# 评估模型准确率
def evaluate(test_data, net, device):
    # 初始化正确预测的数量和总预测数量
    n_correct = 0
    n_total = 0
    # 不计算梯度
    with torch.no_grad():
        # 遍历测试数据
        for (x, y) in test_data:
            # 将数据移动到指定设备
            x, y = x.to(device), y.to(device)
            # 将输入数据展平并移动到指定设备
            outputs = net(x.view(-1, 28 * 28).to(device))
            # 遍历输出
            for i, output in enumerate(outputs):
                # 如果预测结果与真实标签相同，则正确预测数量加一
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                # 总预测数量加一
                n_total += 1
    # 返回正确预测数量与总预测数量的比值
    return n_correct / n_total

# 主函数
def main():
    # 检查是否有可用的GPU，如果没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取训练和测试数据集
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net().to(device)  # 将模型移到GPU
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)  # 添加L2正则化
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)  # 使用动态学习率调度

    print("Initial accuracy:", evaluate(test_data, net, device))

    train_accuracy = []
    test_accuracy = []

    # 训练模型
    epochs = 15  # 增加训练周期
    for epoch in range(epochs):
        net.train()
        for (x, y) in train_data:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = net(x.view(-1, 28 * 28).to(device))
            loss = torch.nn.CrossEntropyLoss()(output, y)  # 使用CrossEntropyLoss
            loss.backward()
            optimizer.step()

        # 每个epoch的训练和测试准确率
        train_acc = evaluate(train_data, net, device)
        test_acc = evaluate(test_data, net, device)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        print(f"Epoch {epoch + 1}: Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")

        # 调用ReduceLROnPlateau，根据测试集准确率调整学习率
        scheduler.step(test_acc)  # 传递测试集准确率
        # 保存模型
    torch.save(net.state_dict(), 'model.pth')
    print("Model saved to 'optimize_model.pth'")

    # 绘制训练和测试准确率曲线
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # 可视化预测结果
    net.eval()
    for n, (x, _) in enumerate(test_data):
        if n > 5:
            break
        x = x.to(device)
        predict = torch.argmax(net(x[0].view(-1, 28 * 28).to(device)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28).cpu())  # 将图像从GPU移回CPU
        plt.title(f"Prediction: {int(predict)}")
    plt.show()

if __name__ == "__main__":
    main()
