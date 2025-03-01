import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Net(torch.nn.Module):

    # 初始化函数
    def __init__(self):
        # 调用父类的初始化函数
        super().__init__()
        # 定义第一个全连接层，输入维度为28*28，输出维度为64
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        # 定义第二个全连接层，输入维度为64，输出维度为64
        self.fc2 = torch.nn.Linear(64, 64)
        # 定义第三个全连接层，输入维度为64，输出维度为64
        self.fc3 = torch.nn.Linear(64, 64)
        # 定义第四个全连接层，输入维度为64，输出维度为10
        self.fc4 = torch.nn.Linear(64, 10)

    # 前向传播函数
    def forward(self, x):
        # 将输入x通过第一层全连接层，并使用ReLU激活函数
        x = torch.nn.functional.relu(self.fc1(x))
        # 将输入x通过第二层全连接层，并使用ReLU激活函数
        x = torch.nn.functional.relu(self.fc2(x))
        # 将输入x通过第三层全连接层，并使用ReLU激活函数
        x = torch.nn.functional.relu(self.fc3(x))
        # 将输入x通过第四层全连接层，并使用log_softmax激活函数
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        # 返回输出x
        return x


# 定义一个函数，用于获取数据加载器
def get_data_loader(is_train):
    # 将数据转换为张量
    to_tensor = transforms.Compose([transforms.ToTensor()])
    # 加载MNIST数据集
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    # 返回数据加载器，批量大小为15，并打乱数据
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net, device):
    # 初始化正确预测的数量和总预测数量
    n_correct = 0
    n_total = 0
    # 不计算梯度
    with torch.no_grad():
        # 遍历测试数据
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)  # 将数据转移到GPU
            outputs = net.forward(x.view(-1, 28 * 28).to(device))  # 将输入也移到GPU
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU
    print(f"Using device: {device}")

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net().to(device)  # 将模型移到GPU

    print("initial accuracy:", evaluate(test_data, net, device))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_accuracies = []  # 用于存储每个 epoch 的训练集准确率
    test_accuracies = []  # 用于存储每个 epoch 的测试集准确率

    for epoch in range(2):
        net.train()  # 切换到训练模式
        correct_train = 0
        total_train = 0

        for (x, y) in train_data:
            x, y = x.to(device), y.to(device)  # 将数据转移到GPU
            net.zero_grad()
            output = net.forward(x.view(-1, 28 * 28).to(device))  # 将输入移到GPU
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()

            # 计算训练集准确率
            for i, output in enumerate(output):
                if torch.argmax(output) == y[i]:
                    correct_train += 1
                total_train += 1

        # 计算训练集准确率
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # 计算测试集准确率
        test_accuracy = evaluate(test_data, net, device)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # 绘制训练集和测试集准确率对比图
    plt.plot(range(1, 3), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, 3), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy Comparison')
    plt.legend()
    plt.show()

    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        x = x.to(device)  # 将数据转移到GPU
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28).to(device)))  # 前向传播在GPU上执行
        plt.figure(n)
        plt.imshow(x[0].view(28, 28).cpu())  # 绘图时移回CPU
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
