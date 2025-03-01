**MNIST手写数字分类 - PyTorch实现**  
**项目描述**  
项目实现了一个神经网络，用于MNIST数据集的手写数字分类。它使用了全连接层（Fully Connected Layers）、批归一化（Batch Normalization）以及Dropout来防止过拟合。训练过程中，使用了动态学习率调度（ReduceLROnPlateau）和L2正则化来提升模型的泛化能力。  
**文件结构**  
├── /MNIST                  # 数据集文件夹  
├── origin.py               # 原版程序（未优化）来自https://gitee.com/kongfanhe/pytorch-tutorial  
├── optimize.py             # 优化程序  
├── test.py                 # 测试文件  
├── verify.py               # 模型验证程序  
├── model.pth               # 模型文件（模型的状态字典）  
**代码解释**  
1.神经网络结构 (Net Class)   
该网络包含5个全连接层，使用批归一化和ReLU激活函数，最后通过Dropout减少过拟合风险。  
2.数据预处理  
使用torchvision.transforms对MNIST数据进行旋转和转换为Tensor。  
3.训练与测试  
使用Adam优化器和CrossEntropyLoss作为损失函数。  
训练过程中动态调整学习率，防止学习率过大导致的不稳定训练。  
4.可视化结果   
绘制训练和测试的准确率曲线。  
![c6c13ca47954a57d19f3e7aa55abc7e](https://github.com/user-attachments/assets/b2d109da-37f0-4819-9aed-0ff546e71790)  
测试集准确率  
![image](https://github.com/user-attachments/assets/096ad372-2175-4ba0-80a8-d049107fd409)
