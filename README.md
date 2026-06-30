MNIST 手写数字识别
一个基于 TensorFlow/Keras 和 Flask 的 MNIST 手写数字识别 Web Demo。项目使用卷积神经网络训练 MNIST 数据集，并提供一个浏览器画板，用户可以手写数字并获得模型预测结果。
功能
使用 CNN 模型识别 0-9 的手写数字
基于 Flask 提供本地 Web 服务
前端使用 HTML Canvas 绘制数字
通过 /predict 接口提交画板图片并返回预测结果
仓库中包含已训练好的 model.h5，可直接运行推理服务
项目结构
.
|-- app.py                  # Flask 应用，加载模型并提供预测接口
|-- model.py                # 训练 MNIST CNN 模型并保存为 model.h5
|-- model.h5                # 已训练好的模型文件
|-- static/
|   `-- script.js           # 画板绘制、清空和预测请求逻辑
`-- templates/
    `-- index.html          # Web 页面模板
环境要求
建议使用 Python 3.9+。
主要依赖：
pip install flask tensorflow pillow numpy
快速开始
克隆仓库：
git clone https://github.com/MapleSugarMochi/MNIST.git
cd MNIST
安装依赖：
pip install flask tensorflow pillow numpy
启动 Web 应用：
python app.py
打开浏览器访问：
http://127.0.0.1:5000
在画板上手写一个数字，点击识别按钮即可查看预测结果。
重新训练模型
如果需要重新训练模型，运行：
python model.py
脚本会自动下载 MNIST 数据集，训练一个简单的卷积神经网络，并将训练后的模型保存为：
model.h5
训练完成后，再运行 python app.py 即可使用新的模型进行预测。
模型结构
model.py 中定义的模型结构如下：
Conv2D，32 个卷积核，ReLU 激活
MaxPooling2D
Conv2D，64 个卷积核，ReLU 激活
MaxPooling2D
Flatten
Dense，64 个神经元，ReLU 激活
Dense，10 个神经元，Softmax 输出
模型使用：
优化器：Adam
损失函数：Sparse Categorical Crossentropy
评价指标：Accuracy
训练轮数：5 epochs
接口说明
POST /predict
提交一张 Base64 编码的图片，返回预测数字。
请求体示例：
{
  "image": "data:image/png;base64,..."
}
响应示例：
{
  "prediction": 7
}
说明
前端画板使用黑色背景和白色笔迹。后端在预测前会将图片转为灰度图、缩放到 28x28，并进行反色和归一化处理，使其更接近 MNIST 数据集的输入格式。
许可证
本项目使用 MIT License。详情请查看 LICENSE 文件。
