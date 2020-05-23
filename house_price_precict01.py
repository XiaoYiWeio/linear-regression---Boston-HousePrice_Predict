# 导入需要用到的package
import numpy as np
import matplotlib.pyplot as plt


# 读入训练数据
def load_data():
    datafile = 'housing.data'
    data = np.fromfile(datafile, sep=' ')
    # 前13项为影响因素,14项为价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)
    # reshape 变为[N,14]形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    # 拆分train_data（80%）test_data（20%）
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    # 归一化处理，axis =0 表示列
    maximums, minimus, avgs = training_data.max(axis=0), training_data.min(axis=0),\
        training_data.sum(axis=0) / training_data.shape[0]
    for i in range(feature_num):
        # print(maximums[i], minimus[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimus[i])
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data, maximums[-1], minimus[-1], avgs[-1]

# 模型设计
class Network(object):
    def __init__(self, num_of_weights):
        # 隐藏层神经元个数为权重数
        self.w1 = np.random.randn(num_of_weights,num_of_weights)
        print(self.w1.shape)
        self.w2 = np.random.randn(num_of_weights, 1) # 隐含层参数
        self.b1 = np.zeros(num_of_weights)
        self.b2 = 0

    def load_model_test(self):
        read_dictionary = np.load('my_model_best.npy').item()
        self.w1 = read_dictionary['w1']
        self.w2 = read_dictionary['w2']
        self.b1 = read_dictionary['b1']
        self.b2 = read_dictionary['b2']
        # 读取数据
        train_data, test_data, max, min, avg = load_data()
        # 测试集验证结果
        print("train_error: {:.3f}%".format(100 * (net.test(train_data))))
        # 测试集验证结果
        print("test_error: {:.3f}%".format(100 * (net.test(test_data))))

    def forward(self, x):       # 前向计算
        z1 = np.dot(x, self.w1) + self.b1
        z2 = np.dot(z1, self.w2) + self.b2
        return z1, z2

    def loss(self, z, y):       # 损失函数
        error = z - y
        cost = error * error
        cost = np.mean(cost)    # 取均值
        return cost

    def gradient_w2(self, x, y, regulation):     # 梯度下降法学习
        z1, z2 = self.forward(x)
        m = x.shape[0]
        gradient_w2 = (z2 - y)*z1
        gradient_w2 = np.mean(gradient_w2, axis=0)
        gradient_w2 = gradient_w2[:, np.newaxis] + regulation * self.w2 / m
        gradient_b2 = z2 - y
        gradient_b2 = np.mean(gradient_b2)
        return gradient_w2, gradient_b2

    def gradient_w1(self, x, y, regulation):
        z1, z2 = self.forward(x)
        m = x.shape[0]
        gradient_b1 = np.dot((z2 - y), self.w2.T)
        gradient_w1 = np.dot(gradient_b1.T, x) / m + self.w1 * regulation / m
        gradient_b1 = np.mean(gradient_b1)

        return gradient_w1, gradient_b1

    def update_w1(self, gradient_w1, gradient_b1, eta=0.01):
        self.w1 = self.w1 -eta * gradient_w1
        self.b1 = self.b1 - eta * gradient_b1

    def update_w2(self, gradient_w2, gradient_b2, eta=0.01):  # 更新参数
        self.w2 = self.w2 - eta * gradient_w2
        self.b2 = self.b2 - eta*gradient_b2

    def test(self, test_data):
        a = test_data[:, :-1]
        b = test_data[:, -1:]
        x, c = self.forward(a)
        c = c *(max- min) +avg
        b = b *(max- min) +avg
        return (np.mean(np.abs((c-b)/b)))


    def train(self, training_data,  testing_data, num_epoches, batch_size=10, eta=0.01,regulation=0.1):
        n = len(training_data)
        losses = []
        test_error = float('inf')
        for epoch_id in range(num_epoches):
            # 每轮迭代前将训练数据顺序随机打乱
            # 按每次取batch_size条数据方式取出
            np.random.shuffle(training_data)
            # mini_batches 取出时：list中每个元素含batch_size条数据：每条的类型为 batch_size * 14
            mini_batches = [training_data[k:k+batch_size]for k in range(0, n, batch_size)]
            # 双层循环
            for iter_id, mini_batch in enumerate(mini_batches):
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                k, a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w2, gradient_b2 = self.gradient_w2(x, y,regulation)
                gradient_w1, gradient_b1 = self.gradient_w1(x, y,regulation)
                self.update_w2(gradient_w2, gradient_b2, eta)
                self.update_w1(gradient_w1, gradient_b1, eta)
                losses.append(loss)
                print('Epoch:{:3d} / iter {:3d}, loss={:.4f}'.format(
                    epoch_id,iter_id,loss
                ))
            if epoch_id % 100 == 0:
                if test_error > self.test(test_data):
                    test_error = self.test(test_data)
                    print('Epoch:{:3d}, test_loss={:.4f}'.format(epoch_id, test_error))
                else:
                    return losses

        return losses





# 读取数据
train_data, test_data, max, min, avg = load_data()

# 创建网络
net = Network(13)
# 启动训练
losses = net.train(train_data, test_data, num_epoches=100, batch_size=100, eta=0.1, regulation=0.1)

# 画出损失函数变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()

# 测试集验证结果
print("train_error: {:.3f}%".format(100*(net.test(train_data))))
# 测试集验证结果
print("test_error: {:.3f}%".format(100*(net.test(test_data))))

result = {"w1": net.w1, "w2": net.w2, "b1": net.b1, "b2": net.b2}
np.save('my_model.npy', result)
'''
#读取：
# Load
read_dictionary = np.load('my_file.npy').item()
print(read_dictionary['w1'])
# displays "parameter:w1"
'''
net.load_model_test()



