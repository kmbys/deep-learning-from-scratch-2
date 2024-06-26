# %%
# 1章 ニューラルネットワークの復習

# %%
# 1.1 数学とPythonの復讐

# %%
# 1.1.1 ベクトルと行列

# %%
import numpy as np

# %%
x = np.array([1, 2, 3])

# %%
x.__class__

# %%
x.shape

# %%
x.ndim

# %%
W = np.array([[1, 2, 3], [4, 5, 6]])

# %%
W.shape

# %%
W.ndim


# %%
# 1.1.2 行列の要素ごとの演算

# %%
W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])
W + X

# %%
W * X

# %%
# 1.1.3 ブロードキャスト

# %%
A = np.array([[1, 2], [3, 4]])
A * 10

# %%
A = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
A * b

# %%
# 1.1.4 ベクトルの内積と行列の積

# %%
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)

# %%
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.dot(A, B)

# %%
# 1.2 ニューラルネットワークの推論の全体像

# %%
# 1.2.1 ニューラルネットワークの推論の全体図

# %%
import numpy as np
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
x = np.random.randn(10, 2)
h = np.dot(x, W1) + b1

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
a = sigmoid(h)

# %%
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(10, 2)
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

h = np.dot(x, W1) + b1
a = sigmoid(h)
s = np.dot(a, W2) + b2

# %%
# 1.2.2 レイヤとしてのクラス化と順伝搬の実装

# %%
import numpy as np

class Sigmoid:
    def __init__(self):
        self.params = []
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

# %%
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out

# %%
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
    
        self.params = [layer.params for layer in self.layers]

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

# %%
a = ['A', 'B']
a + ['C', 'D']

# %%
x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)

# %%
# 1.3 ニューラルネットワークの学習

# %%
# 1.3.4 計算グラフ

# %%
# 1.3.4.3 Repeat ノード

# %%
import numpy as np
D, N = 8, 7
x = np.random.randn(1, D) # 入力
y = np.repeat(x, N, axis=0) # forward
dy = np.random.randn(N, D) # 仮の勾配
dx = np.sum(dy, axis=0, keepdims=True) # backward

# %%
# 1.3.4.4 Sum ノード

# %%
import numpy as np
D, N = 8, 7
x = np.random.randn(N, D) # 入力
y = np.sum(x, axis=0, keepdims=True) # forward
dy = np.random.randn(1, D) # 仮の勾配
dx = np.repeat(dy, N, axis=0) # backward

# %%
# 1.3.4.5 MatMul ノード

# %%
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        self.x = x
        return np.dot(x, W)
    
    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

# %%
# 1.3.5 勾配の導出と逆伝搬の実装

# %%
# 1.3.5.1 Sigmoid レイヤ

# %%
class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# %%
# 1.3.5.2 Affine レイヤ

# %%
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out
    
    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, x.T)
        dW = np.dot(dout, W.T)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

# %%
# 1.3.6 重みの更新

# %%
class SGD:
    def __init(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

# %%
# 1.4 ニューラルネットワークで問題を解く

# %%
# 1.4.1 スパイラル・データセット

# %%
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x', x.shape)
print('t', t.shape)

# %%
# 1.4.2 ニューラルネットワークの実装

# %%
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = [
            Affine(
                0.01 * np.random.randn(input_size, hidden_size),
                np.zeros(hidden_size)
            ),
            Sigmoid(),
            Affine(
                0.01 * np.random.randn(hidden_size, output_size),
                np.zeros(output_size)
            )
        ]
        self.loss_layer = SoftmaxWithLoss()
        self.params = [layer.params for layer in self.layers]
        self.grads = [layer.grads for layer in self.layers]

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

# %%
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from ch01.two_layer_net import TwoLayerNet

# ハイパーパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# データの読み込み、モデルとオプティマイザの生成
x, t = spiral.load_data()
model = TwoLayerNet(2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# 学習で使用する変数
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 定期的に学習経過を出力
        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(f'| epoch {epoch + 1} | iter {iters + 1} | loss {avg_loss:.2f}')
            loss_list.append(avg_loss)
            total_loss = 0
            loss_count = 0

plt.plot(loss_list)

# %%
import numpy as np
np.random.permutation(10)

# %%
np.random.permutation(10)

# %%
from common.optimizer import SGD
from common.trainer import Trainer
from dataset import spiral
from ch01.two_layer_net import TwoLayerNet

x, t = spiral.load_data()

trainer = Trainer(
    model=TwoLayerNet(input_size=2, hidden_size=10, output_size=3),
    optimizer=SGD(lr=1.0),
)
trainer.fit(
    x=x,
    t=t,
    max_epoch=300,
    batch_size=30,
    eval_interval=10,
)
trainer.plot()

# %%
import numpy as np
a = np.random.randn(3)
a.dtype

# %%
b = np.random.randn(3).astype(np.float32)
b.dtype

# %%
c = np.random.randn(3).astype('f')
c.dtype
