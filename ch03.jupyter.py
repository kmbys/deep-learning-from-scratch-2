# %%
# 3章 word2vec

# %%
# 3.1 推論ベースの手法とニューラルネットワーク

# %%
# 3.1.3 ニューラルネットワークにおける単語の処理方法

# %%
import numpy as np

c = np.array([[1, 0, 0, 0, 0, 0, 0]]) # 入力
W = np.random.randn(7, 3) # 重み
h = c @ W # 中間ノード

# %%
W

# %%
h

# %%
import numpy as np
from common.layers import MatMul

c = np.array([[1, 0, 0, 0, 0, 0, 0]])
W = np.random.randn(7, 3)
h = MatMul(W).forward(c)

# %%
W

# %%
h

# %%
# 3.2 シンプルな word2vec

# %%
# CBOW モデルの推論処理

# %%
import numpy as np
from common.layers import MatMul

# サンプルのコンテキストデータ
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 1, 0, 0, 0, 0, 0]])

# 重みの初期化
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# レイヤの生成
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in) # 重みを共有していることに注意
out_layer = MatMul(W_out)

# 順伝播
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)

# %%
s

# %%
# 3.2.2 CBOW モデルの学習

# %%
# 3.3 学習データの準備

# %%
# 3.3.1 コンテキストデータとターゲット

# %%
from common.util import preprocess
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

# %%
corpus

# %%
id_to_word

# %%
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]

    contexts = [
        [
            corpus[idx + t] for t in range(-window_size, window_size + 1) if t != 0
        ] for idx in range(window_size, len(corpus) - window_size)
    ]

    return np.array(contexts), np.array(target)

# %%
contexts, target = create_contexts_target(corpus)

# %%
contexts

# %%
target

# %%
# 3.3.2 one-hot 表現への変換

# %%
from common.util import preprocess, create_contexts_target, convert_one_hot

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

contexts, target = create_contexts_target(corpus)
target = convert_one_hot(target, vocab_size=len(word_to_id))
contexts = convert_one_hot(contexts, vocab_size=len(word_to_id))

# %%
# 3.4 CBOW モデルの実装

# %%
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocalb_size, hidden_size):
        W_in = 0.01 * np.random.randn(vocalb_size, hidden_size).astype('f')
        W_out = 0.01 * np.random.randn(hidden_size, vocalb_size).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layaer0, self.in_layaer1, self.out_layer]
        self.params = [layer.params for layer in layers]
        self.grads = [layer.grads for layer in layers]

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:,0])
        h1 = self.in_layer1.forward(contexts[:,1])
        h = 0.5 * (h0 + h1)
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = 0.5 * self.out_layer.backward(ds)
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None

# %%
# 3.4.1 学習コードの実装

# %%
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import preprocess, create_contexts_target, convert_one_hot
from ch03.simple_cbow import SimpleCBOW

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size=1)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size=5)
trainer = Trainer(
    model=model,
    optimizer=Adam()
)

trainer.fit(contexts, target, max_epoch=1000, batch_size=3)
trainer.plot()

# %%
for word_id, word in id_to_word.items():
    print(word, model.word_vecs[word_id])

# %%
# 3.5 word2vec に関する補足

# %%
