# %%
# 4章 word2vec の高速化

# %%
# 4.1 word2vecの改良➀

# %%
# 4.1.2 Embedding レイヤの実装

# %%
import numpy as np
W = np.arange(21).reshape(7, 3)
W

# %%
W[2]

# %%
W[5]

# %%
idx = np.array([1, 0, 3, 0])
W[idx]

# %%
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backword(self, dout):
        dW, = self.grads
        dW[...] = 0

        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]
        return None

# %%
# 4.2 word2vec の改良➁

# %%
# 4.2.4 多値分類から二値分類へ（実装編）

# %%
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
    
    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        self.cache = (h, target_W)
        return out
    
    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0]), 1
        dtarget_W = dout * h
        self.embed.backword(dtarget_W)
        dh = dout * target_W
        return dh

# %%
# 4.2.6 Negative Sampling のサンプリング手法

# %%
import numpy as np
np.random.choice(10)

# %%
np.random.choice(10)

# %%
words = ['you', 'say', 'goodbye', 'I', 'hello', '.']
np.random.choice(words)

# %%
np.random.choice(words, size=5)

# %%
np.random.choice(words, size=5, replace=False)

# %%
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
np.random.choice(words, p=p)

# %%
p = [0.7, 0.29, 0.1]
new_p = np.power(p, 0.75)
new_p /= np.sum(new_p)
new_p

# %%
from ch04.negative_sampling_layer import UnigramSampler

corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
power = 0.75
sample_size = 2

sampler = UnigramSampler(corpus, power, sample_size)

target = np.array([1, 3, 0])
negative_sample = sampler.get_negative_sample(target)
negative_sample


# %%
# 4.2.7 Negative Sampling の実装

# %%
from common.layers import SigmoidWithLoss

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.params = [layer.params for layer in self.embed_dot_layers]
        self.grads = [layer.grads for layer in self.embed_dot_layers]

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例のフォワード
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 負例のフォワード
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh

# %%
# 4.3 改良版 word2vec の学習

# %%
# 4.3.1 CBOW モデルの実装

# %%
import numpy as np
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        W_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype('f')
        W_out = 0.01 * np.random.randn(vocab_size, hidden_size).astype('f')

        self.in_layers = [Embedding(W_in) for i in range(2 * window_size)]
        self.ns_loss = NegativeSamplingLoss(W_out, corpus)

        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
            h *= 1 / len(self.in_layers)
            loss = self.ns_loss.forward(h, target)
            return loss

    def backword(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backword(dout)
        return None

# %%
# 4.3.2 CBOW モデルの学習コード

# %%
from common import config
from common.np import *
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from ch04.cbow import CBOW
from common.util import create_contexts_target, to_cpu
from dataset import ptb

# データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data('train')

contexts, target = create_contexts_target(corpus, window_size=5)

# モデルなどの生成
model = CBOW(
    vocab_size=len(word_to_id),
    hidden_size=100,
    window_size=5,
    corpus=corpus,
)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 学習開始
trainer.fit(contexts, target, max_epoch=10, batch_size=100)

# 後ほど利用できるように、必要なデータを保存
word_vecs = model.word_vecs
params = {
    'word_vecs': word_vecs.astype(np.float16),
    'word_to_id': word_to_id,
    'id_to_word': id_to_word,
}
with open('cbow_params.pkl', 'wb') as f:
    pickle.dump(params, f, -1)

# %%
# 4.3.2 CBOW モデルの評価

# %%
from common.util import most_similar
import pickle

with open('cbow_params.pkl', 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

most_similar('you', word_to_id, id_to_word, word_vecs)

# %%
most_similar('year', word_to_id, id_to_word, word_vecs)

# %%
most_similar('car', word_to_id, id_to_word, word_vecs)

# %%
most_similar('toyota', word_to_id, id_to_word, word_vecs)

# %%
from common.util import analogy

analogy('man', 'king', 'woman', word_to_id, id_to_word, word_vecs)

# %%
analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)

# %%
analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)

# %%
analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)

# %%
analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)
