# %%
# 2.3 カウントベースの手法

# %%
# 2.3.1 Python によるコーパスの下準備

# %%
text = 'You say goodby and I say hello.'
text = text.lower()
text = text.replace('.', ' .')
text

# %%
words = text.split(' ')
words

# %%
word_to_id = {}
id_to_word = {}
for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

# %%
id_to_word

# %%
word_to_id

# %%
id_to_word[1]

# %%
word_to_id['hello']

# %%
import numpy as np
corpus = [word_to_id[word] for word in words]
corpus = np.array(corpus)
corpus

# %%
def preprocess(text):
    words = text.lower().replace('.', ' .').split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            id = len(word_to_id)
            word_to_id[word] = id
            id_to_word[id] = word

    corpus = np.array([word_to_id[word] for word in words])

    return corpus, word_to_id, id_to_word

# %%
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

# %%
# 2.3.4 共起行列

# %%
from common.util import preprocess
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)

# %%
print(id_to_word)

# %%
C = np.array([
    [0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0],
], dtype=np.int32)

# %%
C[0]

# %%
C[4]

# %%
C[word_to_id['goodbye']]

# %%
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + 1
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

# %%
# 2.3.5 ベクトル間の類似度

# %%
def cos_similarity(x, y):
    nx = x / np.sqrt(np.sum(x**2))
    ny = y / np.sqrt(np.sum(y**2))
    return x @ y

# %%
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return x @ y

# %%
from common.util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
C = create_co_matrix(corpus, len(word_to_id))

cos_similarity(C[word_to_id['you']], C[word_to_id['i']])

# %%
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # クエリを取り出す
    if query not in word_to_id:
        print(f'{query} is not found')
        return

    print(f'\n[query] {query}')
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # コサイン類似度の算出
    vocab_size = len(word_to_id)
    similarity = np.array([
        cos_similarity(
            word_matrix[i],
            query_vec
        ) for i in range(vocab_size)
    ])

    # コサイン類似度の結果から、その値を高い順に出力
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f' {id_to_word[i]}: {similarity[i]}')
        count += 1
        if count >= top:
            return

# %%
x = np.array([100, -20, 2])
x.argsort()

# %%
(-x).argsort()

# %%
from common.util import preprocess, create_co_matrix, most_similar

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
C = create_co_matrix(corpus, len(id_to_word))
most_similar('you', word_to_id, id_to_word, C)

# %%
# 2.4 カウントベースの手法の改善

# %%
# 2.4.1 相互情報量

# %%
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt +=1
                if cnt % (total//100 + 1) == 0:
                    print(f'{(100*cnt/total)}%% done')
    
    return M

# %%
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
C = create_co_matrix(corpus, len(id_to_word))
W = ppmi(C)

np.set_printoptions(precision=3)
print('co-occurence matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)

# %%
# 2.4.3 SVD による次元削減

# %%
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
C = create_co_matrix(corpus, len(id_to_word))
W = ppmi(C)

U, S, V = np.linalg.svd(W)

# %%
C[0] # 共起行列

# %%
W[0] # PPMI 行列

# %%
U[0] # SVD

# %%
U[0, :2]

# %%
import matplotlib.pyplot as plt
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()

# %%
# 2.4.4 PTB データセット

# %%
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')

# %%
len(corpus)

# %%
corpus[:30]

# %%
id_to_word[0]

# %%
id_to_word[1]

# %%
id_to_word[2]

# %%
word_to_id['car']

# %%
word_to_id['happy']

# %%
word_to_id['lexus']

# %%
# 2.4.5 PTB データセットでの評価

# %%
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')

print('counting co-ocurrence...')
C = create_co_matrix(corpus, len(id_to_word), window_size=2)

print('calculating PPMI...')
W = ppmi(C, verbose=True)

print('calculating SVD...')
from sklearn.utils.extmath import randomized_svd
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

word_vecs = U[:, :wordvec_size]

for query in ['you', 'year', 'car', 'toyota']:
    most_similar(query, word_to_id, id_to_word, word_vecs)

# %%
