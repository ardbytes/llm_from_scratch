### Simplified self-attention

The goal is to transform input embeddings of a token into its corresponding
context vector.

A context vector can be interpreted as an enriched embedding
vector.

The operator `@` denotes matrix multiplication.

1. Input embeddings ( x.shape = 6x3 )
2. Attention scores ( a = x @ x.T; a.shape = 6x6 )
3. Attention weights ( w = softmax(a); w.shape = 6x6 )
4. Context vectors ( c = w @ x; c.shape = 6x3 )

### Self-attention with trainable weights

The goal is the same as the previous section. Only difference is, instead of
using the input vector directly to conpute attenstion scores, Weight matrices
are used to project the input embeddings into query, key, and value vectors. 
These vectors are then used to compute attention scores.

1. Input embeddings ( x.shape = 6x3 )
2. Key, Value, and Query Weight matrices with random values: Wk, Wv, Wq
3. Wk.shape = Wv.shape = Wq.shape = 3x2
4. Query vectors ( q = x @ Wq; q.shape = 6x2 )
5. Key vectors ( k = x @ Wk; k.shape = 6x2 )
6. Value vectors ( v = x @ Wv; v.shape = 6x2 )
7. Attention scores ( a = q @ k.T; a.shape = 6x6 )
8. Attention weights ( w = softmax(a); w.shape = 6x6 )
9. Context Vector ( c = w @ v; c.shape = 6x2 )
