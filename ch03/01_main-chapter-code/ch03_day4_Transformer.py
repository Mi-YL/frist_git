import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, d_in=3
d_out = 2 # the output embedding size, d_out=2

# 下面，我们初始化三个权重矩阵；请注意，为了简化输出并便于示范，我们将 requires_grad=False,但如果我们要在模型训练中使用这些权重矩阵，应将 requires_grad=True，以便在训练过程中更新这些矩阵。
torch.manual_seed(123)
#固定随机种子确保可复现性
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
#初始化三个矩阵来存放
#不要求梯度降低了复杂度
query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value 
#点积计算
print(query_2)
keys = inputs @ W_key 
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
#中途检验下
keys_2 = keys[1] # Python starts index at 0
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)
#计算注意力跟query值的点积
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
#压缩函数, 有利于储存与比较
print(attn_weights_2)
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)