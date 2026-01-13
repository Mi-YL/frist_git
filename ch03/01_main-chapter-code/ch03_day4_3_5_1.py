import torch.nn as nn
import torch

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        #权重初始化

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T 
        #Query跟Key的计算 得出初始的分数传递到后面进行归一化操作
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        #直接基于注意力对于文本计算
        return context_vec

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1] # the input embedding size, d_in=3
d_out = 2 # the output embedding size, d_out=2
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
# Reuse the query and key weight matrices of the
# SelfAttention_v2 object from the previous section for convenience
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs) 
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
#用mask的数据重新算了一次注意力权重，将注意力权重中对应位置设置为负无穷大，然后进行softmax操作，得到最终的注意力权重
#隐藏未知信息的attention score最简单的方法是通过 PyTorch 的 tril 函数进行掩蔽，其中主对角线以下的元素（包括对角线本身）设置为 1，主对角线以上的元素设置为 0：
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
#Mask矩阵,直接保留Diagonal下部分的,上部分掩盖掉
print(mask_simple)
#然后，我们可以将注意力权重与这个mask相乘，以将对角线以上的注意力得分置为零：
masked_simple = attn_weights*mask_simple
print(masked_simple)
#简单的效果图
#如果在 softmax 之后进行掩蔽，它会破坏 softmax 所创建的概率分布。
# softmax 确保所有输出值的总和为 1。
# 如果在 softmax 之后进行掩蔽，就需要重新归一化输出，确保其总和为 1，这会使过程更加复杂，并可能带来意想不到的效果。
# 我们可以用以下方式确保所有的数据都是归一化的：
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
#掩码之后的softmax
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
#创建一个全1的三角,去上部分变成0
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
#有掩码的地方变为负无穷
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) 
# dropout rate of 50%丢包率doge
example = torch.ones(6, 6) 
# create a matrix of ones满的6*6矩阵被1包圆了
print(dropout(example))
#输出需要被放大相应的倍数,为了维持恒定的期望值
torch.manual_seed(123)
print(dropout(attn_weights))