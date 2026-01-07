import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 2nd input token is the query
# x = inputs.shape
attn_scores_2 = torch.empty(inputs.shape[0])
#建立一个未初始化的张量来记录注意力得分
# print("Attention Scores 2:", attn_scores_2)
# print(x)
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) 
    # 相似性度量计算attention分数
    # 从公式上看也就是点乘
# print(attn_scores_2)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum() 
#归一化,这里是属于加权式的归一化
# print(attn_scores_2.sum())
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())

# 然而，在实践中，使用softmax函数进行归一化更为常见，因为它能够更好地处理极端值，并且在训练过程中具有更理想的梯度特性，因此推荐使用。下面是一个简单的softmax函数实现，用于缩放并对向量元素进行归一化，使它们的和为1：
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
# print(torch.exp(attn_scores_2))
attn_weights_2_naive = softmax_naive(attn_scores_2)

# print("Attention weights:", attn_weights_2_naive)
# print("Sum:", attn_weights_2_naive.sum())

##用SoftMax做归一化, 处理好极端值
#有合理的梯度数据表现力
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print("Attention weights:", attn_weights_2)
# print("Sum:", attn_weights_2.sum())
#用torch优化过的softmax对边缘值也挺友好的

query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
#创造一个内容的零向量
print(query.shape)
print(context_vec_2)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
    #把不同内容的向量+起来

print(context_vec_2)