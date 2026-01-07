import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

attn_scores = torch.empty(6, 6)
#建立个空表来储存相关联程度

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
        #一点点计算相关性并输入表格
print(attn_scores)
#事实上就是实现了两个单词之间的关联度列表输出
attn_scores_2 = inputs @ inputs.T
print(attn_scores_2)
#有简单的方法整合方法计算

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
#归一化处理
print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
#重复了上一个操作