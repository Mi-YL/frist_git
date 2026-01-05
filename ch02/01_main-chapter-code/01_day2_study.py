from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken


class GPTDatasetV1(Dataset):
    #让GPT初始化一个类型
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})#id是文本内容编码过来的，默认情况下，tiktoken会将特殊令牌视为普通文本处理（或抛出错误），通过设置 allowed_special={"<|endoftext|>"} ，我们明确允许文本中的 <|endoftext|> 特殊令牌被识别为特殊令牌

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(#raw_text 中创建一个数据加载器 但是所批次
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)#创建一个数据加载器 dataloader ，它将 raw_text 文本转换为批次大小为 1 的数据样本

data_iter = iter(dataloader)#数据加载器 dataloader 转换为一个迭代器
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
#创建一个数据加载器 dataloader ，它将 raw_text 文本转换为批次大小为 8 的数据样本
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
print("\nInputs shape:\n", inputs.shape)
max_length = 4
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)#映射为tensor

token_embeddings = token_embedding_layer(inputs)#调用token_embedding_layer将输入inputs映射为对应的嵌入向量。
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
#目的是为输入序列中的每个位置生成一个向量,表明位置信息
pos_embeddings = pos_embedding_layer(torch.arange(max_length))#生成一个连续整数的1D tensor

print(pos_embeddings.shape)
input_embeddings = token_embeddings + pos_embeddings#特征是词语信息跟位置信息的结合
print(input_embeddings.shape)
