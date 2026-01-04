from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))#验证下载并输出版本信息
tokenizer = tiktoken.get_encoding("gpt2")#初始化gpt2编码器
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})#输出分词的id,可以允许endoftext
print(integers)

strings = tokenizer.decode(integers)
#按照数字解码回去
print(strings)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)#读入了一个text并编码到enc_text里面
print(len(enc_text))

enc_sample = enc_text[50:]#从第五十一个开始向后

context_size = 4 #sliding windows4

x = enc_sample[:context_size]#开头四个
y = enc_sample[1:context_size+1]#第二个开始的四个

print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    #文本成输入 context,先输出有什么,然后输出下一个是什么编号
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)

for i in range(1, context_size+1):
    #文本成输入 context,先输出有什么,然后输出下一个是什么单词
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))