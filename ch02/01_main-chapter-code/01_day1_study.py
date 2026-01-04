import os##导入os库    
import re
import urllib.request ##导入request库

if not os.path.exists("the-verdict.txt"):##如果文件不存在则创建，防止因文件已存在而报错
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read() ##读入文件按照utf-8
#print("Total number of character:", len(raw_text))##先输出总长度
#print(raw_text[:99])##输出前一百个内容

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    preprocessed = f.read() ##读入文件按照utf-8

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', preprocessed)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(preprocessed[:30])
#print(len(preprocessed))

#all_words = sorted(set(preprocessed))#从去掉重复的字符
#vocab_size = len(all_words)#计总的单词书
#print(vocab_size)
#print(all_words[:50])
#vocab = {token:integer for integer,token in enumerate(all_words)}##先把word进行编号,再按照单词或者标点为索引(有HashList那味道了)
all_tokens = sorted(list(set(preprocessed)))#set去重 list把处理后的重新变为列表,然后排序
all_tokens.extend(["<|endoftext|>", "<|unk|>"])#加上未知的表示

vocab = {token:integer for integer,token in enumerate(all_tokens)}
int_to_str = {i:s for s,i in vocab.items()} 
# print(int_to_str)
#遍历 enumerate(all_tokens) 中的每个元组 (integer, token)，以 token 作为键，integer 作为值创建字典条目。
# print(vocab)
#print(len(vocab.items()))
# for i, item in enumerate(list(vocab.items())[-5:]):#输出后五个内容与其标号
#     print(item)
# for i, item in enumerate(vocab.items()):
#     #print(item)
#     if i >= 50:
#         break ##遍历到前五十个

class SimpleTokenizerV1:#一个实例的名字创立
    def __init__(self, vocab): ## 初始化一个字符串
        self.str_to_int = vocab #单词到整数的映射
        self.int_to_str = {i:s for s,i in vocab.items()} 
        #方便解码,进行整数到词汇的反向映射
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)##正则化分词标点符号
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()## 去掉两端空格与全部的空句
        ]
        ids = [self.str_to_int[s] for s in preprocessed]##整理完的额字符串列表对应到id,从字典出来
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids]) #映射整数id到字符串。join是用前面那个(“ ”)联结成一个完整的字符串
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #使用正则表达式，去除标点符号前的多余空格
        # \s+匹配一个或者多个空白  \1 替换到匹配
        return text

class SimpleTokenizerV2:##版本2.0,启动!
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}#s为单词,i是key
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)#正则化按照标点分类
        preprocessed = [item.strip() for item in preprocessed if item.strip()]#去掉两头与所有空余句
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
            #遍历 preprocessed 中的每个 item，如果 item 存在于 self.str_to_int（即词汇表）中，就保留 item
            #如果不存在（即该单词或符号未定义在词汇表中），就替换为特殊标记 <|unk|>。
            #拓展:推导式（如列表推导式）是一种紧凑的语法，专门用于生成新列表（或其他容器）
            #与普通 for 循环相比，它更加简洁和高效，但逻辑复杂时可能会降低可读性。
        ]

        ids = [self.str_to_int[s] for s in preprocessed]#单词或标点映射为整数列表
        return ids
        

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


# tokenizer = SimpleTokenizerV1(vocab) #用vocab创造一个实例
# text = """"It's the last he painted, you know," 
#            Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text) #按照这个例子里的encode函数处理text
# print(ids)
# decoded_text = tokenizer.decode(ids)
# print(decoded_text)

# tokenizer1 = SimpleTokenizerV1(vocab)  ##用vocab创造一个实例
# text = "Hello, do you like tea. Is this-- a test?"
# tokenizer1.encode(text)
tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))#用句子分隔符链接两个句子
print(text) #跟第一个一样,但不会报错了
ids = tokenizer.encode(text) #按照这个例子里的encode函数处理text
print(ids)
decoded_text = tokenizer.decode(ids)
print(decoded_text)
