import re

text = "Hello, world. This, is a test."
text1 = "Hello, world. Is this-- a test?"
result = re.split(r'([,.]|\s)', text)##正则表达式按照空白字符与,.进行分割
result = [x for x in result if x.strip()]##去掉空字符串
result1 = re.split(r'([,.:;?_!"()\']|--|\s)', text1) ##就是按照常用的符号分割
result1 = [item.strip() for item in result1 if item.strip()]##去掉两端的空白字符 也是去掉了空字符串与仅包含空白字符的项
print(result)
print(result1)