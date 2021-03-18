import os
import re
str2 = '../data/train\03754\98253.png'

text = '你好！吃早饭了吗？再见。'
print(type(str))
out = str.split('\\')

str3 = r"{r}".format(str2)
print(str3)
out2 = re.split('\\\\|/', r"{r}".format{str2})
print(out)
print(out2)
