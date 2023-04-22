import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义数据集
corpus = ["i love playing basketball",
          "he likes watching movies",
          "she hates eating vegetables",
          "they enjoy good music"]#corpus数据集

# 建立词表
word_to_idx = {}
idx_to_word = {}
for sentence in corpus:
    for word in sentence.split():#返回word列表
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)#哈希表
            idx_to_word[len(word_to_idx)-1]=word

# 构建数据集
data = []
for sentence in corpus:
    sentence_data = []
    for word in sentence.split():
        sentence_data.append(word_to_idx[word])
    data.append(sentence_data)

# 将数据转换为Tensor类型
data = torch.tensor(data, dtype=torch.long)

# 定义超参数
embedding_size = 10
hidden_size = 20
num_layers = 1
num_epochs = 200
batch_size = 2
learning_rate = 0.01

# 定义语言模型
class LanguageModel(nn.Module):#括号内的是继承的意思
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(LanguageModel, self).__init__()#super，父类的方法相对于子类来说是超方法，所以用super调用
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        #嵌入层，vocab是词汇表长度，embed是输出词向量长度，接受词id张量，返回词向量张量
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)#输出层，输出是一个词表长度的向量
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
#out=LSTM在每个时间步长上的输出张量，形状为(batch_size, seq_len, hidden_size)。
#hidden=LSTM最后一个时间步长的隐状态和细胞状态，作为下一个时间步长的输入，形状均为(num_layers, batch_size, hidden_size)。
#其中，batch_size指的是输入的样本数量，seq_len指的是序列的长度。
# 初始化模型和损失函数
model = LanguageModel(len(word_to_idx), embedding_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    hidden = None
    for i in range(0, len(data)-batch_size, batch_size):
        inputs = data[i:i+batch_size,:-1]#-1在python中为自动计算，前面省略默认为0，所以-1求得结尾
        targets = data[i:i+batch_size,1:]
        outputs, hidden = model(inputs, hidden)#通过模型类名来调用forward函数
        loss = criterion(outputs.view(-1, len(word_to_idx)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 测试模型
model.eval()#开启评估模式
hidden = None
input_seq = torch.tensor([word_to_idx['i'], word_to_idx['love']], dtype=torch.long).unsqueeze(0)
output_seq = []
with torch.no_grad():#上下文管理器，表示下文禁用梯度
    output, hidden = model(input_seq, hidden)
    _, predicted = torch.max(output.data, 2)
    output_seq.append(predicted)
    input_seq = predicted.squeeze(0).unsqueeze(0)#清空内容
    out=predicted.tolist()
    print(idx_to_word [out[0][0]])
    print(idx_to_word [out[0][1]])
