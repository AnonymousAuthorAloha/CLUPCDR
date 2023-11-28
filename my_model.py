import torch
import torch.nn as nn
import torch.nn.functional as F

device=torch.device("cuda:0")

class Attention(nn.Module):
    def __init__(self, vector_dim, num_heads, num_layers, feedforward_dim, a_weight=1.0, b_weight=1.0):
        super(Attention, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=vector_dim, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=feedforward_dim).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)
        self.a_weight = a_weight
        self.b_weight = b_weight
    
    def forward(self, a, b):
        # 将向量a和b合并成一个序列
        # 形状为 [sequence_len, batch_size, vector_dim]
        a.to(device)
        b.to(device)
        sequence = torch.stack([a * self.a_weight, b * self.b_weight], dim=0).to(device)
        
        # 通过Transformer Encoder
        encoded_sequence = self.transformer_encoder(sequence).to(device)
        
        # 在Encoder的每个子层后面添加加权的向量a作为残差
        encoded_sequence += a * self.a_weight
        encoded_sequence.to(device)
        # 聚合策略：取平均或其他方式
        aggregated_vector = encoded_sequence.mean(dim=0)
        
        return aggregated_vector
    
class LookupEmbedding(torch.nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim).to(device)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim).to(device)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb

class Emcdr(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.mapping=torch.nn.Linear(emb_dim,emb_dim).to(device)

    def forward(self, src_emb):
        src_emb = self.mapping.forward(src_emb).to(device)
        return src_emb

class Augment(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.input_layer=nn.Linear(emb_dim,emb_dim*2).to(device)
        self.hidden_layer=nn.Linear(emb_dim*2,emb_dim*2).to(device)
        self.output_layer=nn.Linear(emb_dim*2,emb_dim).to(device)

    def forward(self, src_emb):
        # 应用第一个线性层和tanh激活函数
        src_emb=src_emb.to(device)
        x = torch.tanh(self.input_layer(src_emb))
        
        # 应用隐藏层和tanh激活函数
        x = torch.tanh(self.hidden_layer(x))
        
        # 应用输出层
        output = self.output_layer(x)
        
        return output

class Encoder(torch.nn.Module):
    def __init__(self, emb_dim, layers_num):
        super().__init__()
        self.layers_num = layers_num
        
        num_layer_dim=emb_dim*(2**(layers_num-2))
        # 输入层
        self.input_layer = nn.Linear(emb_dim, num_layer_dim).to(device)
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        
        for _ in range(layers_num - 2): # 减2是因为我们已经有了一个输入层和一个输出层
            if(num_layer_dim>emb_dim):
                if(num_layer_dim/2>emb_dim):
                    self.hidden_layers.append(nn.Linear(num_layer_dim, num_layer_dim//2).to(device))
                    num_layer_dim=num_layer_dim//2
                else:
                    self.hidden_layers.append(nn.Linear(num_layer_dim, emb_dim).to(device))
                    num_layer_dim=emb_dim
            else:
                self.hidden_layers.append(nn.Linear(emb_dim,emb_dim).to(device))                    
                num_layer_dim=emb_dim
        
        # 输出层
        self.output_layer = nn.Linear(num_layer_dim, emb_dim).to(device)
        
    def swish(self, x):
        return x * torch.sigmoid(x)
    
    def mish(self, x):
        return x * torch.tanh(F.softplus(x))
         
    def forward(self, x,av='tanh'): 
        if av=='relu':       
            x = F.relu(self.input_layer(x.to(device))).to(device)
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
            x = self.output_layer(x)
            return x
        elif av=='sigmoid':
            x = torch.sigmoid(self.input_layer(x.to(device))).to(device)
            for layer in self.hidden_layers:
                x = torch.sigmoid(layer(x))
            x = self.output_layer(x)
            return x
        elif av=='tanh':
            x = torch.tanh(self.input_layer(x.to(device))).to(device)
            for layer in self.hidden_layers:
                x = torch.tanh(layer(x))
            x = self.output_layer(x)
            return x
        elif av=='leaky_relu':
            x = F.leaky_relu(self.input_layer(x.to(device))).to(device)
            for layer in self.hidden_layers:
                x = F.leaky_relu(layer(x))
            x = self.output_layer(x)
            return x
        elif av=='swish':
            x = self.swish(self.input_layer(x.to(device))).to(device)
            for layer in self.hidden_layers:
                x = self.swish(layer(x))
            x = self.output_layer(x)
            return x
        elif av=='mish':
            x = self.mish(self.input_layer(x.to(device))).to(device)
            for layer in self.hidden_layers:
                x = self.mish(layer(x))
            x = self.output_layer(x)
            return x
        elif av=='prelu':
            prelu = torch.nn.PReLU().to(device)
            x = prelu(self.input_layer(x.to(device))).to(device)
            for layer in self.hidden_layers:
                x = prelu(layer(x))
            x = self.output_layer(x)
            return x
        elif av=='elu':
            x = F.elu(self.input_layer(x.to(device))).to(device)
            for layer in self.hidden_layers:
                x = F.elu(layer(x))
            x = self.output_layer(x)
            return x

# 示例：
# emb_dim = 128
# hidden_size = 256
# layers_num = 3
# model = MLP(emb_dim=emb_dim, hidden_size=hidden_size, layers_num=layers_num)
# print(model)
