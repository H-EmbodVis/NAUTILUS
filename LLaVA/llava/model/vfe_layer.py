import torch
import torch.nn as nn


class CrossAttentionNetwork(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionNetwork, self).__init__()
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Linear layer
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        self.linear_init()
    
    def forward(self, query, key, value):
        # query, key, value shape (sequence_length, batch_size, embedding_dim)
        attn_output, attn_weights = self.cross_attention(query, key, value)
        output = self.output_linear(attn_output)
        
        return output, attn_weights

    def linear_init(self):
        nn.init.kaiming_normal(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)
        nn.init.kaiming_normal(self.cross_attention.in_proj_weight)
        nn.init.zeros_(self.cross_attention.in_proj_bias)
        nn.init.kaiming_normal(self.cross_attention.out_proj.weight)
        nn.init.zeros_(self.cross_attention.out_proj.bias)

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        self.fc1 = torch.nn.Linear(input_size, hidden_sizes[0]) 
        self.fc2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = torch.nn.Linear(hidden_sizes[1], output_size) 

        self.relu = torch.nn.ReLU()

        for param in self.parameters():
            param.requires_grad = True
        
        self.weight_init()

    def forward(self, x):

        x = self.relu(self.fc1(x)) 
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def weight_init(self):
        torch.nn.init.kaiming_normal(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.kaiming_normal(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.kaiming_normal(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(self.fc3.bias)