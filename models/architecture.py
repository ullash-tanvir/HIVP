import torch
from transformers import BertModel

class AttentionLayer(torch.nn.Module):
    """Custom attention layer for sequence representation"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        weights = self.attention(x)
        output = torch.sum(weights * x, dim=1)
        return output, weights

class ResidualBlock(torch.nn.Module):
    """Residual block with batch normalization and dropout"""
    def __init__(self, in_dim, out_dim, activation, dropout_rate):
        super().__init__()
        activations = {
            'relu': torch.nn.ReLU(), 
            'leaky_relu': torch.nn.LeakyReLU(),
            'gelu': torch.nn.GELU(), 
            'selu': torch.nn.SELU(), 
            'elu': torch.nn.ELU()
        }
        self.activation = activations[activation]
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.BatchNorm1d(out_dim),
            self.activation,
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(out_dim, out_dim),
            torch.nn.BatchNorm1d(out_dim),
        )
        self.shortcut = torch.nn.Linear(in_dim, out_dim) if in_dim != out_dim else torch.nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.layers(x)
        return self.activation(out + identity)

class EnhancedProtBert(torch.nn.Module):
    """Enhanced ProtBert model with attention and residual connections"""
    def __init__(self, config, arch_params):
        super().__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")
        
        # Freeze BERT parameters
        for p in self.bert.parameters():
            p.requires_grad = False
            
        # Architecture parameters
        self.hidden_dims = arch_params['hidden_dims']
        self.dropout_rate = arch_params['dropout_rate']
        self.activation = arch_params['activation']
        self.use_attention = arch_params['use_attention']
        self.num_heads = arch_params.get('num_heads', 1)
        
        # Attention mechanism
        if self.use_attention:
            self.attention_heads = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(
                    embed_dim=config.hidden_size, 
                    num_heads=self.num_heads,
                    dropout=self.dropout_rate
                ) for _ in range(self.num_heads)
            ])
            self.att_pool = AttentionLayer(config.hidden_size)

        # Feature extraction layers
        layers = []
        input_dim = config.hidden_size * (self.num_heads if self.use_attention else 1)
        for h in self.hidden_dims:
            layers.append(ResidualBlock(input_dim, h, self.activation, self.dropout_rate))
            input_dim = h
        self.feature_layers = torch.nn.Sequential(*layers)
        
        # Classification head
        self.classifier = torch.nn.Linear(self.hidden_dims[-1], config.num_labels)

    def forward(self, input_ids):
        with torch.no_grad():
            seq_out = self.bert(input_ids).last_hidden_state
            
        if self.use_attention:
            heads = []
            for att in self.attention_heads:
                perm = seq_out.permute(1, 0, 2)
                att_out, _ = att(perm, perm, perm)
                heads.append(att_out.permute(1, 0, 2))
            seq_out = torch.cat(heads, dim=2) if len(heads) > 1 else heads[0]
            pooled, _ = self.att_pool(seq_out)
        else:
            pooled = seq_out.mean(dim=1)
            
        feats = self.feature_layers(pooled)
        return self.classifier(feats)