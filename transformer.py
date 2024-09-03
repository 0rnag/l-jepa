# taken from https://github.com/karpathy/ng-video-lecture

import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedForward(nn.Module):

    def __init__(self, n_embed, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Head(nn.Module):

    def __init__(self, n_embed, head_size, block_size, dropout=0.2):
        super().__init__()

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_embed, num_heads, head_size, block_size, dropout=0.2):
        super().__init__()

        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Block(nn.Module):
    
    def __init__(self, n_embed, num_heads, block_size, dropout=0.2):
        super().__init__()
        
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(n_embed, num_heads, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embed, 4*n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPT(nn.Module):

    def __init__(self, vocab_size, n_embed, num_heads, block_size, num_layers, device, dropout=0.2):
        super().__init__()

        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class AttentionPool(nn.Module):

    def __init__(self, n_embed, encoder_dim):
        super().__init__()

        self.attention = nn.Linear(n_embed, 1)
        self.proj = nn.Linear(n_embed, encoder_dim)

    def forward(self, x):
        weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
        x = (x * weights.unsqueeze(-1)).sum(dim=1)
        return self.proj(x)

class JEPA_Encoder(nn.Module):
    # this should take in a sequence of tokens and output their encoded representations using defined transformer components
    # input: (batch_size, seq_len)
    # output: (batch_size, seq_len, embed_dim), potentially we pool this to get a single vector in (batch_size, encode_dim)
    def __init__(self, vocab_size, n_embed, num_heads, block_size, num_layers, encoder_dim, device, dropout=0.2):
        super().__init__()

        self.device = device
        self.block_size = block_size
        self.encoder_dim = encoder_dim
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        #self.pool = AttentionPool(n_embed, encoder_dim)


    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return x

class JEPA_Encoder_LM(nn.Module):
    def __init__(self, encoder, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.lm_head = nn.Linear(encoder.encoder_dim, vocab_size)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.lm_head(encoded[:, -1, :])


class JEPA_Predictor(nn.Module):
    # this should be a series of feed forward layers that take in the encoded representations and output a prediction
    def __init__(self, encoding_dim, dropout=0.2):
        super().__init__()

        self.ffwd_layers = nn.Sequential(
            FeedForward(encoding_dim, 4*encoding_dim, dropout),
            FeedForward(encoding_dim, 4*encoding_dim, dropout),
            # potentially add more layers here
        )

    def forward(self, x):
        # returns a prediction with shape (batch_size, encoding_dim)
        return self.ffwd_layers(x)

class JEPA_Predictor_Transformer(nn.Module):

    def __init__(self, n_embed, num_heads, block_size, num_layers, device, dropout=0.2):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(n_embed, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        return x



class JEPA_Decoder(nn.Module):
    # to be implemented later, this would be used on downstream tasks such as text generation
    def __init__(self):
        pass
