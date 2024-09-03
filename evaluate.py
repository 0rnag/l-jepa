import torch
import torch.nn.functional as F
from transformer import JEPA_Encoder, JEPA_Encoder_LM
import random
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# -----------------------------
n_embed = config['n_embed']
num_heads = config['num_heads']
num_layers = config['num_layers']
block_size = config['block_size']
encoder_dim = config['encoder_dim']
batch_size = config['batch_size']
learning_rate = config['lr']
ema_decay = config['ema_decay']
ema_final = config['ema_final']
dropout = config['dropout']
train_split = config['train_split']
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
# -----------------------------

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab.append('<mask>')
vocab_size = len(vocab)

encoder = JEPA_Encoder(vocab_size, n_embed, num_heads, block_size, num_layers, encoder_dim, device, dropout).to(device)
encoder.load_state_dict(torch.load('models/jepa_encoder.pth', weights_only=True))

model = JEPA_Encoder_LM(encoder, vocab_size).to(device)
model.load_state_dict(torch.load('models/jepa_encoder_lm.pth', weights_only=True))

encoder.eval()
model.eval()

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
test_data = data[int(len(data)*train_split):]

def get_batch():
    ix = torch.randint(len(test_data) - block_size, (batch_size,))
    x = torch.stack([test_data[i:i+block_size] for i in ix])
    y = torch.stack([test_data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def generate_text(seed_text, max_new_tokens=100):
    model.eval()
    context = seed_text.unsqueeze(0).to(device)
    generated = context.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(context)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
        if generated.size(1) > block_size:
            context = context[:, -block_size:]

    return decode(generated[0].tolist())

ix = random.randint(0, len(data)-block_size)
seed_text = data[ix:ix+block_size]
generated_text = generate_text(seed_text)
print(generated_text[500:])