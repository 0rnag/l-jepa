import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from transformer import JEPA_Encoder, JEPA_Encoder_LM
import random
import signal
import sys
import matplotlib.pyplot as plt
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
num_epochs = config['finetune_num_epochs']
learning_rate = config['finetune_lr']
split_ratio = config['finetune_split_ratio']
dropout = config['finetune_dropout']
train_split = config['finetune_split_ratio']
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
# -----------------------------

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab.append('<mask>')
vocab_size = len(vocab)

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[:int(len(data)*train_split)]

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

encoder = JEPA_Encoder(vocab_size, n_embed, num_heads, block_size, num_layers, encoder_dim, device, dropout).to(device)

checkpoint_path = 'models/jepa_checkpoint_epoch_0_batch_6500.pth'
checkpoint = torch.load(checkpoint_path, weights_only=True)
#encoder.load_state_dict(checkpoint['encoder_state_dict'])
#predictor.load_state_dict(checkpoint['predictor_state_dict'])
#ema_encoder.load_state_dict(checkpoint['ema_encoder_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#encoder.load_state_dict(torch.load('models/jepa_encoder.pth', weights_only=True))
encoder.load_state_dict(checkpoint['encoder_state_dict'])

model = JEPA_Encoder_LM(encoder, vocab_size).to(device)

losses = []

interrupt = False
def signal_handler(sig, frame):
    global interrupt
    print('Interrupting')
    interrupt = True

signal.signal(signal.SIGINT, signal_handler)

for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(model.lm_head.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

try:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batches = (len(data) - block_size) // batch_size
        print(batches)
        exit()
        for _ in range(batches):
            x, y = get_batch()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y[:, -1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / batches
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if interrupt:
            print("Training interrupted")
            break
except KeyboardInterrupt:
    print("Training interrupted")
finally:
    torch.save(model.state_dict(), 'models/jepa_encoder_lm.pth')
    print("Model saved")

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('figures/finetuning_loss.png')
    plt.show()

