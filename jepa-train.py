import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from transformer import JEPA_Encoder, JEPA_Predictor, JEPA_Predictor_Transformer
import signal
import sys
import copy
from datetime import datetime

# -----------------------------
n_embed = 384
num_heads = 6
num_layers = 6
block_size = 256
encoder_dim = 384
batch_size = 64
num_epochs = 1
learning_rate = 3e-4
chunk_size = 512
split_ratio = 0.5
ema_decay = 0.996
ema_final = 1.0
dropout = 0.2
train_split = 0.95
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
# -----------------------------

def signal_handler(sig, frame):
    print('\nInterrupt received. Stopping training and plotting loss')
    global interrupt_training
    interrupt_training = True

signal.signal(signal.SIGINT, signal_handler)

with open('data/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[:int(len(data)*train_split)]
test_data = data[int(len(data)*train_split):]

class JEPA_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - chunk_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+chunk_size]
        split_point = int(chunk_size * split_ratio)
        return chunk[:split_point], chunk[split_point:]
    
train_dataset = JEPA_Dataset(train_data)
test_dataset = JEPA_Dataset(test_data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

encoder = JEPA_Encoder(vocab_size, n_embed, num_heads, block_size, num_layers, encoder_dim, device, dropout).to(device)
predictor = JEPA_Predictor_Transformer(n_embed, num_heads, block_size, num_layers, device, dropout).to(device)
print(sum(p.numel() for p in encoder.parameters())/1e6, 'M parameters for encoder')
print(sum(p.numel() for p in predictor.parameters())/1e6, 'M parameters for predictor')

ema_encoder = copy.deepcopy(encoder)
#ema_encoder = JEPA_Encoder(vocab_size, n_embed, num_heads, block_size, num_layers, encoder_dim, device, dropout).to(device)
#ema_encoder.load_state_dict(encoder.state_dict())

optimizer = optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=learning_rate)

criterion = nn.MSELoss(reduction='mean')

# momentum scheduler
ipe = len(train_dataloader)
ipe_scale = 1.0
momentum_scheduler = (ema_decay + i*(ema_final-ema_decay)/(ipe*num_epochs*ipe_scale) for i in range(int(ipe*num_epochs*ipe_scale)+1))

def evaluate_model():
    encoder.eval()
    predictor.eval()
    all_losses = []

    with torch.no_grad():
        for x1, x2 in test_dataloader:
            x1, x2 = x1.to(device), x2.to(device)

            target = ema_encoder(x2)
            encoded = encoder(x1)
            predicted = predictor(encoded)

            loss = criterion(predicted, target)
            all_losses.append(loss.item())

    encoder.train()
    predictor.train()
    return sum(all_losses) / len(all_losses)

losses = []
interrupt_training = False

try:
    for epoch in range(num_epochs):
        for batch_idx, (x1, x2) in enumerate(train_dataloader):
            if interrupt_training:
                print('Training interrupted. Proceeding to plot loss...')
                break

            x1, x2 = x1.to(device), x2.to(device)

            with torch.no_grad():
                target = ema_encoder(x2)
            
            encoded = encoder(x1)
            predicted = predictor(encoded)

            loss = criterion(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                m = next(momentum_scheduler)
                for ema_param, param in zip(ema_encoder.parameters(), encoder.parameters()):
                    ema_param.data.mul_(m).add_((1.-m) * param.detach().data)

            losses.append(loss.item())

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
                #print(f'Test loss: {evaluate_model():.4f}')

        if interrupt_training:
            break
except Exception as e:
    print(f"An error occurred during training: {e}")
    print("Proceeding to plot loss...")
finally:

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("JEPA Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"figures/jepa_loss_plot_{current_time}.png"
    
    plt.savefig(plot_filename)
    print(f"Loss plot saved as {plot_filename}")

    plt.show()

    torch.save(encoder.state_dict(), "models/jepa_encoder.pth")
    torch.save(predictor.state_dict(), "models/jepa_predictor.pth")

print("Training complete. Models saved.")