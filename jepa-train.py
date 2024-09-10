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
import yaml
import random
from tensorboardX import SummaryWriter
import os

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

writer = SummaryWriter(log_dir='runs/jepa_train')

# -----------------------------
n_embed = config['n_embed']
num_heads = config['num_heads']
num_layers = config['num_layers']
block_size = config['block_size']
encoder_dim = config['encoder_dim']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
learning_rate = config['lr']
train_split = config['train_split']
ema_decay = config['ema_decay']
ema_final = config['ema_final']
dropout = config['dropout']
min_masked_tokens = config['min_masked_tokens']
max_masked_tokens = config['max_masked_tokens']
predictor_num_layers = config['predictor_num_layers']
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
# -----------------------------

with open('data/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

vocab = sorted(list(set(text)))
vocab.append('<mask>')
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
        return len(self.data) - block_size + 1

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+block_size]
        #num_masked = random.randint(int(min_masked_tokens * block_size), int(max_masked_tokens * block_size))
        num_masked = int(max_masked_tokens * block_size)
        split_point = random.randint(0, block_size - num_masked)
        masked_chunk = torch.cat((chunk[0:split_point], torch.tensor(encode(['<mask>'])*num_masked), chunk[split_point+num_masked:]))
        return chunk, masked_chunk
    
train_dataset = JEPA_Dataset(train_data)
test_dataset = JEPA_Dataset(test_data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

encoder = JEPA_Encoder(vocab_size, n_embed, num_heads, block_size, num_layers, encoder_dim, device, dropout).to(device)
predictor = JEPA_Predictor_Transformer(n_embed, num_heads, block_size, predictor_num_layers, device, dropout).to(device)
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

interrupt_training = False
def signal_handler(sig, frame):
    global interrupt_training
    print('\nInterrupt received. Stopping training gracefully...')
    interrupt_training = True

signal.signal(signal.SIGINT, signal_handler)

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

def train():
    batch_losses = []
    epoch_losses = []

    global interrupt_training
    best_loss = float('inf')

    start_epoch = 0
    start_batch = 0
    global_batch_idx = 0
    # Check if there's a checkpoint to resume from
    checkpoints = [f for f in os.listdir('models') if f.startswith('jepa_checkpoint_epoch_')]
    if checkpoints:
        latest_checkpoint = max(checkpoints)
        start_epoch, start_batch, batch_losses = load_checkpoint(os.path.join('models', latest_checkpoint))
        global_batch_idx = start_epoch * len(train_dataloader) + start_batch
        print(f"Resuming from epoch {start_epoch}, batch {start_batch}")

    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0
            for batch_idx, (x1, x2) in enumerate(train_dataloader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue  # Skip batches that were already processed

                if interrupt_training:
                    raise KeyboardInterrupt

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

                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                epoch_loss += batch_loss

                writer.add_scalar('Loss/train', batch_loss, global_batch_idx)

                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
                    # Log GPU memory usage
                    if torch.cuda.is_available():
                        writer.add_scalar('GPU/memory_allocated', torch.cuda.memory_allocated(device), epoch * len(train_dataloader) + batch_idx)
                        writer.add_scalar('GPU/memory_cached', torch.cuda.memory_cached(device), epoch * len(train_dataloader) + batch_idx)
                    
                    if batch_idx % 1000 == 0:
                        save_checkpoint(epoch, batch_idx, batch_losses)


                global_batch_idx += 1

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

            
            save_checkpoint(epoch, len(train_dataloader) - 1, batch_losses)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(encoder.state_dict(), 'models/jepa_encoder_best.pth')
                torch.save(predictor.state_dict(), 'models/jepa_predictor_best.pth')

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving progress...")
    finally:
        plot_loss(batch_losses, epoch_losses)
        writer.close()
        save_checkpoint(epoch, batch_idx)
        print("Training complete. Models and checkpoints saved.")
    

def save_checkpoint(epoch, batch_idx, batch_losses):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'encoder_state_dict': encoder.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
        'ema_encoder_state_dict': ema_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_losses': batch_losses
    }
    torch.save(checkpoint, f"models/jepa_checkpoint_epoch_{epoch}_batch_{batch_idx}.pth")

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    predictor.load_state_dict(checkpoint['predictor_state_dict'])
    ema_encoder.load_state_dict(checkpoint['ema_encoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['batch_idx'], checkpoint['batch_losses']

def plot_loss(batch_losses, epoch_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot batch losses
    ax1.plot(batch_losses)
    ax1.set_title("JEPA Training Loss (Per Batch)")
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")

    # Plot epoch losses
    ax2.plot(epoch_losses)
    ax2.set_title("JEPA Training Loss (Per Epoch)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Loss")

    plt.tight_layout()
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"figures/jepa_loss_plot_{current_time}.png"
    
    plt.savefig(plot_filename)
    print(f"Loss plots saved as {plot_filename}")


if __name__ == "__main__":
    train()