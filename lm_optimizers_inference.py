import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import os
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime

class PennTreebankDataset(Dataset):
    def __init__(self, data_path, seq_length=35):
        self.seq_length = seq_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read().replace('\n', '<eos>')
        
        words = text.split()
        word_counts = Counter(words)
        self.vocab = ['<unk>', '<eos>'] + [word for word, count in word_counts.most_common() if count > 2]
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        self.data = [self.word2idx.get(word, self.word2idx['<unk>']) for word in words]
        self.data = torch.tensor(self.data)
        
    def __len__(self):
        return len(self.data) - self.seq_length - 1
    
    def __getitem__(self, idx):
        return (self.data[idx:idx + self.seq_length],
                self.data[idx + 1:idx + self.seq_length + 1])

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_dim).to(device),
                torch.zeros(1, batch_size, self.hidden_dim).to(device))

class MetricsTracker:
    def __init__(self, save_dir="metrics"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.train_losses = []
        self.valid_losses = []
        self.perplexities = []
        self.batch_times = []
        self.epoch_times = []
        
        self.experiment_start_time = time.time()
        self.current_epoch_start_time = None
        
    def start_epoch(self):
        self.current_epoch_start_time = time.time()
        
    def end_epoch(self):
        epoch_time = time.time() - self.current_epoch_start_time
        self.epoch_times.append(epoch_time)
        
    def add_batch_time(self, batch_time):
        self.batch_times.append(batch_time)
        
    def add_metrics(self, train_loss, valid_loss, perplexity):
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.perplexities.append(perplexity)
        
    def save_metrics(self, optimizer_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics = {
            "optimizer": optimizer_name,
            "total_training_time": time.time() - self.experiment_start_time,
            "train_losses": self.train_losses,
            "valid_losses": self.valid_losses,
            "perplexities": self.perplexities,
            "epoch_times": self.epoch_times,
            "average_batch_time": np.mean(self.batch_times)
        }
        
        # Save metrics to JSON
        metrics_file = os.path.join(self.save_dir, f"metrics_{optimizer_name}_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot learning curves
        self.plot_learning_curves(optimizer_name, timestamp)
        
        return metrics_file
    
    def plot_learning_curves(self, optimizer_name, timestamp):
        plt.figure(figsize=(12, 8))
        
        # Plot training and validation loss
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Learning Curves - {optimizer_name}')
        plt.legend()
        
        # Plot perplexity
        plt.subplot(2, 1, 2)
        plt.plot(self.perplexities, label='Perplexity', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Perplexity over Epochs')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"learning_curves_{optimizer_name}_{timestamp}.png"))
        plt.close()

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs=10):
    metrics = MetricsTracker()
    best_valid_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        metrics.start_epoch()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()
            
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            hidden = model.init_hidden(batch_size, device)
            
            optimizer.zero_grad()
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            
            train_loss += loss.item()
            
            batch_time = time.time() - batch_start_time
            metrics.add_batch_time(batch_time)
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        metrics.end_epoch()
        
        # Validation
        valid_loss = evaluate(model, valid_loader, criterion, device)
        avg_train_loss = train_loss/len(train_loader)
        perplexity = np.exp(valid_loss)
        
        metrics.add_metrics(avg_train_loss, valid_loss, perplexity)
        
        print(f'Epoch: {epoch+1}')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}')
        print(f'Perplexity: {perplexity:.4f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Save all metrics at the end of training
    metrics_file = metrics.save_metrics(optimizer.__class__.__name__)
    print(f"Metrics saved to {metrics_file}")

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            hidden = model.init_hidden(batch_size, device)
            
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            total_loss += loss.item()
            
    return total_loss / len(data_loader)

def main():
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 200
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = PennTreebankDataset('ptb.train.txt')
    valid_dataset = PennTreebankDataset('ptb.valid.txt')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    
    model = LSTMLanguageModel(
        vocab_size=len(train_dataset.vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_model(model, train_loader, valid_loader, criterion, optimizer, device, EPOCHS)

if __name__ == '__main__':
    main()