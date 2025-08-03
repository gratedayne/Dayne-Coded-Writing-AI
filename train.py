import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from model import MiniGPT
from tokenizer import SimpleTokenizer

class LyricsDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=32):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = []
        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - seq_len):
                self.data.append(tokens[i:i+seq_len+1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

def train():
    texts = [
        "My lines are mixed carving outlines they carry every hit",
        "Butterflies watching eyes fold secrets in the wings",
        # Add more of your own writing here
    ]

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)

    dataset = LyricsDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    vocab_size = len(tokenizer.vocab) + 1
    model = MiniGPT(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "minigpt_trained.pth")

if __name__ == "__main__":
    train()