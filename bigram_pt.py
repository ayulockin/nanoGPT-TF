# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)

# Hyperparameters
batch_size = 256
block_size = 8
learning_rate = 1e-3
split_percentage = 0.9
device = "cuda" if torch.cuda.is_available() else "cpu"
train_iter = 1000
eval_iter = 200
eval_interval = 10
n_embd = 32
head_size = 16

# Data
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Generate Vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)
print("".join(chars))

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}


def encode_text(string_input):
    return [stoi[c] for c in string_input]


def decode_ints(integer_input):
    return "".join([itos[i] for i in integer_input])


data = torch.tensor(encode_text(text), dtype=torch.long)
n = int(split_percentage*len(data))
train_data = data[:n]
val_data = data[n:]

print("Length of train and test data: ", len(train_data), len(val_data))


def get_batch(split):
    # generate batch of (x, y)
    data = train_data if split=="train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, 16)
        q = self.query(x) # (B, T, 16)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

        wei = wei.masked_fill(self.tril==0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v

        return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embd = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_embd = self.positional_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = token_embd + pos_embd # (B, T, n_embd)

        x = self.sa_head(x)

        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
    
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, 1+T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:,-1,:] # (B, C)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32

for iter in range(train_iter):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']}, val loss {losses['val']}")

    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode_ints(m.generate(idx, 500)[0].tolist()), sep="\n")
