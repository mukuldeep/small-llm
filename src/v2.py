import torch
import torch.nn as nn
from torch.nn import functional as F

#hyper params
batch_size = 32 # no of independent seq to process in parallel
block_size = 8 # max context len for prediction
max_iters = 10000
eval_interval = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32



file_path =  '/content/drive/MyDrive/colab_datas/small_llm/input.txt'

# Read the whole file as a single string
with open(file_path, 'r') as file:
    text = file.read()

print(text[:1000])



# unique chars in the text
chars = sorted(set(text))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)


# char to int mapping
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: takes string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a String

print(encode("hii there!"))
print(decode(encode("hii there!")))

#create tensor from text
data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape, data.dtype)
print(data[0:100])

# split data to train and validate dataset
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
  context = x[:t+1]
  target = y[t]
  print(f"input={context} taget={target}")


## 
torch.manual_seed(1337)

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data)-block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])

  x,y = x.to(device),y.to(device)

  return x,y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)

print('targets:')
print(yb.shape)
print(yb)

print("-------")

for b in range(batch_size):
  for t in range(block_size):
    context = xb[b,:t+1]
    target = yb[b,t]
    print(f"input={context} taget={target}")

print(xb) # input to transformer

## eval loss
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      x,y = get_batch(split)
      logits, loss = model(x,y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


##

torch.manual_seed(1337)


class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias = False)
    self.query = nn.Linear(n_embed, head_size, bias = False)
    self.value = nn.Linear(n_embed, head_size, bias = False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)

    wei = q @ k.transpose(-2,-1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)

    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
  
  def forward(self, x):
    return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, n_embed),
      nn.ReLU(),
    )
  
  def forward(self, x):
    return self.net(x)

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table  = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    #self.sa_head = Head(n_embed)
    self.sa_heads = MultiHeadAttention(4,n_embed//4)
    self.ffwd = FeedForward(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets = None):
    B,T = idx.shape
    # logits = self.token_embedding_table(idx) # (B,T,C)
    token_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
    x = token_emb + pos_emb
    #x = self.sa_head(x)
    x = self.sa_heads(x)
    x = self.ffwd(x)
    logits = self.lm_head(x) # (B,T,C=vocab size)


    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits,loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim = 1)
      idx_next = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = BigramLanguageModel()
m = model.to(device)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

# idx = torch.zeros((1,1), dtype=torch.long)
# print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))

# pyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# train
for i in range(max_iters):
  if i % eval_interval ==0:
    losses = estimate_loss()
    print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb,yb = get_batch('train')
  # xb.to(device)
  # yb.to(device)
  logits,loss = model(xb,yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()

  #print(loss.item())


#generate
context = torch.zeros((1,1), dtype=torch.long, device = device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))




