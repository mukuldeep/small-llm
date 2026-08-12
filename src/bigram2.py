import torch
import torch.nn as nn
from torch.nn import functional as F

#hyper params
batch_size = 32 # no of independent seq to process in parallel
block_size = 8 # max context len for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


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
xb.to(device)
yb.to(device)
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

class BigramLanguageModel(nn.Module):
  def __init__(self,vocab_size):
    super().__init__()
    self.token_embedding_table  = nn.Embedding(vocab_size, vocab_size)
    #self.position_embedding_table = nn.Embedding(block_size, n_embed)
    #self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets = None):
    B,T = idx.shape
    logits = self.token_embedding_table(idx) # (B,T,C)

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
      logits, loss = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim = 1)
      idx_next = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)
xb.to(device)
yb.to(device)
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
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb,yb = get_batch('train')
  xb.to(device)
  yb.to(device)
  logits,loss = model(xb,yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()

  #print(loss.item())


#generate
context = torch.zeros((1,1), dtype=torch.long, device = device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

