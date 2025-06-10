import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pprint
from copy import deepcopy

training_data = {
    "oi":            "ola como posso ajudar <end>",
    "quem e legal":  "maria <end>",
    "quem e maria":  "uma pessoa legal <end>",
    "maria e legal": "sim <end>",
    "quem e voce":   "sou seu assistence pessoal <end>"
}

def get_data_and_vocab():

    data_words = list(training_data.keys())
    target_words = list(training_data.values())

    vocabulary = set()
    for sentence in data_words + target_words:
        vocabulary.update(sentence.lower().split())

    vocabulary = sorted(vocabulary - {"<end>"})
    vocabulary.insert(0, "")
    vocabulary.append("<end>")

    word_to_ix = {word: ix for ix, word in enumerate(vocabulary)}
    ix_to_word = {ix: word for word, ix in word_to_ix.items()}

    return training_data, data_words, target_words, vocabulary, word_to_ix, ix_to_word

def words_to_tensor(seq_batch: list[str],
                    device: torch.device) -> torch.Tensor:
    tensors = [
        torch.tensor([word_to_ix[w] for w in seq.lower().split()],
                     dtype=torch.long,
                     device=device)
        for seq in seq_batch
    ]
    return pad_sequence(tensors, batch_first=True, padding_value=0)


def tensor_to_words(tensor: torch.Tensor) -> list[str]:
    sentences = []
    for seq in tensor.cpu().tolist():
        words = []
        for ix in seq:
            if ix == 0:
                continue
            w = ix_to_word.get(ix, "")
            words.append(w)
            if w == "<end>":
                break
        sentences.append(" ".join(words))
    return sentences

class SelfAttention(nn.Module):
    def __init__(self, embed_size, head_count):
        super().__init__()
        self.embed_size = embed_size
        self.head_count = head_count
        self.head_dim = embed_size // head_count
        assert self.head_dim * head_count == embed_size, \
            "Embed size deve ser divisível por head_count"

        self.qkv_proj = nn.Linear(embed_size, embed_size * 3)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        qkv = self.qkv_proj(x).view(batch_size, seq_length, 3,
                                     self.head_count, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(seq_length, seq_length,
                                     device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attention = F.softmax(scores, dim=-1)

        out = (attention @ v)                  
        out = out.transpose(1, 2).contiguous() 
        out = out.view(batch_size, seq_length, self.embed_size)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head_count, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_size, head_count)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * 4, embed_size),
        )

    def forward(self, x):
        attn = self.attention(x)
        x = self.norm1(x + self.dropout(attn))
        ff  = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, head_count, num_layers,
                 max_seq_len=100, dropout=0.1):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, embed_size)
        self.pos_embed  = nn.Parameter(torch.zeros(1, max_seq_len, embed_size))
        self.dropout    = nn.Dropout(dropout)
        self.layers     = nn.ModuleList([
            TransformerBlock(embed_size, head_count, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out     = nn.Linear(embed_size, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for n, p in self.named_parameters():
            if 'weight' in n and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x):
        seq_length = x.size(1)
        x = self.embed(x) + self.pos_embed[:, :seq_length, :]
        x = self.dropout(x)
        for block in self.layers:
            x = block(x)
        return self.fc_out(x)

def prepare_dataset(training_data, device):
    input_seqs = []
    label_seqs = []

    for src, tgt in training_data.items():
        src_idxs = [word_to_ix[w] for w in src.lower().split()]
        tgt_idxs = [word_to_ix[w] for w in tgt.lower().split()]

        seq_idxs = torch.tensor(src_idxs + tgt_idxs, dtype=torch.long, device=device)

        labels = torch.full((len(seq_idxs),), -100, dtype=torch.long, device=device)
        labels[len(src_idxs):] = torch.tensor(tgt_idxs, dtype=torch.long, device=device)

        input_seqs.append(seq_idxs)
        label_seqs.append(labels)

    input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    label_padded = pad_sequence(label_seqs, batch_first=True, padding_value=-100)
    return input_padded, label_padded

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }, path)

def load_checkpoint(model, optimizer, scheduler, path, device):
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint carregado, retomando do epoch {start_epoch}")
        return start_epoch
    except FileNotFoundError:
        print("Nenhum checkpoint encontrado, treinamento começará do zero.")
        return 0

def train(model, inputs, labels, vocab_size,
          epochs=750, lr=1e-3, print_every=25,
          checkpoint_path="checkpoint.pt", device="cpu"):

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    start_epoch = load_checkpoint(model, optimizer, scheduler,
                                  path=checkpoint_path, device=device)

    for epoch in range(start_epoch, epochs + 1):
        optimizer.zero_grad()
        logits = model(inputs)
        logits = logits[:, :-1, :].contiguous()
        shifted = labels[:, 1:].contiguous()

        loss = criterion(logits.view(-1, vocab_size), shifted.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % print_every == 0 or epoch == start_epoch:
            lr_cur = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}/{epochs} — Loss: {loss.item():.4f} — LR: {lr_cur:.6f}")

        if epoch % 100 == 0 or epoch == epochs:
            save_checkpoint(model, optimizer, scheduler, epoch, path=checkpoint_path)

    print("Treinamento concluído.")

def generate(model, prompt, word_to_ix, ix_to_word,
             max_len=20, temperature=0.7, top_k=5, device='cpu'):

    model.eval()
    tokens = [word_to_ix[w] for w in prompt.lower().split()]
    generated = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(generated)[0, -1, :] / temperature
            vals, idxs = torch.topk(logits, top_k)
            probs = F.softmax(vals, dim=-1)
            choice = torch.multinomial(probs, 1).item()
            nxt = idxs[choice].unsqueeze(0)
            generated = torch.cat([generated, nxt.unsqueeze(0)], dim=1)
            if ix_to_word[nxt.item()] == "<end>":
                break

    words = tensor_to_words(generated)[0].split()
    return " ".join(words[len(tokens):])

if __name__ == "__main__":
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_data, data_words, target_words, vocabulary, word_to_ix, ix_to_word = get_data_and_vocab()
    vocab_size = len(vocabulary)

    embed_size, head_count = 128, 4
    num_layers, dropout    = 4, 0.1
    max_seq_len            = 100

    input_batch, label_batch = prepare_dataset(training_data, device)

    model = MiniGPT(vocab_size, embed_size, head_count, num_layers,
                    max_seq_len, dropout).to(device)

    timestamp       = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"checkpoint_{timestamp}.pt"

    modelo_inicial = deepcopy(model)
    testes = ["oi","quem e legal","quem e maria","maria e legal","quem e voce"]
    inp_t = words_to_tensor(testes, device=device)
    print("Saídas ALEATÓRIAS (antes do treino):")
    pprint.pprint(tensor_to_words(modelo_inicial(inp_t).argmax(-1)))    

    print("Iniciando treinamento/fine-tuning...")
    train(
        model, input_batch, label_batch, vocab_size,
        epochs=700, lr=1e-3, print_every=50,
        checkpoint_path=checkpoint_path, device=device
    )

    print("\nGerações APÓS TREINAMENTO:")
    for p in ["oi","quem e legal","quem e maria","maria e legal","quem e voce"]:
        print(f"Prompt: '{p}' → {generate(model, p, word_to_ix, ix_to_word, device=device)}")
