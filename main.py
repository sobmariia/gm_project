import os
import random
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup
from diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType, get_named_beta_schedule, BertDenoiser
import random


class HaikuDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=32):
        df = pd.read_csv(csv_path)
        print(df.head())
        self.lines = df['haiku'].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        parts = self.lines[idx].split('[SEP]')
        if len(parts) == 3:
            text = (
                parts[0].strip() + f" {self.tokenizer.sep_token} " +
                parts[1].strip() + f" {self.tokenizer.sep_token} " +
                parts[2].strip()
            )
        else:
            text = self.lines[idx].replace('[SEP]', ' ')
        enc = self.tokenizer(
            text,
            padding='max_length', truncation=True,
            max_length=self.max_len, return_tensors='pt'
        )
        return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)


def main(
    csv_path='corrected.csv', model_name='google-bert/bert-base-uncased',
        epochs=25, batch_size=4, lr=1e-6, warmup_steps=200
):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
        )
    print(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    for p in lm_model.base_model.parameters():
        p.requires_grad = False
    for p in lm_model.cls.predictions.decoder.parameters():
        p.requires_grad = True
    if hasattr(lm_model.cls.predictions, 'bias'):
        lm_model.cls.predictions.bias.requires_grad = True

    betas = get_named_beta_schedule('cosine', num_diffusion_timesteps=50)
    betas = torch.tensor(betas, dtype=torch.float32, device=device)
    diffusion = GaussianDiffusion(
        tokenizer=tokenizer,
        lm_model=lm_model,
        num_steps=50,
        target_syllables=[5, 7, 5]  
    )
    loss_history = []

    denoiser = BertDenoiser(lm_model, unfreeze_layers=6).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, denoiser.parameters()), lr=lr)
    total_steps = (len(pd.read_csv(csv_path)) // batch_size + 1) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    dataset = HaikuDataset(csv_path, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    denoiser.train()
    for ep in range(1, epochs+1):
        tot_loss = 0.0
        for ids, attn in loader:
            ids = ids.to(device)
            attn = attn.to(device)
            loss = diffusion.loss(ids, attn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            tot_loss += loss.item() * ids.size(0)
        print(f"Epoch {ep}/{epochs} avg_loss={tot_loss / len(dataset):.4f}")
        epoch_loss = tot_loss / len(dataset)
        loss_history.append(epoch_loss)

    torch.save(lm_model.state_dict(), 'haiku_diffusion.pt')
    denoiser.eval()
    with torch.no_grad():
        gen_ids = diffusion.p_sample_loop(batch_size=1, device=device)

    haiku = diffusion.decode_to_haiku(gen_ids)[0]
    print("=== Generated Haiku ===")
    print(haiku)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()





if __name__ == '__main__':
    main()

