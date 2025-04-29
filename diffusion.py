from typing import Optional, Dict,  List
import enum
import math
import torch.nn.functional as F
import numpy as np
import random
import pyphen
from functools import lru_cache
import torch
from transformers import PreTrainedTokenizer, AutoModelForMaskedLM
import torch.nn as nn
from nn import mean_flat
from losses import normal_kl, discretized_gaussian_log_likelihood, discretized_text_log_likelihood

    

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":

        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar2(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar2(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1 - alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()  
    START_X = enum.auto()  
    EPSILON = enum.auto()  


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  
    RESCALED_MSE = (
        enum.auto()
    )  
    KL = enum.auto() 
    RESCALED_KL = enum.auto()  
    E2E_KL = enum.auto()
    E2E_MSE = enum.auto()
    E2E_Simple_MSE = enum.auto()
    E2E_Simple_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


from typing import List
import enum
import math
import torch
import torch.nn.functional as F
import numpy as np
import random
import pyphen
from transformers import PreTrainedTokenizer, AutoModelForMaskedLM
import torch.nn as nn

class BertDenoiser(nn.Module):
    def __init__(self, lm_model: AutoModelForMaskedLM, unfreeze_layers: int = 2):
        super().__init__()
        self.lm = lm_model
        self.lm.tie_weights()
        total = len(self.lm.bert.encoder.layer)
        for idx, layer in enumerate(self.lm.bert.encoder.layer):
            requires = (idx >= total - unfreeze_layers)
            for p in layer.parameters(): p.requires_grad = requires
        for p in self.lm.cls.predictions.decoder.parameters(): p.requires_grad = True
        if hasattr(self.lm.cls.predictions, 'bias'):
            self.lm.cls.predictions.bias.requires_grad = True

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor):

        return self.lm(input_ids=input_ids, attention_mask=attention_mask).logits

class GaussianDiffusion:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 lm_model: AutoModelForMaskedLM,
                 num_steps: int = 50,
                 target_syllables: List[int] = [5,7,5]):
        self.tokenizer = tokenizer
        self.denoiser = BertDenoiser(lm_model)
        self.num_timesteps = num_steps
        self.target_syllables = target_syllables
        self.syll = pyphen.Pyphen(lang='en_US')

    def count_syllables(self, word: str) -> int:
        clean = word.replace('##','')
        parts = self.syll.inserted(clean).split('-')
        return max(1, len(parts))

    def compute_importance(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        
        B,L = input_ids.shape
        imp = torch.zeros((B,L), device=input_ids.device)
        sep = self.tokenizer.sep_token_id
        for i in range(B):
            toks = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            sep_pos = [j for j,v in enumerate(input_ids[i].tolist()) if v==sep]
            segments = []
            start=1
            for idx,pos in enumerate(sep_pos[:2]): segments.append((start,pos,idx)); start=pos+1
            end = sep_pos[2] if len(sep_pos)>2 else L
            segments.append((start,end,2))
            for st,en,idx in segments:
                tgt = self.target_syllables[idx]
                sc = sum(self.count_syllables(tok) for tok in toks[st:en])
                for j in range(st,en): imp[i,j] = abs(sc - tgt)
        return imp

    def retention_mask(self, input_ids: torch.LongTensor, t: int) -> torch.BoolTensor:
        B,L = input_ids.shape
        device = input_ids.device
        imp = self.compute_importance(input_ids)
        mask = torch.zeros((B,L), dtype=torch.bool, device=device)
        specials = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
        mask |= (imp == 0)
        for tok in specials: mask |= (input_ids == tok)
        return mask

    def q_sample(self, input_ids: torch.LongTensor, t: int) -> torch.LongTensor:
        keep = self.retention_mask(input_ids, t)
        noisy = input_ids.clone()
        noisy[~keep] = self.tokenizer.mask_token_id
        return noisy

    def loss(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B,L = input_ids.shape
        t = random.randrange(self.num_timesteps)
        noisy = self.q_sample(input_ids, t)
        logits = self.denoiser(noisy, attention_mask, t)
        logits = logits.view(-1, logits.size(-1))
        target = input_ids.view(-1)
        prev_mask = self.retention_mask(input_ids, t-1) if t>0 else torch.ones_like(input_ids, dtype=torch.bool)
        masked_target = input_ids.masked_fill(~prev_mask, self.tokenizer.mask_token_id).view(-1)
        gamma = (self.num_timesteps - t) / self.num_timesteps
        ce = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        return gamma * ce(logits, target) + ce(logits, masked_target)


    @torch.no_grad()



    def p_sample_loop(self, batch_size: int, device: torch.device, temperature: float = 0.8):

        template = self.initialize_haiku_structure(batch_size, device)
        ha_id, mask_id = (
        self.tokenizer.convert_tokens_to_ids("ha"),
        self.tokenizer.mask_token_id,
        )
        x = template.clone()
        x[x == ha_id] = mask_id

        imp = self.compute_importance(template)[0] 
        L = imp.shape[0]
        specials = {
        self.tokenizer.cls_token_id,
        self.tokenizer.sep_token_id,
        self.tokenizer.pad_token_id,
        }
        idxs = [i for i in range(L) if template[0, i].item() not in specials]
        idxs_sorted = sorted(idxs, key=lambda i: float(imp[i]))

        reveal_masks = []
        T = self.num_timesteps

        for step, t in enumerate(reversed(range(T))):
            to_open = set(idxs_sorted[: min(len(idxs_sorted), step + 1)])
            mask = torch.zeros((batch_size, L), dtype=torch.bool, device=device)

            for pos in to_open:
                mask[:, pos] = True
            reveal_masks.append(mask)

        for mask_t, t in zip(reveal_masks, reversed(range(T))):
            attn = (x != self.tokenizer.pad_token_id).long()
            logits = self.denoiser(x, attn, t)
            probs = torch.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view_as(x)
            x = torch.where(mask_t, sampled, x)

        attn = (x != self.tokenizer.pad_token_id).long()
        final_logits = self.denoiser(x, attn, 0)
        final_tokens = torch.argmax(final_logits, dim=-1)
        x = torch.where(x == mask_id, final_tokens, x)

        return x

    def initialize_haiku_structure(self, batch_size: int, device: torch.device) -> torch.LongTensor:

        parts = self.target_syllables
        ha_id = self.tokenizer.convert_tokens_to_ids("ha")
        tokens = [self.tokenizer.cls_token_id]

        for cnt in parts:
            tokens += [ha_id] * cnt 
            tokens.append(self.tokenizer.sep_token_id)

        return torch.tensor(tokens, device=device).unsqueeze(0).repeat(batch_size, 1)

    def decode_to_haiku(self, ids: torch.LongTensor) -> List[str]:
        sep_tok = self.tokenizer.sep_token
        cls_tok = self.tokenizer.cls_token
        pad_tok = self.tokenizer.pad_token
        mask_tok = self.tokenizer.mask_token
        haikus = []
        for seq in ids:
            toks = self.tokenizer.convert_ids_to_tokens(seq.tolist(), skip_special_tokens=False)
            lines, curr = [], []
            for tkn in toks:
                if tkn == sep_tok:
                    if curr:
                        lines.append(self.tokenizer.convert_tokens_to_string(curr).strip())
                        curr = []
                    if len(lines) == len(self.target_syllables): break
                elif tkn in (cls_tok, pad_tok, mask_tok): continue
                else: curr.append(tkn)
            if curr and len(lines) < len(self.target_syllables):
                lines.append(self.tokenizer.convert_tokens_to_string(curr).strip())
            while len(lines) < len(self.target_syllables): lines.append("")
            haikus.append("\n".join(lines))
        return haikus






    
