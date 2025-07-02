import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

# Устройство: GPU если доступен, иначе CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Модель: GTE-base (12 слоёв)
model_name = "thenlper/gte-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    output_hidden_states=True
).to(device).eval()

# Создание vocab head вручную
vocab_size = tokenizer.vocab_size
vocab_head = torch.nn.Linear(model.config.hidden_size, vocab_size, bias=False).to(device)

# Фиксированный слой (для данной модели 4-9 диапазон слоев без эмбеддинга, семантики и лингвистики)
FIXED_LAYER = 7


@torch.no_grad()
def get_probs(hidden_state):
    logits = vocab_head(hidden_state)
    probs = F.softmax(logits, dim=-1)
    print(
        f"[probs] min={probs.min().item():.6f}, max={probs.max().item():.6f}, any NaN: {torch.isnan(probs).any().item()}"
    )
    return probs


def kl_divergence(p, q, eps=1e-8):
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    log_ratio = torch.log(p / q)
    log_ratio[torch.isnan(log_ratio)] = 0.0
    log_ratio[torch.isinf(log_ratio)] = 0.0
    result = torch.sum(p * log_ratio)
    if torch.isnan(result) or torch.isinf(result):
        return torch.tensor(0.0, device=p.device)
    return result


@torch.no_grad()
def get_middle_layer_features(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    last_token_index = inputs["input_ids"].shape[1] - 1

    try:
        print(f"[Fluoroscopy] Используется фиксированный слой: {FIXED_LAYER}")
        return hidden_states[FIXED_LAYER][:, last_token_index, :].squeeze().float().cpu()

    except Exception as e:
        print(f"[Ошибка] {e}")
        fallback = min(FIXED_LAYER, len(hidden_states) - 1)
        print(f"[Резерв] Возврат слоя {fallback}")
        return hidden_states[fallback][:, last_token_index, :].squeeze().float().cpu()
