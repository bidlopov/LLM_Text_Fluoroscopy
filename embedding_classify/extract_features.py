import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "EleutherAI/pythia-1.4b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    return_dict=True,
    output_hidden_states=True
).to(device).eval()


@torch.no_grad()
def get_probs(hidden_state, vocab_head):
    logits = vocab_head(hidden_state)
    probs = F.softmax(logits, dim=-1)
    print(
        f"[probs] min={probs.min().item():.6f}, max={probs.max().item():.6f}, any NaN: {torch.isnan(probs).any().item()}")
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
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # tuple of (layer, batch, seq_len, hidden_size)
    last_token_index = inputs["input_ids"].shape[1] - 1

    try:
        q0 = get_probs(hidden_states[0][:, last_token_index, :], model.embed_out)
        qN = get_probs(hidden_states[-1][:, last_token_index, :], model.embed_out)

        max_div, best_layer = -1, None
        for i in range(5, 20):  # for i in range(1, len(hidden_states) - 1):
            qi = get_probs(hidden_states[i][:, last_token_index, :], model.embed_out)
            div = (kl_divergence(q0, qi) + kl_divergence(qN, qi)).item()
            print(f"[DEBUG] Layer {i}, KL-div = {div:.6f}")
            if div > max_div:
                max_div = div
                best_layer = i

        if best_layer is None:
            raise ValueError("Не удалось выбрать лучший слой — KL-дивергенции равны?")

        print(f"[Fluoroscopy] Выбран слой: {best_layer}, KL-сумма: {max_div:.4f}")
        return hidden_states[best_layer][:, last_token_index, :].squeeze().float().cpu()

    except Exception as e:
        print(f"[Ошибка] {e}")
        print("[Резерв] Возврат фиксированного слоя 20")
        return hidden_states[20][:, last_token_index, :].squeeze().float().cpu()
