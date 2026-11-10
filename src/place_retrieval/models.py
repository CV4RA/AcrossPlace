"""The tested SigLIP2 LoRA and residual projections, plus BLIP ITM scoring."""
import torch
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoProcessor, BlipForImageTextRetrieval, BlipProcessor


class ResidualProjection(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, dimension, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, value):
        return value + self.linear(value)


class HeadPair(nn.Module):
    def __init__(self, dimension, residual=True):
        super().__init__()
        self.image = ResidualProjection(dimension) if residual else nn.Identity()
        self.text = ResidualProjection(dimension) if residual else nn.Identity()

    def encode(self, value, modality):
        return F.normalize(getattr(self, modality)(value.float()), dim=-1)


def load_siglip(directory, training, device):
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    is_blip = config.model_type == "blip"
    constructor = BlipForImageTextRetrieval if is_blip else AutoModel
    backbone = constructor.from_pretrained(directory, local_files_only=True, use_safetensors=True)
    if config.model_type not in ("siglip", "siglip2", "clip", "blip"):
        raise ValueError("Expected a supported paired encoder")
    processor_class = BlipProcessor if is_blip else AutoProcessor
    processor = processor_class.from_pretrained(directory, local_files_only=True, use_fast=False)
    backbone.requires_grad_(False)
    targets = []
    towers = training.get("adapt_towers", ["vision", "text"])
    if any(t not in ("vision", "text") for t in towers):
        raise ValueError("Unknown adaptation tower")
    if training.get("lora_targets") == "all-linear":
        targets = [name for name, module in backbone.named_modules() if isinstance(module, nn.Linear)]
    for tower, cfg in (("vision_model", backbone.config.vision_config),
                       ("text_model", backbone.config.text_config)):
        if training.get("lora_targets") == "all-linear" or tower.replace("_model", "") not in towers:
            continue
        count = training["lora_last_layers"]
        if count < 1 or count > cfg.num_hidden_layers:
            raise ValueError("lora_last_layers is outside the backbone")
        tower_targets = [f"{tower}.encoder.layers.{layer}.self_attn.{projection}"
                       for layer in range(cfg.num_hidden_layers - count, cfg.num_hidden_layers)
                       for projection in ("q_proj", "v_proj")]
        targets.extend(tower_targets)
    for name in targets:
        backbone.get_submodule(name)
    if targets:
        lora = LoraConfig(r=training["lora_rank"], lora_alpha=training["lora_alpha"],
                          lora_dropout=training.get("lora_dropout", 0.0),
                          use_rslora=training.get("use_rslora", False), bias="none", target_modules=targets)
        backbone = inject_adapter_in_model(lora, backbone)
    backbone = backbone.to(device)
    dimension = config.image_text_hidden_size if is_blip else (getattr(config, "projection_dim", None) or config.text_config.hidden_size)
    heads = HeadPair(dimension, training.get("residual_heads", True)).to(device)
    return backbone, processor, heads


def adapter_state(backbone):
    return get_peft_model_state_dict(backbone) if getattr(backbone, "peft_config", None) else {}


def restore_coarse(backbone, heads, state):
    """Accept both release checkpoints and the original experimental LoRA files."""
    heads.load_state_dict(state["heads"])
    if state["lora"]:
        set_peft_model_state_dict(backbone, state["lora"])
    restored = adapter_state(backbone)
    if set(restored) != set(state["lora"]):
        raise ValueError("Adapter tensor names do not match the model")
    for name, value in state["lora"].items():
        if not torch.equal(value.cpu(), restored[name].detach().cpu()):
            raise ValueError(f"Adapter tensor did not restore exactly: {name}")


def raw_features(backbone, inputs, modality):
    if backbone.config.model_type == "blip":
        if modality == "image":
            hidden = backbone.vision_model(pixel_values=inputs, return_dict=True).last_hidden_state[:, 0]
            return backbone.vision_proj(hidden)
        tokens = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        hidden = backbone.text_encoder(**tokens, return_dict=True).last_hidden_state[:, 0]
        return backbone.text_proj(hidden)
    if modality == "image":
        value = backbone.get_image_features(pixel_values=inputs)
    else:
        value = backbone.get_text_features(**inputs)
    return value if isinstance(value, torch.Tensor) else value.pooler_output


def context_length(backbone, training):
    return min(backbone.config.text_config.max_position_embeddings,
               training.get("text_length", backbone.config.text_config.max_position_embeddings))


def load_blip(directory, device):
    backbone = BlipForImageTextRetrieval.from_pretrained(
        directory, local_files_only=True, use_safetensors=True
    ).requires_grad_(False).to(device).eval()
    processor = BlipProcessor.from_pretrained(directory, local_files_only=True, use_fast=False)
    return backbone, processor
