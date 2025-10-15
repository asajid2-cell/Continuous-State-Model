"""
Wrapper around the base Llama trunk with LoRA adapters and quantization setup.

Implements loading of a quantized causal LM, optional LoRA adapters, and exposes
forward outputs required by the delta-driven learning loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import logging

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


def _resolve_dtype(name: Optional[str]) -> torch.dtype:
    """Map a string dtype name to a torch dtype, defaulting to bfloat16."""

    if not name:
        return torch.bfloat16
    try:
        return getattr(torch, name)
    except AttributeError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


@dataclass
class LoraSettings:
    """Configuration container for LoRA adapters."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Sequence[str] = field(
        default_factory=lambda: (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    )


@dataclass
class TrunkConfig:
    """Configuration values required to instantiate the trunk."""

    base_name: str
    revision: str = "main"
    quantization: Optional[str] = "bitsandbytes-4bit"
    tokenizer_name: Optional[str] = None
    padding_side: Optional[str] = "left"
    device: Optional[str] = None
    device_map: Optional[str] = "auto"
    torch_dtype: Optional[str] = "float16"
    cache_dir: Optional[str] = None
    lora: Optional[LoraSettings] = None


class StreamingTrunk(torch.nn.Module):
    """
    Streaming capable Llama wrapper with LoRA support.

    Responsibilities per Phase A:
      * Load the requested checkpoint with quantization.
      * Attach LoRA adapters to targeted modules.
      * Expose features and logits required for prediction and consistency losses.
    """

    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model: Optional[torch.nn.Module] = None
        self._dtype = _resolve_dtype(cfg.torch_dtype)
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    def load(self) -> None:
        """Load weights and move model to target device."""

        load_kwargs: Dict[str, Any] = {
            "revision": self.cfg.revision,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
            "low_cpu_mem_usage": True,
        }

        logger.info("Loading base model '%s' (revision=%s)", self.cfg.base_name, self.cfg.revision)

        if self.cfg.cache_dir:
            logger.debug("Using cache directory: %s", self.cfg.cache_dir)
            load_kwargs["cache_dir"] = self.cfg.cache_dir

        use_cuda = torch.cuda.is_available()
        want_bnb4 = (self.cfg.quantization == "bitsandbytes-4bit")
        bnb_ok = False
        if want_bnb4 and use_cuda:
            try:
                import bitsandbytes as _  # noqa: F401
                bnb_ok = True
            except Exception as exc:  # pragma: no cover
                logger.warning("bitsandbytes not available; falling back to non-quantized load: %s", exc)

        if want_bnb4 and use_cuda and bnb_ok:
            logger.debug("Applying 4-bit quantization via bitsandbytes")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = self.cfg.device_map or "auto"
        else:
            if not use_cuda:
                logger.warning("CUDA not available; loading model on CPU with float32.")
                load_kwargs["torch_dtype"] = torch.float32
                load_kwargs["device_map"] = "cpu"
            else:
                logger.debug("Using torch dtype %s without quantization", self._dtype)
                load_kwargs["torch_dtype"] = self._dtype
                load_kwargs["device_map"] = self.cfg.device_map or "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_name,
            **load_kwargs,
        )

        self._load_tokenizer()

        if self.cfg.lora:
            logger.info(
                "Attaching LoRA adapters (r=%s, alpha=%s, dropout=%s) to modules: %s",
                self.cfg.lora.r,
                self.cfg.lora.alpha,
                self.cfg.lora.dropout,
                ", ".join(self.cfg.lora.target_modules),
            )
            lora_cfg = LoraConfig(
                r=self.cfg.lora.r,
                lora_alpha=self.cfg.lora.alpha,
                lora_dropout=self.cfg.lora.dropout,
                bias="none",
                target_modules=list(self.cfg.lora.target_modules),
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            # Print trainable parameters to confirm LoRA attachment during Phase A bring-up.
            try:
                self.model.print_trainable_parameters()
            except AttributeError:
                pass

        if self.cfg.device:
            logger.debug("Moving model to device: %s", self.cfg.device)
            self.model.to(self.cfg.device)

        self.model.eval()
        logger.info("Base model loaded successfully. Trainable parameters: %s", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def forward(self, input_ids: torch.Tensor, **kwargs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run a forward pass and return logits plus auxiliary outputs."""

        if self.model is None:
            raise RuntimeError("Call `load()` before using the trunk.")

        attention_mask = kwargs.get("attention_mask", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=kwargs.get("use_cache", True),
        )

        logger.debug(
            "Forward pass complete: batch=%s, seq_len=%s",
            input_ids.size(0),
            input_ids.size(1),
        )

        aux = {
            "hidden_states": outputs.hidden_states[-1],
            "past_key_values": outputs.past_key_values,
        }
        return outputs.logits, aux

    @property
    def base_model(self) -> torch.nn.Module:
        """Expose the underlying transformers model."""

        if self.model is None:
            raise RuntimeError("Model not loaded yet.")
        return self.model

    @property
    def primary_device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("Model not loaded yet.")
        for param in self.model.parameters():
            if param.device.type != "meta":
                return param.device
        logger.warning("No materialized parameters found; defaulting to CPU device.")
        return torch.device("cpu")

    def _load_tokenizer(self) -> None:
        """Load and configure the tokenizer associated with the model."""

        tokenizer_name = self.cfg.tokenizer_name or self.cfg.base_name
        logger.info("Loading tokenizer '%s'", tokenizer_name)
        tokenizer_kwargs: Dict[str, Any] = {}
        if self.cfg.cache_dir:
            tokenizer_kwargs["cache_dir"] = self.cfg.cache_dir
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
        if self.cfg.padding_side:
            tokenizer.padding_side = self.cfg.padding_side
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Tokenizer lacked pad_token; using eos_token id %s", tokenizer.pad_token_id)
        self.tokenizer = tokenizer
        logger.info("Tokenizer loaded (vocab_size=%s, pad=%s)", tokenizer.vocab_size, tokenizer.pad_token)

    def encode(self, text: str, **kwargs: Any) -> torch.Tensor:
        """Tokenize text into input ids tensor."""

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        enc = self.tokenizer(text, return_tensors="pt", **kwargs)
        return enc["input_ids"]

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token ids into text."""

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: Optional[bool] = None,
        **generate_kwargs: Any,
    ) -> str:
        """Generate text from a prompt using the underlying model."""

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generation.")

        device = self.primary_device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(device)
        if do_sample is None:
            do_sample = temperature > 0
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )
        logger.debug("Generating text (prompt_tokens=%s, max_new_tokens=%s)", inputs["input_ids"].shape[-1], max_new_tokens)
        outputs = self.model.generate(**inputs, **generation_config)
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        logger.debug("Generation complete (generated_tokens=%s)", generated.size(0))
        return response

    def build_chat_prompt(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Construct a chat prompt using the tokenizer template when available."""

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        # Fallback: simple role/content transcript.
        transcript = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            transcript.append(f"{role}: {content}")
        if add_generation_prompt:
            transcript.append("ASSISTANT:")
        return "\n".join(transcript)

    @torch.inference_mode()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> str:
        """Generate a response for a list of chat messages."""

        prompt = self.build_chat_prompt(messages, add_generation_prompt=True)
        logger.info("Running chat generation (messages=%s)", len(messages))
        reply = self.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        return reply




