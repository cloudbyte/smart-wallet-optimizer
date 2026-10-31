import json
import os
import sys
from typing import List, Optional


LOCAL_MODEL_ID = "Qwen/Qwen3-0.6B"
COLAB_MODEL_ID = "Qwen/Qwen3-4B"
MODEL_ENV_VAR = "WALLET_AGENT_MODEL"
DEVICE_ENV_VAR = "WALLET_AGENT_DEVICE"
DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def is_colab_runtime() -> bool:
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def default_model_id() -> str:
    configured_model = os.environ.get(MODEL_ENV_VAR)
    if configured_model:
        return configured_model
    return COLAB_MODEL_ID if is_colab_runtime() else LOCAL_MODEL_ID


def default_device() -> str:
    configured_device = os.environ.get(DEVICE_ENV_VAR)
    if configured_device:
        return configured_device
    return "auto" if is_colab_runtime() else "cpu"


class Qwen3AgentModel:
    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_id = model_id or default_model_id()
        self.device = device or default_device()
        self._tokenizer = None
        self._model = None
        self.embedding_model_id = DEFAULT_EMBEDDING_MODEL_ID
        self._embedding_tokenizer = None
        self._embedding_model = None

    def _load_model(self) -> None:
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=(
                    torch.float16
                    if self.device != "cpu" and torch.cuda.is_available()
                    else torch.float32
                ),
                device_map=self.device,
            )
            self._model.eval()

    def choose_tool(self, context: str, available_actions: List[str]) -> Optional[str]:
        generated = self._generate(
            "You are the action selector in a credit-card recommendation agent. "
            "Choose exactly one action from available_actions. Use the observations "
            "to choose the most useful next tool. Never invent an action. Return "
            'only a JSON object in this form: {"action":"exact_action_name"}.',
            json.dumps(
                {"context": json.loads(context), "available_actions": available_actions},
                indent=2,
            ),
            max_new_tokens=48,
        )
        try:
            action = json.loads(generated).get("action")
        except (json.JSONDecodeError, AttributeError):
            return None
        return action if isinstance(action, str) else None

    def critique_tree(self, facts: str) -> str:
        return self._generate(
            "You are a critic comparing candidate card branches in a Tree-of-Thought "
            "agent. Use only the supplied recommendations, node values, and retrieved "
            "evidence. Write one concise qualitative sentence naming a candidate and "
            "copying an exact phrase from its retrieved evidence. Do not invent "
            "benefits, rules, or conditions. Do not write numbers, dollar values, "
            "percentages, or claims not directly supported by the evidence.",
            facts,
            max_new_tokens=96,
        )

    def explain(self, facts: str) -> str:
        return self._generate(
            "You explain a credit-card recommendation to a user. "
            "Use only the supplied JSON facts. Write at most two short "
            "sentences. The recommendation and calculations are already "
            "determined; do not change the selected card. Name the "
            "recommended card and briefly relate it to the merchant or "
            "category. Do not include numbers, financial terms, fees, "
            "approvals, eligibility, or card benefits.",
            facts,
            max_new_tokens=96,
        )

    def embed(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        import torch
        import torch.nn.functional as functional

        if not texts:
            return []
        self._load_embedding_model(torch)

        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = self._embedding_tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._embedding_model.device)
            with torch.inference_mode():
                outputs = self._embedding_model(**batch, return_dict=True)
                token_vectors = outputs.last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(token_vectors.dtype)
                pooled = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                    min=1
                )
                normalized = functional.normalize(pooled, p=2, dim=1)
            vectors.extend(normalized.cpu().tolist())
        return vectors

    def _load_embedding_model(self, torch) -> None:
        if self._embedding_model is None:
            from transformers import AutoModel, AutoTokenizer

            if torch.cuda.is_available():
                device = torch.device("cuda")
                dtype = torch.float16
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
                dtype = torch.float32
            else:
                device = torch.device("cpu")
                dtype = torch.float32

            self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_id
            )
            self._embedding_model = AutoModel.from_pretrained(
                self.embedding_model_id,
                torch_dtype=dtype,
            ).to(device)
            self._embedding_model.eval()

    def _generate(self, system_prompt: str, facts: str, max_new_tokens: int) -> str:
        self._load_model()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": facts},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        import torch
        from transformers import GenerationConfig

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_k=50,
                    top_p=1.0,
                    bos_token_id=self._tokenizer.bos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    pad_token_id=self._tokenizer.eos_token_id,
                ),
            )

        response = self._tokenizer.decode(
            generated[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        if not response:
            raise RuntimeError(f"{self.model_id} returned an empty explanation.")
        return response


Qwen3Explainer = Qwen3AgentModel
