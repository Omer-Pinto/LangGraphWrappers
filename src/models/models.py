import os
from enum import Enum
from typing import Optional


class Model(str, Enum):
    """Enum for common LLM model identifiers across different providers."""

    # OpenAI Models
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O1 = "o1"
    O1_MINI = "o1-mini"
    O1_PREVIEW = "o1-preview"

    # OpenAI Models - Additional (based on research)
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    O3 = "o3"
    O3_MINI = "o3-mini"
    O4_MINI = "o4-mini"
    CHATGPT_4O_LATEST = "chatgpt-4o-latest"
    GPT_3_5_TURBO = "gpt-3.5-turbo"

    # Anthropic Models (your existing entries - UNCHANGED)
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_OPUS_4_1 = "claude-opus-4-1-20250514"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # Meta Llama Models (your existing entries - UNCHANGED)
    LLAMA_3_3_70B = "meta-llama/Llama-3.3-70B-Instruct"
    LLAMA_3_1_405B = "meta-llama/Llama-3.1-405B-Instruct"
    LLAMA_3_1_70B = "meta-llama/Llama-3.1-70B-Instruct"
    LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"
    LLAMA_3_70B = "meta-llama/Llama-3-70B-Instruct"
    LLAMA_3_8B = "meta-llama/Llama-3-8B-Instruct"

    # Meta Llama Models - Additional
    LLAMA_4_SCOUT_17B = "meta-llama/llama-4-scout-17b-16e-instruct"
    LLAMA_4_MAVERICK_17B = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B"
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B"
    LLAMA_3_2_11B_VISION = "meta-llama/Llama-3.2-11B-Vision"
    LLAMA_3_2_90B_VISION = "meta-llama/Llama-3.2-90B-Vision"
    LLAMA_GUARD_3_8B = "meta-llama/llama-guard-3-8b"
    LLAMA_GUARD_4_12B = "meta-llama/llama-guard-4-12b"

    # DeepSeek Models
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"
    DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"
    DEEPSEEK_V3_1 = "deepseek-ai/DeepSeek-V3.1"
    DEEPSEEK_V3_2_EXP = "deepseek-ai/DeepSeek-V3.2-Exp"
    DEEPSEEK_R1 = "deepseek-r1"
    DEEPSEEK_R1_DISTILL_LLAMA_70B = "deepseek-r1-distill-llama-70b"

    # Mistral Models
    MISTRAL_LARGE_LATEST = "mistral-large-latest"
    MISTRAL_LARGE_24_11 = "mistral-large-24.11"
    MISTRAL_MEDIUM_LATEST = "mistral-medium-latest"
    MISTRAL_MEDIUM_3 = "mistral-medium-3"
    MISTRAL_SMALL_LATEST = "mistral-small-latest"
    MISTRAL_SMALL_3_1 = "mistral-small-3.1"
    MISTRAL_TINY = "mistral-tiny"
    MAGISTRAL_SMALL = "magistral-small-2506"
    MAGISTRAL_MEDIUM = "magistral-medium-2506"
    MAGISTRAL_MEDIUM_LATEST = "magistral-medium-latest"
    CODESTRAL_LATEST = "codestral-latest"
    CODESTRAL_25_01 = "codestral-25.01"
    CODESTRAL_2 = "codestral-2"
    PIXTRAL_12B = "pixtral-12b"
    MISTRAL_NEMO = "mistral-nemo"
    MISTRAL_7B = "mistral-7b"
    MISTRAL_OCR = "mistral-ocr-25.05"
    MISTRAL_EMBED = "mistral-embed"

    # Google Gemini Models
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_PRO_PREVIEW = "gemini-2.5-pro-preview-05-06"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_PREVIEW = "gemini-2.5-flash-preview-05-20"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_FLASH_NATIVE_AUDIO = "gemini-2.5-flash-native-audio-preview-09-2025"
    GEMINI_2_5_COMPUTER_USE = "gemini-2.5-computer-use"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_FLASH_LATEST = "gemini-flash-latest"
    GEMINI_PRO_LATEST = "gemini-pro-latest"
    GEMINI_EMBEDDING_001 = "gemini-embedding-001"
    GEMMA_3 = "gemma-3"
    GEMMA_3N = "gemma-3n"
    GEMMA_2 = "gemma-2"
    GEMMA_2_9B = "gemma2-9b-it"

    # Groq Models (via Groq API)
    GROQ_LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    GROQ_LLAMA_3_3_70B_SPECDEC = "llama-3.3-70b-specdec"
    GROQ_LLAMA_3_1_8B = "llama-3.1-8b-instant"
    GROQ_LLAMA_3_70B = "llama3-70b-8192"
    GROQ_LLAMA_3_8B = "llama3-8b-8192"
    GROQ_LLAMA_3_2_1B = "llama-3.2-1b-preview"
    GROQ_LLAMA_3_2_3B = "llama-3.2-3b-preview"
    GROQ_LLAMA_3_2_11B_VISION = "llama-3.2-11b-vision-preview"
    GROQ_LLAMA_3_2_90B_VISION = "llama-3.2-90b-vision-preview"
    GROQ_GEMMA_2_9B = "gemma2-9b-it"
    GROQ_MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GROQ_DEEPSEEK_R1_LLAMA_70B = "deepseek-r1-distill-llama-70b"
    GROQ_DEEPSEEK_R1_LLAMA_70B_SPECDEC = "deepseek-r1-distill-llama-70b-specdec"
    GROQ_WHISPER_LARGE_V3 = "whisper-large-v3"
    GROQ_WHISPER_LARGE_V3_TURBO = "whisper-large-v3-turbo"
    GROQ_DISTIL_WHISPER_LARGE_V3 = "distil-whisper-large-v3-en"
    GROQ_OPENAI_GPT_OSS_120B = "openai/gpt-oss-120b"
    GROQ_OPENAI_GPT_OSS_20B = "openai/gpt-oss-20b"
    GROQ_KIMI_K2 = "moonshotai/kimi-k2-instruct"
    GROQ_KIMI_K2_0905 = "moonshotai/kimi-k2-instruct-0905"

    # Ollama Models (commonly used, subset of 100+ available)
    OLLAMA_LLAMA_3_1 = "llama3.1"
    OLLAMA_LLAMA_3_2 = "llama3.2"
    OLLAMA_LLAMA_3_2_VISION = "llama3.2-vision"
    OLLAMA_QWEN_3 = "qwen3"
    OLLAMA_QWEN_2_5 = "qwen2.5"
    OLLAMA_QWEN_3_CODER = "qwen3-coder"
    OLLAMA_MISTRAL = "mistral"
    OLLAMA_MISTRAL_LARGE_2 = "mistral-large-2"
    OLLAMA_MIXTRAL = "mixtral"
    OLLAMA_PHI_4 = "phi-4"
    OLLAMA_PHI_4_MINI = "phi-4-mini"
    OLLAMA_GEMMA_2 = "gemma2"
    OLLAMA_GEMMA_3 = "gemma3"
    OLLAMA_CODELLAMA = "codellama"
    OLLAMA_DEEPSEEK_CODER = "deepseek-coder"
    OLLAMA_DEEPCODER = "deepcoder"
    OLLAMA_STARCODER = "starcoder"
    OLLAMA_DOLPHIN_MISTRAL = "dolphin-mistral"
    OLLAMA_DOLPHIN_LLAMA_3_1 = "dolphin3"
    OLLAMA_WIZARDLM = "wizardlm"
    OLLAMA_VICUNA = "vicuna"
    OLLAMA_ORCA = "orca"
    OLLAMA_TINYLLAMA = "tinyllama"
    OLLAMA_COMMAND_R = "command-r"
    OLLAMA_BAKLLAVA = "bakllava"
    OLLAMA_LLAVA = "llava"
    OLLAMA_OLMO_2 = "olmo2"
    OLLAMA_NOUS_HERMES = "nous-hermes"
    OLLAMA_OPENAI_GPT_OSS = "gpt-oss"

    @property
    def provider(self) -> str:
        """Returns the provider name for the model."""
        if self.value.startswith("gpt-") or self.value.startswith("o1"):
            return "openai"
        elif self.value.startswith("claude-"):
            return "anthropic"
        elif "llama" in self.value.lower():
            return "meta"
        elif self.value.startswith("gemini-"):
            return "google"
        elif "mistral" in self.value.lower() or "mixtral" in self.value.lower():
            return "mistral"
        elif self.value.startswith("command-"):
            return "cohere"
        return "unknown"

    def __str__(self) -> str:
        return self.value

class ModelUrl(str, Enum):
    OPENAI = "https://api.openai.com/v1"
    GROQ = "https://api.groq.com/openai/v1"


def get_api_key_for_model(model: Model) -> Optional[str]:
    if model in [
        Model.GROQ_LLAMA_3_3_70B,
        Model.GROQ_LLAMA_3_3_70B_SPECDEC,
        Model.GROQ_LLAMA_3_1_8B,
        Model.GROQ_LLAMA_3_70B,
        Model.GROQ_LLAMA_3_8B,
        Model.GROQ_LLAMA_3_2_1B,
        Model.GROQ_LLAMA_3_2_3B,
        Model.GROQ_LLAMA_3_2_11B_VISION,
        Model.GROQ_LLAMA_3_2_90B_VISION,
        Model.GROQ_GEMMA_2_9B,
        Model.GROQ_MIXTRAL_8X7B,
        Model.GROQ_DEEPSEEK_R1_LLAMA_70B,
        Model.GROQ_DEEPSEEK_R1_LLAMA_70B_SPECDEC,
        Model.GROQ_WHISPER_LARGE_V3,
        Model.GROQ_WHISPER_LARGE_V3_TURBO,
        Model.GROQ_DISTIL_WHISPER_LARGE_V3,
        Model.GROQ_OPENAI_GPT_OSS_120B,
        Model.GROQ_OPENAI_GPT_OSS_20B,
        Model.GROQ_KIMI_K2,
        Model.GROQ_KIMI_K2_0905,
    ]:
        return os.environ.get("GROQ_API_KEY")
    else:
        return None

def get_base_url_for_model(model: Model) -> Optional[str]:
    """
    Returns the appropriate base_url for a given model.
    Returns None for OpenAI models (uses default).

    Args:
        model: A Model enum value

    Returns:
        The base_url string, or None if default OpenAI URL should be used
    """
    model_value = model.value

    # OpenAI models - use default (None)
    if (model_value.startswith("gpt-") or
            model_value.startswith("o1") or
            model_value.startswith("o3") or
            model_value.startswith("o4") or
            model_value.startswith("chatgpt-")):
        return None

    # Groq models
    if model in [
        Model.GROQ_LLAMA_3_3_70B,
        Model.GROQ_LLAMA_3_3_70B_SPECDEC,
        Model.GROQ_LLAMA_3_1_8B,
        Model.GROQ_LLAMA_3_70B,
        Model.GROQ_LLAMA_3_8B,
        Model.GROQ_LLAMA_3_2_1B,
        Model.GROQ_LLAMA_3_2_3B,
        Model.GROQ_LLAMA_3_2_11B_VISION,
        Model.GROQ_LLAMA_3_2_90B_VISION,
        Model.GROQ_GEMMA_2_9B,
        Model.GROQ_MIXTRAL_8X7B,
        Model.GROQ_DEEPSEEK_R1_LLAMA_70B,
        Model.GROQ_DEEPSEEK_R1_LLAMA_70B_SPECDEC,
        Model.GROQ_WHISPER_LARGE_V3,
        Model.GROQ_WHISPER_LARGE_V3_TURBO,
        Model.GROQ_DISTIL_WHISPER_LARGE_V3,
        Model.GROQ_OPENAI_GPT_OSS_120B,
        Model.GROQ_OPENAI_GPT_OSS_20B,
        Model.GROQ_KIMI_K2,
        Model.GROQ_KIMI_K2_0905,
    ]:
        return "https://api.groq.com/openai/v1"

    # Anthropic models
    if model_value.startswith("claude-"):
        return "https://api.anthropic.com/v1"

    # DeepSeek models
    if (model_value.startswith("deepseek-") or
            "DeepSeek" in model_value):
        return "https://api.deepseek.com/v1"

    # Mistral models
    if (model_value.startswith("mistral-") or
            model_value.startswith("magistral-") or
            model_value.startswith("codestral-") or
            model_value.startswith("pixtral-")):
        return "https://api.mistral.ai/v1"

    # Google Gemini models
    if (model_value.startswith("gemini-") or
            model_value.startswith("gemma-")):
        return "https://generativelanguage.googleapis.com/v1beta"

    # Ollama models (local)
    if model in [
        Model.OLLAMA_LLAMA_3_1,
        Model.OLLAMA_LLAMA_3_2,
        Model.OLLAMA_LLAMA_3_2_VISION,
        Model.OLLAMA_QWEN_3,
        Model.OLLAMA_QWEN_2_5,
        Model.OLLAMA_QWEN_3_CODER,
        Model.OLLAMA_MISTRAL,
        Model.OLLAMA_MISTRAL_LARGE_2,
        Model.OLLAMA_MIXTRAL,
        Model.OLLAMA_PHI_4,
        Model.OLLAMA_PHI_4_MINI,
        Model.OLLAMA_GEMMA_2,
        Model.OLLAMA_GEMMA_3,
        Model.OLLAMA_CODELLAMA,
        Model.OLLAMA_DEEPSEEK_CODER,
        Model.OLLAMA_DEEPCODER,
        Model.OLLAMA_STARCODER,
        Model.OLLAMA_DOLPHIN_MISTRAL,
        Model.OLLAMA_DOLPHIN_LLAMA_3_1,
        Model.OLLAMA_WIZARDLM,
        Model.OLLAMA_VICUNA,
        Model.OLLAMA_ORCA,
        Model.OLLAMA_TINYLLAMA,
        Model.OLLAMA_COMMAND_R,
        Model.OLLAMA_BAKLLAVA,
        Model.OLLAMA_LLAVA,
        Model.OLLAMA_OLMO_2,
        Model.OLLAMA_NOUS_HERMES,
        Model.OLLAMA_OPENAI_GPT_OSS,
    ]:
        return "http://localhost:11434/v1"

    # Meta Llama models (via API providers like Together AI, Replicate, etc.)
    # These typically need a third-party provider URL
    if model_value.startswith("meta-llama/"):
        # Default to Together AI for Meta models
        return "https://api.together.xyz/v1"

    # Unknown model
    return None