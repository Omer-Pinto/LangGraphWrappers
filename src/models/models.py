from enum import Enum


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

    # Anthropic Models
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_OPUS_4_1 = "claude-opus-4-1-20250514"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # Meta Llama Models
    LLAMA_3_3_70B = "meta-llama/Llama-3.3-70B-Instruct"
    LLAMA_3_1_405B = "meta-llama/Llama-3.1-405B-Instruct"
    LLAMA_3_1_70B = "meta-llama/Llama-3.1-70B-Instruct"
    LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"
    LLAMA_3_70B = "meta-llama/Llama-3-70B-Instruct"
    LLAMA_3_8B = "meta-llama/Llama-3-8B-Instruct"

    # Google Gemini Models
    GEMINI_2_0_FLASH_EXP = "gemini-2.0-flash-exp"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_FLASH_8B = "gemini-1.5-flash-8b"

    # Mistral Models
    MISTRAL_LARGE = "mistral-large-latest"
    MISTRAL_SMALL = "mistral-small-latest"
    MIXTRAL_8X7B = "open-mixtral-8x7b"
    MIXTRAL_8X22B = "open-mixtral-8x22b"

    # Cohere Models
    COMMAND_R_PLUS = "command-r-plus"
    COMMAND_R = "command-r"

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