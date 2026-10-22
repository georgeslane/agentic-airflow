"""Ready-made :class:`~agentic_airflow.models.ModelBackend` adapters.

* :class:`agentic_airflow.backends.openai_compatible.OpenAICompatibleBackend`
  - any OpenAI-style ``/chat/completions`` endpoint (vLLM, Ollama, llama.cpp,
  LM Studio, OpenAI...). Uses only the standard library.
* :class:`agentic_airflow.backends.anthropic.AnthropicBackend` - Claude via the
  optional ``anthropic`` SDK.

Import them from their modules so the core package stays dependency-free.
"""
