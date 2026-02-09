"""Ollama client: RAG context + query -> answer text."""

import ollama


class OllamaClient:
    """Local LLM via Ollama. Prompt: system + Context + Question -> Answer."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self._client = ollama.Client(host=base_url)

    def generate(
        self,
        query: str,
        context: str = "",
        system_prompt: str | None = None,
    ) -> str:
        """Generate answer from context and query. Returns only the answer text."""
        system = system_prompt or (
            "Answer briefly with the same language as the question, based only on the context below. "
            "If the context does not contain the answer, say you don't know."
        )
        user = ""
        if context and context.strip():
            user = f"Context:\n{context.strip()}\n\nQuestion: {query}\n\nAnswer:"
        else:
            user = f"Question: {query}\n\nAnswer:"
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        msg = response.get("message") or {}
        return (msg.get("content") or "").strip()
