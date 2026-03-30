import numpy as np
import requests
from typing import List, Optional
from scripts.logger import get_logger

log = get_logger("embedder")


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        batch_size: int = 64,
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self._dim: Optional[int] = None

        log.info(f"Connecting to Ollama embedder ({model}) at {base_url}...")
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()

            dim = self._embed_batch(["dim_probe"])[0].shape[0]
            self._dim = dim
            log.success(f"OllamaEmbedder ready — model={model}, dim={dim}")

        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {base_url}. "
                f"Is 'ollama serve' running and '{model}' pulled? Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Cannot initialize Ollama embedder at {base_url} "
                f"with model '{model}'. Error: {e}"
            ) from e

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        all_vecs = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            if show_progress:
                batch_num = i // self.batch_size + 1
                log.info(f"Embedding batch {batch_num}/{n_batches} ({len(batch)} texts)...")
            vecs = self._embed_batch(batch)
            all_vecs.append(np.vstack(vecs))

        return np.vstack(all_vecs).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": texts,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if "embeddings" in data and data["embeddings"]:
                return [np.array(x, dtype=np.float32) for x in data["embeddings"]]

            raise RuntimeError(f"Unexpected /api/embed response: {data}")

        except requests.exceptions.HTTPError as e:
            if e.response is None or e.response.status_code != 404:
                raise RuntimeError(f"Ollama /api/embed returned error: {e}") from e

        vectors = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if "embedding" not in data or not data["embedding"]:
                raise RuntimeError(f"Unexpected /api/embeddings response: {data}")

            vectors.append(np.array(data["embedding"], dtype=np.float32))

        return vectors

    @staticmethod
    def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms
