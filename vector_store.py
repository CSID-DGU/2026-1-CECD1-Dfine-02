"""
vector_store.py — (멘토 페르소나 벡터 DB)

멘토 페르소나를 임베딩해 FAISS 인덱스로 저장하고,
맥락 멘토 선택을 위한 ANN(근사 최근접) 검색을 제공한다.

- 임베딩 소스(persona_text): consumption_style + hobbies + skills + life_quote + intro
- 인덱스: 코사인 유사도(정규화 벡터의 내적, IndexFlatIP)
- 영속화: 임베딩/메타는 mentor_index_meta.json, FAISS 인덱스는 mentor_index.faiss
- faiss 미설치 환경에서는 numpy 브루트포스로 자동 폴백 (멘토 수가 적어 성능 동일)

빌드:  python vector_store.py            # mentor_cards.jsonl → 인덱스 파일 생성
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

BASE_DIR   = Path(__file__).parent
CARDS_PATH = BASE_DIR / "data" / "mentor_cards.json"
INDEX_PATH = BASE_DIR / "data" / "mentor_index.faiss"
META_PATH  = BASE_DIR / "data" / "mentor_index_meta.json"

EMBED_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
_embed_model = None


# ── 임베딩 ───────────────────────────────────────────────────────────────────
def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def _embed(texts: list[str]) -> np.ndarray:
    return _get_model().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")


def persona_text(card: dict) -> str:
    """멘토 카드 → 임베딩 대상 페르소나 문장 (실제 스키마)"""
    d = card.get("details", {})
    parts = [
        card.get("summary", ""),
        d.get("career_goals_and_ambitions", ""),
        d.get("cultural_background", ""),
        d.get("culinary_persona", ""),
        " ".join(d.get("hobbies_and_interests_list", [])),
        card.get("labels", {}).get("primary_tag", ""),
    ]
    return " ".join(p for p in parts if p)


def _try_import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


# ── 벡터 스토어 ──────────────────────────────────────────────────────────────
class MentorVectorStore:
    """멘토 페르소나 임베딩 저장소 + ANN 검색"""

    def __init__(self, uuids: list[str], embeddings: np.ndarray):
        self.uuids = uuids
        self.embeddings = embeddings.astype("float32")   # (N, dim), L2 정규화됨
        self.dim = int(embeddings.shape[1])
        self._faiss = _try_import_faiss()
        self._index = None
        if self._faiss is not None:
            self._index = self._faiss.IndexFlatIP(self.dim)
            self._index.add(self.embeddings)

    # 빌드 ----------------------------------------------------------------------
    @classmethod
    def build(cls, cards: list[dict]) -> "MentorVectorStore":
        uuids = [c["uuid"] for c in cards]
        embs  = _embed([persona_text(c) for c in cards])
        return cls(uuids, embs)

    # 영속화 --------------------------------------------------------------------
    def save(self, index_path: Path = INDEX_PATH, meta_path: Path = META_PATH):
        meta = {
            "model": EMBED_MODEL_NAME,
            "dim": self.dim,
            "uuids": self.uuids,
            "embeddings": self.embeddings.tolist(),   # 폴백/재빌드용 source-of-truth
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        if self._faiss is not None and self._index is not None:
            self._faiss.write_index(self._index, str(index_path))

    @classmethod
    def load(cls, index_path: Path = INDEX_PATH, meta_path: Path = META_PATH) -> "MentorVectorStore":
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        embs  = np.asarray(meta["embeddings"], dtype="float32")
        store = cls(meta["uuids"], embs)
        # 저장된 faiss 인덱스가 있으면 그대로 사용
        if store._faiss is not None and Path(index_path).exists():
            store._index = store._faiss.read_index(str(index_path))
        return store

    # 검색 ----------------------------------------------------------------------
    def search(
        self,
        query_text: str,
        k: int = 5,
        exclude_uuids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """query_text와 가장 가까운 멘토 (uuid, score) 리스트를 score 내림차순 반환"""
        exclude_uuids = exclude_uuids or set()
        q = _embed([query_text])[0]                       # (dim,)

        if self._index is not None:
            # 제외 멘토를 고려해 넉넉히 검색
            top = min(len(self.uuids), k + len(exclude_uuids))
            scores, idxs = self._index.search(q.reshape(1, -1), top)
            pairs = [(self.uuids[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]
        else:
            sims  = self.embeddings @ q                   # (N,)
            order = np.argsort(-sims)
            pairs = [(self.uuids[i], float(sims[i])) for i in order]

        out = [(u, s) for u, s in pairs if u not in exclude_uuids]
        return out[:k]


# ── 데이터 로드 ──────────────────────────────────────────────────────────────
def _load_cards(path: Path = CARDS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_store() -> "MentorVectorStore | None":
    """저장된 인덱스를 로드. 없으면 None (호출부에서 폴백 처리)."""
    if META_PATH.exists():
        return MentorVectorStore.load()
    return None


# ── 빌드 진입점 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cards = _load_cards()
    print(f"멘토 {len(cards)}명 임베딩 중… (model={EMBED_MODEL_NAME})")
    store = MentorVectorStore.build(cards)
    store.save()
    backend = "FAISS" if store._faiss is not None else "numpy 폴백"
    print(f"인덱스 저장 완료 ({backend}, dim={store.dim})")
    print(f"  meta : {META_PATH}")
    if store._faiss is not None:
        print(f"  index: {INDEX_PATH}")
    # 간단 검색 데모
    demo = "친구랑 같이 한정판 굿즈를 사고 싶어 SNS에서 보고 충동적으로"
    print(f"\n[검색 데모] '{demo}'")
    by_uuid = {c["uuid"]: c for c in cards}
    for uuid, score in store.search(demo, k=3):
        card = by_uuid[uuid]
        print(f"  {score:.3f}  {card['profile']['name']} "
              f"({card['labels']['archetype']}, {card['labels']['primary_tag']})")
