from types import SimpleNamespace

import pytest
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.services.vector_store import VectorStore
from scripts.load_to_qdrant import load_documents


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.created: list[dict] = []
        self.indexes: list[tuple] = []
        self.upserts: list[dict] = []
        self.queries: list[dict] = []
        self.closed = False

    async def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

    async def create_collection(self, **kwargs):
        self.created.append(kwargs)
        self.collections.add(kwargs["collection_name"])

    async def get_collection(self, collection_name):
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=VectorParams(size=3, distance=Distance.COSINE)))
        )

    async def create_payload_index(self, **kwargs):
        self.indexes.append((kwargs["field_name"], kwargs["field_schema"]))

    async def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    async def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(points=["typed-point"])

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_ensure_collection_creates_cosine_collection_and_payload_indexes() -> None:
    client = FakeQdrantClient()
    store = VectorStore(url="http://unused", api_key=None, collection="documents", dim=3, client=client)

    await store.ensure_collection()

    assert client.created[0]["vectors_config"].size == 3
    assert client.created[0]["vectors_config"].distance == Distance.COSINE
    assert {name for name, _ in client.indexes} == {"source", "created_at", "tenant_id", "status"}


@pytest.mark.asyncio
async def test_upsert_batches_and_search_returns_points() -> None:
    client = FakeQdrantClient()
    store = VectorStore(url="http://unused", api_key=None, collection="documents", dim=3, client=client)
    points = [PointStruct(id=index, vector=[1.0, 0.0, 0.0], payload={}) for index in range(3)]

    await store.upsert(points, batch_size=2)
    result = await store.search([1.0, 0.0, 0.0], top_k=1)
    await store.close()

    assert [call["wait"] for call in client.upserts] == [False, True]
    assert result == ["typed-point"]
    assert client.closed is True


@pytest.mark.asyncio
async def test_wrong_vector_dimension_is_explained_before_upsert() -> None:
    store = VectorStore(
        url="http://unused", api_key=None, collection="documents", dim=3, client=FakeQdrantClient()
    )

    with pytest.raises(ValueError, match="размерность 2, ожидалось 3"):
        await store.upsert([PointStruct(id=1, vector=[1.0, 0.0], payload={})])


def test_loader_builds_120_deterministic_domain_documents() -> None:
    first = load_documents()
    second = load_documents()

    assert len(first) == 120
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert {"source", "text", "created_at", "tenant_id", "status"} <= first[0].keys()
