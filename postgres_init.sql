CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

-- One row per ingested source. Holds source-level metadata (author, title,
-- ...) once, instead of duplicating it onto every chunk.
CREATE TABLE document (
    id           BIGSERIAL PRIMARY KEY,
    external_id  TEXT NOT NULL UNIQUE,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- One row per retrieval unit. external_id is Chunk.id ("{document_id}#{index}").
CREATE TABLE chunk (
    id           BIGSERIAL PRIMARY KEY,
    external_id  TEXT NOT NULL UNIQUE,
    document_id  BIGINT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    text         TEXT NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX chunk_document_id_idx ON chunk (document_id);

CREATE INDEX chunk_bm25_idx
    ON chunk
    USING bm25 (id, text)
    WITH (key_field = 'id');

-- One embedding table per (model, dims). Created on demand by the app,
-- but here is the shape for the default model (multilingual-e5-small, 384 dims):
CREATE TABLE multilingual_e5_small_384 (
    chunk_id  BIGINT PRIMARY KEY REFERENCES chunk(id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL
);

CREATE INDEX multilingual_e5_small_384_hnsw_idx
    ON multilingual_e5_small_384
    USING hnsw (embedding vector_ip_ops);
