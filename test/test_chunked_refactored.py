import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "llm-graph-builder", "backend", "src")))

import pytest
from chunked_kg_creator_refactored import ChunkedKGCreator

# Dummy objects to simulate CreateChunksofDocument behavior
class DummyRawChunk:
    def __init__(self, text, start_pos=0, end_pos=None):
        self.text = text
        self.page_content = text
        self.start_pos = start_pos
        self.end_pos = end_pos if end_pos is not None else len(text)

class FakeCreateChunks:
    def __init__(self, pages, graph):
        pass

    def split_file_into_chunks(self, size, overlap):
        # Return two simple dummy chunks
        return [
            DummyRawChunk("abc", 0, 3),
            DummyRawChunk("def", 3, 6)
        ]

@pytest.fixture(autouse=True)
def patch_create_chunks(monkeypatch):
    # Patch CreateChunksofDocument to use our fake implementation
    import create_chunks
    monkeypatch.setattr(create_chunks, "CreateChunksofDocument", FakeCreateChunks)

def test_chunk_text_delegation():
    creator = ChunkedKGCreator(chunk_size=3, chunk_overlap=0)
    chunks = creator._chunk_text("abcdef")
    assert isinstance(chunks, list)
    assert len(chunks) == 2
    assert chunks[0]["text"] == "abc"
    assert chunks[0]["chunk_id"] == 0
    assert chunks[0]["start_pos"] == 0
    assert chunks[0]["end_pos"] == 3
    assert chunks[1]["text"] == "def"
    assert chunks[1]["chunk_id"] == 1
    assert chunks[1]["start_pos"] == 3
    assert chunks[1]["end_pos"] == 6
