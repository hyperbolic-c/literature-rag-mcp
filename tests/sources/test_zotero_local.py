import pytest
from dataclasses import dataclass


def test_abstract_source_interface():
    from literature_rag_mcp.sources.base import AbstractSource
    assert hasattr(AbstractSource, 'get_items')
    assert hasattr(AbstractSource, 'get_item_by_key')
    assert hasattr(AbstractSource, 'get_attachments')


def test_zotero_local_implements_interface():
    from literature_rag_mcp.sources.zotero_local import ZoteroLocalSource
    from literature_rag_mcp.sources.base import AbstractSource
    assert issubclass(ZoteroLocalSource, AbstractSource)
