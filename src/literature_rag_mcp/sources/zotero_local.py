"""Zotero local SQLite source implementation.

完全复用 zotero-mcp/local_db.py 的 SQL 查询模式，确保 schema 兼容性。
Schema: itemData + itemDataValues + fields 组合（fieldID=1=title, 2=abstract, 16=extra, DOI via fields table）
"""

import platform
import sqlite3
from pathlib import Path

from .base import AbstractSource, SourceItem, SourceAttachment


class ZoteroLocalSource(AbstractSource):
    """Read items from local Zotero SQLite database."""

    def __init__(self, db_path: str = "", storage_path: str = ""):
        self.db_path = db_path or self._find_zotero_db()
        self.storage_path = storage_path or self._get_storage_dir()
        self._connection: sqlite3.Connection | None = None

    def _find_zotero_db(self) -> str:
        """Auto-detect Zotero database location."""
        system = platform.system()
        if system == "Darwin":
            db_path = Path.home() / "Zotero" / "zotero.sqlite"
        elif system == "Windows":
            db_path = Path.home() / "Zotero" / "zotero.sqlite"
        else:
            db_path = Path.home() / "Zotero" / "zotero.sqlite"

        if not db_path.exists():
            raise FileNotFoundError(f"Zotero database not found at {db_path}")
        return str(db_path)

    def _get_storage_dir(self) -> Path:
        """Infer storage directory from database path."""
        return Path(self.db_path).parent / "storage"

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._connection is None:
            uri = f"file:{self.db_path}?immutable=1"
            self._connection = sqlite3.connect(uri, uri=True)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def get_items(self, limit: int | None = None) -> list[SourceItem]:
        """Get items from local database.

        完全复用 local_db.py:get_items_with_text() 的 SQL 模式。
        """
        conn = self._get_connection()

        # Query参考 local_db.py:398-450 - 使用 itemData + itemDataValues + fields 组合
        query = """
            SELECT i.key,
                   i.itemID,
                   it.typeName as item_type,
                   i.dateAdded,
                   i.dateModified,
                   title_val.value as title,
                   abstract_val.value as abstract,
                   extra_val.value as extra,
                   doi_val.value as doi,
                   GROUP_CONCAT(
                       CASE
                           WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                           THEN c.lastName || ', ' || c.firstName
                           WHEN c.lastName IS NOT NULL
                           THEN c.lastName
                           ELSE NULL
                       END, '; '
                   ) as creators
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID

            -- Get title (fieldID = 1)
            LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
            LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID

            -- Get abstract (fieldID = 2)
            LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
            LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID

            -- Get extra field (fieldID = 16)
            LEFT JOIN itemData extra_data ON i.itemID = extra_data.itemID AND extra_data.fieldID = 16
            LEFT JOIN itemDataValues extra_val ON extra_data.valueID = extra_val.valueID

            -- Get DOI via fields table
            LEFT JOIN fields doi_f ON doi_f.fieldName = 'DOI'
            LEFT JOIN itemData doi_data ON i.itemID = doi_data.itemID AND doi_data.fieldID = doi_f.fieldID
            LEFT JOIN itemDataValues doi_val ON doi_data.valueID = doi_val.valueID

            -- Get creators
            LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
            LEFT JOIN creators c ON ic.creatorID = c.creatorID

            WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')

            GROUP BY i.itemID, i.key, i.itemTypeID, it.typeName, i.dateAdded, i.dateModified,
                     title_val.value, abstract_val.value, extra_val.value

            ORDER BY i.dateModified DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        items = []
        for row in conn.execute(query):
            creators = row["creators"] or ""
            # 去重：GROUP_CONCAT 可能产生重复
            unique_creators = "; ".join(dict.fromkeys(creators.split("; ")))
            items.append(SourceItem(
                key=row["key"],
                title=row["title"] or "",
                item_type=row["item_type"],
                creators=unique_creators,
                abstract=row["abstract"] or "",
                doi=row["doi"] or "",
                date=row["dateAdded"] or "",
            ))
        return items

    def get_item_by_key(self, key: str) -> SourceItem | None:
        """Get a single item by key."""
        conn = self._get_connection()

        query = """
            SELECT i.key,
                   it.typeName as item_type,
                   title_val.value as title,
                   abstract_val.value as abstract,
                   extra_val.value as extra,
                   doi_val.value as doi,
                   i.dateAdded,
                   GROUP_CONCAT(
                       CASE
                           WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                           THEN c.lastName || ', ' || c.firstName
                           WHEN c.lastName IS NOT NULL
                           THEN c.lastName
                           ELSE NULL
                       END, '; '
                   ) as creators
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID

            LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
            LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID

            LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
            LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID

            LEFT JOIN itemData extra_data ON i.itemID = extra_data.itemID AND extra_data.fieldID = 16
            LEFT JOIN itemDataValues extra_val ON extra_data.valueID = extra_val.valueID

            LEFT JOIN fields doi_f ON doi_f.fieldName = 'DOI'
            LEFT JOIN itemData doi_data ON i.itemID = doi_data.itemID AND doi_data.fieldID = doi_f.fieldID
            LEFT JOIN itemDataValues doi_val ON doi_data.valueID = doi_val.valueID

            LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
            LEFT JOIN creators c ON ic.creatorID = c.creatorID

            WHERE i.key = ?

            GROUP BY i.itemID, i.key, it.typeName, title_val.value, abstract_val.value, extra_val.value
        """

        for row in conn.execute(query, (key,)):
            creators = row["creators"] or ""
            unique_creators = "; ".join(dict.fromkeys(creators.split("; ")))
            return SourceItem(
                key=row["key"],
                title=row["title"] or "",
                item_type=row["item_type"],
                creators=unique_creators,
                abstract=row["abstract"] or "",
                doi=row["doi"] or "",
                date=row["dateAdded"] or "",
            )
        return None

    def get_attachments(self, item_key: str) -> list[SourceAttachment]:
        """Get attachments for an item.

        完全复用 local_db.py:_iter_parent_attachments() 的 SQL 模式。
        """
        conn = self._get_connection()

        # First get the item's itemID from key
        item_id_query = "SELECT itemID FROM items WHERE key = ?"
        row = conn.execute(item_id_query, (item_key,)).fetchone()
        if not row:
            return []
        parent_item_id = row["itemID"]

        # Query attachments (参考 local_db.py:147)
        query = """
            SELECT ia.itemID as attachmentItemID,
                   ia.parentItemID as parentItemID,
                   ia.path as path,
                   ia.contentType as contentType,
                   att.key as attachmentKey
            FROM itemAttachments ia
            JOIN items att ON att.itemID = ia.itemID
            WHERE ia.parentItemID = ?
        """

        attachments = []
        for row in conn.execute(query, (parent_item_id,)):
            attachments.append(SourceAttachment(
                key=row["attachmentKey"],
                parent_key=item_key,
                filename=row["path"] or "",
                content_type=row["contentType"] or "",
            ))
        return attachments
