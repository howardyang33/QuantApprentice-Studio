"""CLI for the minimal QuantApprentice research memory store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .._paths import project_root
from .store import ITEM_TYPE_TO_DIR, MemoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuantApprentice Pilot 1 research memory CLI")
    parser.add_argument(
        "--memory-dir",
        default=str(project_root() / "research_memory"),
        help="Root directory for the file-backed research memory",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize the research memory directory skeleton")

    add_parser = subparsers.add_parser("add", help="Create one memory item and append one manifest entry")
    add_parser.add_argument("--type", required=True, choices=sorted(ITEM_TYPE_TO_DIR))
    add_parser.add_argument("--title", required=True)
    add_parser.add_argument("--summary", default="")
    add_parser.add_argument("--status", default="draft")
    add_parser.add_argument("--payload-json", default=None, help="Inline JSON object with type-specific fields")
    add_parser.add_argument("--payload-file", default=None, help="Path to a JSON file with type-specific fields")
    add_parser.add_argument("--tag", action="append", default=[], help="Repeatable tag value")
    add_parser.add_argument("--link-id", action="append", default=[], help="Repeatable linked memory_id")
    add_parser.add_argument("--source-label", default="cli")
    add_parser.add_argument("--item-id", default=None, help="Optional explicit type-specific id")
    add_parser.add_argument("--dummy", action="store_true", help="Mark the item as a dummy example")
    add_parser.add_argument("--bootstrap-example", action="store_true", help="Mark the item as a bootstrap example")

    get_parser = subparsers.add_parser("get", help="Read one memory item by memory_id or path")
    get_group = get_parser.add_mutually_exclusive_group(required=True)
    get_group.add_argument("--memory-id", default=None)
    get_group.add_argument("--path", default=None)

    list_parser = subparsers.add_parser("list", help="List manifest summaries")
    list_parser.add_argument("--type", default=None, choices=sorted(ITEM_TYPE_TO_DIR))
    list_parser.add_argument("--limit", type=int, default=20)

    return parser


def load_payload(payload_json: Optional[str], payload_file: Optional[str]) -> Dict[str, Any]:
    if payload_json and payload_file:
        raise ValueError("Use only one of --payload-json or --payload-file")
    if payload_json:
        data = json.loads(payload_json)
    elif payload_file:
        data = json.loads(Path(payload_file).read_text(encoding="utf-8"))
    else:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Payload must be a JSON object")
    return data


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    store = MemoryStore(args.memory_dir)

    if args.command == "init":
        print(json.dumps(store.init_storage(), indent=2, ensure_ascii=False))
        return

    if args.command == "add":
        payload = load_payload(args.payload_json, args.payload_file)
        created = store.create_item(
            item_type=args.type,
            title=args.title,
            summary=args.summary,
            status=args.status,
            payload=payload,
            tags=args.tag,
            linked_ids=args.link_id,
            source_label=args.source_label,
            is_dummy=args.dummy,
            bootstrap_example=args.bootstrap_example,
            item_id=args.item_id,
        )
        output = {
            "memory_id": created.item["memory_id"],
            "item_type": created.item["item_type"],
            "item_id": created.manifest_entry["item_id"],
            "path": str(created.path),
            "status": created.item["status"],
            "title": created.item["title"],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if args.command == "get":
        item = store.get_item(memory_id=getattr(args, "memory_id", None), path=getattr(args, "path", None))
        print(json.dumps(item, indent=2, ensure_ascii=False))
        return

    if args.command == "list":
        items = store.list_items(item_type=args.type, limit=args.limit)
        print(json.dumps({"count": len(items), "items": items}, indent=2, ensure_ascii=False))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
