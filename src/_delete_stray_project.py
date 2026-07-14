"""
Dry-run (default) / execute delete of a stray project's chunks from the
shared salit ChromaDB collection, scoped by project_id.

Usage:
    python _delete_stray_project.py <project_id>              # dry run — prints only
    python _delete_stray_project.py <project_id> --execute     # actually deletes
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ingest_to_chroma import get_collection


def main():
    if len(sys.argv) < 2:
        print("Usage: python _delete_stray_project.py <project_id> [--execute]")
        sys.exit(1)
    project_id = sys.argv[1]
    execute = "--execute" in sys.argv[2:]

    col = get_collection()
    victims = col.get(where={"project_id": project_id}, include=["metadatas"])
    ids = victims["ids"]
    metas = victims["metadatas"]

    print(f"project_id={project_id!r}: {len(ids)} chunk(s) matched\n")
    for cid, m in sorted(zip(ids, metas), key=lambda x: x[0]):
        print(f"  {cid}   (chunk_type={m.get('chunk_type')})")

    if not execute:
        print(f"\n[DRY RUN] {len(ids)} chunk(s) would be deleted. "
              f"Re-run with --execute to actually delete.")
        return

    if not ids:
        print("\nNothing to delete.")
        return
    col.delete(where={"project_id": project_id})
    print(f"\n[EXECUTED] Deleted {len(ids)} chunk(s) for project_id={project_id!r}.")


if __name__ == "__main__":
    main()
