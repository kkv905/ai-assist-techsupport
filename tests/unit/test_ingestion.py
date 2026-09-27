from pathlib import Path

from app.services.ingestion import document_metadata


def test_document_metadata_uses_category_version_and_timestamp(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = data_dir / "network" / "vpn-guide-v1.2.md"
    path.parent.mkdir(parents=True)
    path.write_text("VPN", encoding="utf-8")

    metadata = document_metadata(path, data_dir)

    assert metadata["source"] == "vpn-guide-v1.2.md"
    assert metadata["category"] == "network"
    assert metadata["version"] == "1.2"
    assert metadata["last_modified"].endswith("+00:00")
