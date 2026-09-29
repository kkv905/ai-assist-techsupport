"""Export Mermaid descriptions for both M6B3 graphs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.agent_graph import custom_graph, prebuilt_graph  # noqa: E402


def main() -> None:
    docs = ROOT / "docs"
    for graph, name in ((custom_graph, "custom"), (prebuilt_graph, "prebuilt")):
        (docs / f"agent-graph-{name}.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")


if __name__ == "__main__":
    main()
