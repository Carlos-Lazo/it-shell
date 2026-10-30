import argparse
from pathlib import Path

from semantic_search_engine.bootstrap import create_engine
from semantic_search_engine.shared.paths import default_config_path
from semantic_search_engine.shared.contracts import QueryRequest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Propuestas de comandos; no ejecuta shell"
    )
    parser.add_argument("query")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--format",
        choices=("json", "markdown", "cli", "terminal"),
        default="cli",
    )
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args(argv)
    response = create_engine(args.config).search(
        QueryRequest(args.query, args.top_k, args.format)
    )
    print(response)


if __name__ == "__main__":
    main()
