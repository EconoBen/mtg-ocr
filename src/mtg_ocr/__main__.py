from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mtg-ocr",
        description="MTG card visual identification using MobileCLIP embeddings",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    args = parser.parse_args()

    if args.version:
        from mtg_ocr import __version__

        print(f"mtg-ocr {__version__}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
