import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="STRIQ: Subspace-Guided Registration for US QC",
        usage="python main.py <command> [<args>]",
    )
    parser.add_argument(
        "command",
        type=str,
        choices=["train", "test", "cache"],
        help="Sub-command to execute.",
    )
    args, remaining = parser.parse_known_args()

    if args.command == "train":
        from scripts.train import main as train_main
        sys.argv = [sys.argv[0]] + remaining
        train_main()
    elif args.command == "test":
        from scripts.test import main as test_main
        sys.argv = [sys.argv[0]] + remaining
        test_main()
    elif args.command == "cache":
        from scripts.cache_features import main as cache_main
        sys.argv = [sys.argv[0]] + remaining
        cache_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
