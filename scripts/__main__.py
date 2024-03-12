import argparse
from pathlib import Path

from scripts.data import Video2Frame


def preprocess(args):
    ...


def train(args):
    ...


def inference(args):
    ...


def main():
    parser = argparse.ArgumentParser(prog="Face Swap implementation")
    sub_parser = parser.add_subparsers()

    preprocess_parser = sub_parser.add_parser("preprocess")
    preprocess_parser.set_defaults(func=preprocess)
    preprocess_parser.add_argument("--data_root", type=Path)

    train_parser = sub_parser.add_parser("train")
    train_parser.set_defaults(func=train)

    inference_parser = sub_parser.add_parser("inference")
    inference_parser.set_defaults(func=inference)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
