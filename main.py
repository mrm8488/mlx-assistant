import argparse

from services.speech_to_text import SpeechToText
from services.text_gen import TextGen


def parse_arguments():
    parser = argparse.ArgumentParser(description="MLX Assistant")
    parser.add_argument(
        "--text_model", "-t", type=str, default="microsoft/phi-2", help="Model ID/Path"
    )
    parser.add_argument(
        "--speech_model", "-s", type=str, default="tiny", help="Whisper model size"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    text_gen_model = TextGen(args.text_model)
    speech_to_text_model = SpeechToText(args.speech_model)


if __name__ == "__main__":
    main()
