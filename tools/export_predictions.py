"""Export prediction json into flat csv/txt. TODO: extend for benchmark upload formats."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_txt", required=True)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for r in data:
            f.write(
                f"{r.get('image_id', -1)} {r.get('category_id', -1)} {r.get('score', 0)} "
                f"{' '.join(str(v) for v in r.get('bbox', []))}\n"
            )


if __name__ == "__main__":
    main()
