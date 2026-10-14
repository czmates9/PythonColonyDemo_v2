from argparse import ArgumentParser, Namespace
from json import dump
from pathlib import Path

from src.image_processing import process_image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Detect circular colony candidates in microscopy images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory for processed images and JSON report.",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=150.0,
        help="Minimum average brightness inside a detected circle.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=10.0,
        help="Minimum brightness difference between circle and surroundings.",
    )
    return parser.parse_args()


def find_images(input_directory: Path) -> list[Path]:
    if not input_directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")
    if not input_directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_directory}")

    return sorted(
        image_path
        for image_path in input_directory.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def save_report(output_directory: Path, results: list) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "report.json"

    report = {
        "processed_files": len(results),
        "total_detected": sum(result.detected_count for result in results),
        "files": [result.to_dict() for result in results],
    }

    with report_path.open("w", encoding="utf-8") as file:
        dump(report, file, indent=2, ensure_ascii=False)

    print(f"Report saved to: {report_path}")


def main() -> None:
    args = parse_arguments()

    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Minimum brightness: {args.brightness}")
    print(f"Minimum contrast: {args.contrast}")

    image_paths = find_images(args.input)

    if not image_paths:
        print("No supported images found.")
        return

    results = []

    for image_path in image_paths:
        print()
        print(f"Processing: {image_path.name}")

        try:
            result = process_image(
                image_path=image_path,
                output_directory=args.output,
                min_brightness=args.brightness,
                min_contrast=args.contrast,
            )
            results.append(result)
            print(f"Detected: {result.detected_count}")
        except Exception as exception:
            print(f"Failed: {image_path.name}")
            print(f"Reason: {exception}")

    save_report(args.output, results)

    print()
    print(f"Processed images: {len(results)}")
    print(
        "Total detected candidates: "
        f"{sum(result.detected_count for result in results)}"
    )


if __name__ == "__main__":
    main()
