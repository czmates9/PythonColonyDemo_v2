from pathlib import Path

import cv2
import numpy as np

from src.models import CircleCandidate, ImageProcessingResult


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Image could not be loaded: {image_path}"
        )

    return image


def prepare_image(
    image: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, _ = gray_image.shape
    roi_height = int(height * 0.88)

    roi = gray_image[:roi_height, :]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(roi)

    blurred = cv2.medianBlur(
        enhanced,
        9
    )

    return roi, blurred


def detect_circles(
    image: np.ndarray
) -> np.ndarray | None:
    return cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=70,
        param1=100,
        param2=33,
        minRadius=20,
        maxRadius=55
    )


def calculate_circle_brightness(
    roi: np.ndarray,
    x: int,
    y: int,
    radius: int
) -> tuple[float, float, float]:
    inner_mask = np.zeros_like(roi)
    ring_mask = np.zeros_like(roi)

    cv2.circle(
        inner_mask,
        (x, y),
        radius,
        255,
        -1
    )

    cv2.circle(
        ring_mask,
        (x, y),
        radius + 15,
        255,
        -1
    )

    cv2.circle(
        ring_mask,
        (x, y),
        radius + 3,
        0,
        -1
    )

    inside_brightness = cv2.mean(
        roi,
        mask=inner_mask
    )[0]

    surrounding_brightness = cv2.mean(
        roi,
        mask=ring_mask
    )[0]

    contrast = (
        inside_brightness
        - surrounding_brightness
    )

    return (
        float(inside_brightness),
        float(surrounding_brightness),
        float(contrast)
    )


def filter_candidates(
    roi: np.ndarray,
    circles: np.ndarray | None,
    min_brightness: float,
    min_contrast: float
) -> list[CircleCandidate]:
    if circles is None:
        return []

    detected_circles = np.round(
        circles[0]
    ).astype(int)

    accepted_circles: list[CircleCandidate] = []

    for x, y, radius in detected_circles:
        (
            inside_brightness,
            surrounding_brightness,
            contrast
        ) = calculate_circle_brightness(
            roi,
            int(x),
            int(y),
            int(radius)
        )

        if (
            inside_brightness >= min_brightness
            and contrast >= min_contrast
        ):
            accepted_circles.append(
                CircleCandidate(
                    x=int(x),
                    y=int(y),
                    radius=int(radius),
                    inside_brightness=float(inside_brightness),
                    surrounding_brightness=float(surrounding_brightness),
                    contrast=float(contrast)
                )
            )

    return accepted_circles


def draw_results(
    image: np.ndarray,
    circles: list[CircleCandidate]
) -> np.ndarray:
    result_image = image.copy()

    for index, circle in enumerate(
        circles,
        start=1
    ):
        cv2.circle(
            result_image,
            (circle.x, circle.y),
            circle.radius,
            (0, 255, 0),
            3
        )

        cv2.circle(
            result_image,
            (circle.x, circle.y),
            3,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            result_image,
            str(index),
            (
                circle.x + circle.radius,
                circle.y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    return result_image


def process_image(
    image_path: Path,
    output_directory: Path,
    min_brightness: float,
    min_contrast: float
) -> ImageProcessingResult:
    image = load_image(image_path)

    roi, prepared_image = prepare_image(
        image
    )

    circles = detect_circles(
        prepared_image
    )

    accepted_circles = filter_candidates(
        roi,
        circles,
        min_brightness,
        min_contrast
    )

    result_image = draw_results(
        image,
        accepted_circles
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_directory
        / f"{image_path.stem}_detected.jpg"
    )

    saved = cv2.imwrite(
        str(output_path),
        result_image
    )

    if not saved:
        raise RuntimeError(
            f"Could not save output image: {output_path}"
        )

    return ImageProcessingResult(
        file_name=image_path.name,
        detected_count=len(
            accepted_circles
        ),
        candidates=accepted_circles
    )
