from dataclasses import asdict, dataclass


@dataclass
class CircleCandidate:
    x: int
    y: int
    radius: int
    inside_brightness: float
    surrounding_brightness: float
    contrast: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageProcessingResult:
    file_name: str
    detected_count: int
    candidates: list[CircleCandidate]

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "detected_count": self.detected_count,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
