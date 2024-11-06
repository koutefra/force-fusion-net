from dataclasses import dataclass

@dataclass(frozen=True)
class RawScenes:

    @dataclass(frozen=True)
    class SceneData:
        id: int
        person_id: int
        start_frame_number: int
        end_frame_number: int
        fps: float
        tag: int | list[int] | None

    @dataclass(frozen=True)
    class TrackData:
        frame_number: int
        person_id: int
        x: float
        y: float

    scenes: list[SceneData]
    tracks: list[TrackData]