from dataclasses import dataclass

@dataclass(frozen=True)

class PushAction:
    box_position: tuple[int,int]
    direction: int
    destination: tuple[int,int]
    player_target: tuple[int,int]