class EntityItem:
    archetypeName: str
    position: list[float]
    scale: list[float]
    rotation: list[float]
    lodDistance: float

    def __init__(self, archetypeName: str, position: list[float], scale: list[float], rotation: list[float], lodDistance: float):
        self.archetypeName = archetypeName
        self.position = position
        self.scale = scale
        self.rotation = rotation
        self.lodDistance = lodDistance