class PriorityLevel:
    REQUIRED = "PRI_REQUIRED"
    HIGH = "PRI_OPTIONAL_HIGH"
    MEDIUM = "PRI_OPTIONAL_MEDIUM"
    LOW = "PRI_OPTIONAL_LOW"

    @staticmethod
    def getLevel(lodDistance: float, hasParent: bool) -> str:
        if hasParent or lodDistance >= 100:
            return PriorityLevel.REQUIRED
        elif lodDistance >= 50:
            return PriorityLevel.HIGH
        elif lodDistance >= 25:
            return PriorityLevel.MEDIUM
        else:
            # entities with priority level low seems to never show up so do not use it
            # return PriorityLevel.LOW
            return PriorityLevel.MEDIUM
