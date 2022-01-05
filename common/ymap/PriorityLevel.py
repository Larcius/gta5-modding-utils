class PriorityLevel:
    REQUIRED = "PRI_REQUIRED"
    HIGH = "PRI_OPTIONAL_HIGH"
    MEDIUM = "PRI_OPTIONAL_MEDIUM"
    LOW = "PRI_OPTIONAL_LOW"

    @staticmethod
    def getLevel(lodDistance: float, hasParent: bool) -> str:
        # there is a bug with level low so don't use it:
        # https://gtaforums.com/topic/919609-relv-v-proprestore/
        # further testing revealed that sometimes props with level MEDIUM or HIGH don't spawn either
        # so just use REQUIRED do avoid issues with <priorityLevel>
        return PriorityLevel.REQUIRED
        # if hasParent or lodDistance >= 100:
        #     return PriorityLevel.REQUIRED
        # elif lodDistance >= 50:
        #     return PriorityLevel.HIGH
        # elif lodDistance >= 25:
        #     return PriorityLevel.MEDIUM
        # else:
        #     return PriorityLevel.LOW
