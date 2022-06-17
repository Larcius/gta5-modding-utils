class Flag:
    ALLOW_FULL_ROTATION = 1 << 0
    DISABLE_EMBEDDED_COLLISION = 1 << 2
    LOD_IN_PARENT = 1 << 3
    UNKNOWN_4 = 1 << 4
    STATIC_ENTITY = 1 << 5
    UNKNOWN_19 = 1 << 19
    UNKNOWN_20 = 1 << 20
    UNKNOWN_21 = 1 << 21

    FLAGS_SLOD4 = FLAGS_SLOD3 = FLAGS_SLOD2 = FLAGS_LOD = UNKNOWN_19 | UNKNOWN_20
    FLAGS_SLOD1 = UNKNOWN_4 | LOD_IN_PARENT | FLAGS_LOD

    FLAGS_HD_EXCLUDE_DEFAULT = STATIC_ENTITY
    FLAGS_HD_DEFAULT = LOD_IN_PARENT | FLAGS_LOD

    FLAGS_ORPHANHD_EXCLUDE_DEFAULT = STATIC_ENTITY
    FLAGS_ORPHANHD_DEFAULT = ALLOW_FULL_ROTATION | UNKNOWN_20 | UNKNOWN_21
