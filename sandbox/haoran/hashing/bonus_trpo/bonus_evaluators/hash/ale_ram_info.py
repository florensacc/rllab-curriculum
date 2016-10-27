montezuma_revenge = {
    "room": dict(
        index=3,
        values=range(24),
        value_type="range",
    ),
    "x": dict(
        index=42,
        values=range(0,152),
        value_type="range",
    ),
    "y": dict(
        index=43,
        values=range(148,256),
        value_type="range",
    ),
    "objects": dict(
        index=67,
        values=range(16,32),
        value_type="categorical",
    ), # 1st level: doors, skeleton, key
    "skeleton_location": dict(
        index=47,
        values=range(20,80), # not exactly the min/max, but good enough
        value_type="range",
    ),
    "beam_wall": dict(
        index=27,
        values=[253,209],
        value_type="categorical",
        meanings=["off","on"]
    ),
    "beam_countdown": dict(
        index=83,
        values=range(37),
        value_type="range",
    ),
}
