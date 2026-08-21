from fidmem.data.segmentation import segment_timestamps


def test_segments_cover_timeline_without_large_gaps() -> None:
    """A regression against dropping time between sparse shot boundaries."""
    events = segment_timestamps(
        120.0,
        shots=(0.0, 12.0, 55.0, 120.0),
        speech_breaks=(28.0, 82.0),
        min_sec=8,
        max_sec=40,
    )

    assert events[0].start_sec == 0.0
    assert events[-1].end_sec == 120.0
    assert all(8 <= event.duration_sec <= 40 for event in events[:-1])
    assert all(
        right.start_sec - left.end_sec <= 0.5
        for left, right in zip(events, events[1:])
    )


def test_segments_prefer_a_nearby_speech_pause_over_a_shot_boundary() -> None:
    """A regression against ignoring ASR pauses when a valid pause exists."""
    events = segment_timestamps(
        60.0,
        shots=(0.0, 20.0, 40.0, 60.0),
        speech_breaks=(29.0,),
        min_sec=8,
        max_sec=40,
    )

    assert [(event.start_sec, event.end_sec) for event in events] == [
        (0.0, 29.0),
        (29.0, 60.0),
    ]


def test_segments_a_short_timeline_at_an_internal_cue() -> None:
    """A regression against ignoring a cue just because no hard cut is needed."""
    events = segment_timestamps(35.0, speech_breaks=(15.0,))

    assert [(event.start_sec, event.end_sec) for event in events] == [
        (0.0, 15.0),
        (15.0, 35.0),
    ]


def test_segments_cue_aware_final_tail_before_hard_cut_is_needed() -> None:
    """A regression against treating the final <=40s timeline remainder as opaque."""
    events = segment_timestamps(70.0, speech_breaks=(30.0, 55.0))

    assert [(event.start_sec, event.end_sec) for event in events] == [
        (0.0, 30.0),
        (30.0, 55.0),
        (55.0, 70.0),
    ]
