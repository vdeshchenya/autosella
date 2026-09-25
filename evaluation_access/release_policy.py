"""Dataset releases withdrawn from research use by the dataset owner."""

CURRENT_RELEASE_ID = "main-f59f2b89a-gfn2-v1"
RETIRED_RELEASE_IDS = frozenset({
    "main-d288fbec4-gfn2-v1",
    "main-f475a971a-gfn2-v1",
    "main-15575704b-gfn2-v1",
})


def check_release_supported(release_id: str) -> None:
    if release_id in RETIRED_RELEASE_IDS:
        raise ValueError(
            f"Dataset {release_id} was withdrawn as invalid; "
            f"create a fresh experiment using {CURRENT_RELEASE_ID}. "
            "Old anchors and scores are historical only."
        )
