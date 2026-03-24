import pytest

from accounts.aeo.aeo_utils import infer_city_from_address


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("123 Main St, Salt Lake City, UT", "Salt Lake City, UT"),
        ("123 Main St, Salt Lake City, UT 84101", "Salt Lake City, UT"),
        ("Austin, TX", "Austin, TX"),
        ("Paris, France", "Paris, France"),
        ("123 Main St, Austin", "Austin"),
        ("Austin", "Austin"),
        ("123 Main Street", ""),
    ],
)
def test_infer_city_from_address_examples(raw: str, expected: str) -> None:
    assert infer_city_from_address(raw) == expected
