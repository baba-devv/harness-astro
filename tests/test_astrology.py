from datetime import date

import pytest

from app.services.astrology import sun_sign


@pytest.mark.parametrize(
    "dob, expected",
    [
        (date(1990, 1, 1), "Capricorn"),
        (date(1990, 1, 19), "Capricorn"),
        (date(1990, 1, 20), "Aquarius"),
        (date(1990, 3, 21), "Aries"),
        (date(1990, 7, 23), "Leo"),
        (date(1990, 12, 21), "Sagittarius"),
        (date(1990, 12, 31), "Capricorn"),
    ],
)
def test_sun_sign(dob, expected):
    assert sun_sign(dob) == expected
