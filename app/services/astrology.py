from datetime import date

# Simplified tropical sun-sign table: (sign, start month, start day), in calendar order - some basic stubbing
_SIGNS = [
    ("Capricorn", 1, 1),
    ("Aquarius", 1, 20),
    ("Pisces", 2, 19),
    ("Aries", 3, 21),
    ("Taurus", 4, 20),
    ("Gemini", 5, 21),
    ("Cancer", 6, 21),
    ("Leo", 7, 23),
    ("Virgo", 8, 23),
    ("Libra", 9, 23),
    ("Scorpio", 10, 23),
    ("Sagittarius", 11, 22),
    ("Capricorn", 12, 22),
]


def sun_sign(dob: date) -> str:
    """Stubbed sun sign from date only; ignores time and place of birth."""
    sign = _SIGNS[0][0]
    for name, month, day in _SIGNS:
        if (dob.month, dob.day) >= (month, day):
            sign = name
    return sign
