import string
import random
from datetime import datetime, timedelta, timezone
import pytz

def create_response(message="success", data=None):
    return {
        "message": message,
        "data": data
    }

def random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def create_display_name(fname: str, mname: str, lname: str):
    # Create display name
    names = []
    names.append(fname)
    if mname:
        names.append(mname)
    names.append(lname)
    return " ".join(names)

def istimeout(date: datetime, timeout: int):
    tz = pytz.timezone("UTC")
    return tz.localize(date) + timedelta(milliseconds=timeout) <= datetime.now(timezone.utc)