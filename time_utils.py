from datetime import datetime, timedelta, timezone


CHINA_TIMEZONE = timezone(timedelta(hours=8), name="CST")


def now_china() -> datetime:
    return datetime.now(CHINA_TIMEZONE)


def now_china_iso() -> str:
    return now_china().isoformat()
