

class Stack(list):

    def top(self):
        return self[-1]

    def empty(self) -> bool:
        return len(self) == 0

    def add(self, item):
        return self.append(item)


def td_format(seconds: float):
    """
    >>> td_format(66666666.6666)
    '2y 41d 14h 31m 6.667s'

    >>> td_format(666666.6666)
    '7d 17h 11m 6.667s'

    >>> td_format(666)
    '11m 6.000s'
    """
    periods = (
        ('y', 60 * 60 * 24 * 365),
        ('d', 60 * 60 * 24),
        ('h', 60 * 60),
        ('m', 60),
    )
    strings = []
    for period, period_seconds in periods:
        if seconds >= period_seconds:
            num, seconds = divmod(seconds, period_seconds)
            strings.append('%i%s' % (num, period))
    strings.append('%.3fs' % seconds)
    return ' '.join(strings)
