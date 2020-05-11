import sys
import threading

from colorama import (
    init,
    Fore,
    Style,
)

init()


class ThreadSafeOutputStream:

    def __init__(self, stream=sys.stderr):
        self.stream = stream
        self.lock = threading.Lock()

    def write(self, string, *args, color: str = None, bold: bool = False):
        reset_style = False
        if color is not None:
            code = getattr(Fore, color.upper())
            string = code + string
            reset_style = True
        if bold:
            string = Style.BRIGHT + string
            reset_style = True
        if reset_style:
            string = string + Style.RESET_ALL
        formatted_string = string % args
        with self.lock:
            self.stream.write(formatted_string)
            self.stream.flush()

    def writeln(self, *args, **kwargs):
        if args:
            string, *args = args
        else:
            string = ''
        self.write(string + '\n', *args, **kwargs)
