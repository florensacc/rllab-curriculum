import datetime
import sys


# Returns escape codes from format codes
def esc(*x):
    return '\033[' + ';'.join(x) + 'm'

# The initial list of escape codes
escape_codes = {
    'reset': esc('0'),
    'bold': esc('01'),
}

# The color names
COLORS = [
    'black',
    'red',
    'green',
    'yellow',
    'blue',
    'purple',
    'cyan',
    'white'
]

PREFIXES = [
    # Foreground without prefix
    ('3', ''), ('01;3', 'bold_'),

    # Foreground with fg_ prefix
    ('3', 'fg_'), ('01;3', 'fg_bold_'),

    # Background with bg_ prefix - bold/light works differently
    ('4', 'bg_'), ('10', 'bg_bold_'),
]

for prefix, prefix_name in PREFIXES:
    for code, name in enumerate(COLORS):
        escape_codes[prefix_name + name] = esc(prefix + str(code))


def _parse_colors(sequence):
    """Return escape codes from a color sequence."""
    return ''.join(escape_codes[n] for n in sequence.split(',') if n)


def colorize(text, color, stream=sys.stdout):
    """
    if stream is a TTY, returns the text with terminal code codes.
    otherwise, returns the text unchanged.

    colors: {black, red, green, yellow, blue, purple, cyan, white}
    examples of color combinations:
    - blue
    - black,bg_green
    - bold_red
    - bg_bold_cyan
    - bold_white,bg_blue

    source: https://github.com/borntyping/python-colorlog/
    """
    if stream.isatty():
        return ''.join([_parse_colors(color), text, _parse_colors('reset')])
    return text

seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def convert_to_seconds(s):
    """
    converts a string like "5s" or "1h" to seconds.
    source: http://stackoverflow.com/questions/3096860/convert-time-string-expressed-as-numbermhdsw-to-seconds-in-python
    """
    unit = s[-1].lower()
    if unit in seconds_per_unit:
        return int(s[:-1]) * seconds_per_unit[unit]
    return int(s)


def time_from_now(s):
    """
    given an offset in seconds, returns the datetime from now
    """
    return datetime.datetime.utcnow() + datetime.timedelta(seconds=s)


def get_index_from_host(host):
    """
    assumes the first part of the host is the index of the host.
    e.g. 10.cirrascale.sci.openai-tech.com has index 10
    """
    (name, _, _) = host.partition('.')
    try:
        return int(name)
    except ValueError:
        return name
