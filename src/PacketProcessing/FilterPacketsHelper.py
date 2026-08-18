"""Helpers for extracting the packet fields consumed by :mod:`PacketProcessing`."""

from pathlib import Path
import shlex


PCAP_SUFFIXES = (".pcap", ".pcapng")


def get_pcap_files(source):
    """Return sorted pcap paths from a file, directory, or path-list file."""
    source = Path(source)
    if source.is_file() and source.suffix.lower() in PCAP_SUFFIXES:
        candidates = [source]
    elif source.is_dir():
        candidates = sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in PCAP_SUFFIXES
        )
    elif source.is_file():
        candidates = []
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            candidate = Path(line.strip())
            candidates.append(candidate if candidate.is_absolute()
                              else source.parent / candidate)
    else:
        raise FileNotFoundError(source)

    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    invalid = [path for path in candidates if path.suffix.lower() not in PCAP_SUFFIXES]
    if invalid:
        raise ValueError("not a pcap/pcapng file: {}".format(invalid[0]))
    return [str(path.resolve()) for path in candidates]


def get_tshark_options(options_file):
    """Parse the checked-in tshark options into an argv list."""
    options = []
    for line in Path(options_file).read_text().splitlines():
        if line.strip():
            options.extend(shlex.split(line))
    return options


def preprocess_line(line):
    """Convert one raw 15-column tshark row to Packet's 13-column schema."""
    fields = line.rstrip("\n").split(",")
    if len(fields) != 15:
        return None

    try:
        fields[4] = str(int(fields[4] or "0", 16))
    except ValueError:
        return None
    for index in (8, 9, 10):
        fields[index] = fields[index].strip() or "0"

    # tshark emits separate TCP and UDP ports. Keep two columns even for IP
    # fragments, where both values can be empty and Packet reuses the last pair
    # seen for the same endpoint/protocol tuple.
    src_port = fields[11].strip() or fields[13].strip()
    dst_port = fields[12].strip() or fields[14].strip()
    return ",".join(fields[:11] + [src_port, dst_port]) + "\n"

