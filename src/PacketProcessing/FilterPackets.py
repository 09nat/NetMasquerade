"""Convert pcap/pcapng files to the 13-column CSV consumed by GenerateFlows.

Example::

    python -m PacketProcessing.FilterPackets --pcap trace.pcap --out csv/
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess
import tempfile

from .FilterPacketsHelper import get_pcap_files, get_tshark_options, preprocess_line


DEFAULT_TSHARK_OPTIONS = Path(__file__).with_name("TsharkOptions.txt")


def extract_packet_csv(pcap_path, output_dir, tshark_options):
    """Extract and normalize one capture; return its output CSV path."""
    output_dir = Path(output_dir)
    output_path = output_dir / (Path(pcap_path).name + ".csv")
    raw_fd, raw_name = tempfile.mkstemp(prefix=".tshark-", suffix=".csv",
                                        dir=str(output_dir))
    os.close(raw_fd)
    try:
        with open(raw_name, "w") as raw_file:
            subprocess.run(
                ["tshark", "-r", pcap_path] + list(tshark_options),
                stdout=raw_file,
                check=True,
                text=True,
            )
        with open(raw_name) as source, output_path.open("w") as destination:
            for line in source:
                row = preprocess_line(line)
                if row is not None:
                    destination.write(row)
    finally:
        try:
            os.unlink(raw_name)
        except FileNotFoundError:
            pass
    return str(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", required=True,
                        help="pcap/pcapng file, directory, or path-list file")
    parser.add_argument("--out", required=True, help="output CSV directory")
    parser.add_argument("--tshark-options",
                        default=str(DEFAULT_TSHARK_OPTIONS),
                        help="tshark field-selection file")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel tshark processes")
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be positive")
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    pcaps = get_pcap_files(args.pcap)
    if not pcaps:
        parser.error("no pcap/pcapng files found")
    options = get_tshark_options(args.tshark_options)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_packet_csv, path, str(output_dir), options)
                   for path in pcaps]
        for future in futures:
            print("wrote {}".format(future.result()))


if __name__ == "__main__":
    main()
