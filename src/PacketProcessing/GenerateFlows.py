"""Convert FilterPackets CSV files to ``list[Flow]`` pickles.

Example::

    python -m PacketProcessing.GenerateFlows --csv csv/ --out flows/
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .Packet import Packet
from .Flow import DEFAULT_PACKET_THRESHOLD, packetsToFlows_and_write


def get_csv_files(source):
    source = Path(source)
    if source.is_file() and source.suffix == ".csv":
        return [source]
    if source.is_dir():
        return sorted(path for path in source.iterdir()
                      if path.is_file() and path.suffix == ".csv")
    raise FileNotFoundError(source)


def generate_flow(csv_path, output_dir, timegap, threshold, max_packets):
    packets = []
    with open(csv_path) as input_file:
        for count, line in enumerate(input_file, start=1):
            if max_packets is not None and count > max_packets:
                break
            fields = line.rstrip("\n").split(",")
            if len(fields) == 13:
                packets.append(Packet(fields))
    output_path = Path(output_dir) / Path(csv_path).name
    return packetsToFlows_and_write(packets, timegap, str(output_path), threshold)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV file or directory")
    parser.add_argument("--out", required=True, help="output flow directory")
    parser.add_argument("--timegap", type=float, default=300.0,
                        help="flow idle timeout in seconds")
    parser.add_argument("--threshold", type=int, default=DEFAULT_PACKET_THRESHOLD,
                        help="maximum packets per flow (default: 510)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-packets", type=int, default=0,
                        help="maximum rows read per CSV; 0 (default) disables the limit")
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.threshold < 2:
        parser.error("--threshold must be at least 2")
    if args.timegap < 0:
        parser.error("--timegap must be non-negative")
    max_packets = None if args.max_packets == 0 else args.max_packets
    if max_packets is not None and max_packets < 1:
        parser.error("--max-packets must be non-negative")

    csv_files = get_csv_files(args.csv)
    if not csv_files:
        parser.error("no CSV files found")
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate_flow, str(path), str(output_dir),
                                   args.timegap, args.threshold, max_packets)
                   for path in csv_files]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
