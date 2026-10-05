"""Extract local Intel HEX samples for comparison with supported MCP readback."""
import argparse
import hashlib
import json
from pathlib import Path


def read_hex(path):
    image = {}
    base = 0
    for line in Path(path).read_text().splitlines():
        record = bytes.fromhex(line[1:])
        assert sum(record) % 256 == 0, "Invalid HEX checksum"
        count, kind = record[0], record[3]
        address = int.from_bytes(record[1:3], "big")
        if kind == 4:
            base = int.from_bytes(record[4:6], "big") << 16
        elif kind == 2:
            base = int.from_bytes(record[4:6], "big") << 4
        elif kind == 0:
            image.update((base + address + i, byte) for i, byte in enumerate(record[4:4 + count]))
    return image


parser = argparse.ArgumentParser()
parser.add_argument("--hp-address", type=lambda s: int(s, 0))
parser.add_argument("--output", required=True)
args = parser.parse_args()
samples = []
for core, path in [("M55_HP", "out/cmsis-executorch/DevKit-E8/Release/cmsis-executorch.hex"),
                   ("M55_HE", "out/M55_HE/DevKit-E8/Release/M55_HE.hex")]:
    image = read_hex(path)
    address = args.hp_address if core == "M55_HP" and args.hp_address else min(image)
    address &= ~1  # Clear the Thumb function-address bit.
    expected = bytes(image[address + i] for i in range(64))
    samples.append({"core": core, "image": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "address": hex(address), "length": 64, "expected_hex": expected.hex()})
Path(args.output).write_text(json.dumps(samples, indent=2) + "\n")
print(json.dumps(samples, indent=2))
