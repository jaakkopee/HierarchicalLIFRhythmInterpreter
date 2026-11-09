#!/usr/bin/env python3
import argparse
import os
import sys
import time

import rtmidi


def list_ports(midi_in):
    try:
        count = int(midi_in.getPortCount())
    except Exception as exc:
        print(f"Failed to count MIDI input ports: {exc}")
        return []

    names = []
    for idx in range(count):
        try:
            name = str(midi_in.getPortName(idx))
        except Exception:
            name = f"port {idx}"
        names.append((idx, name))
    return names


def select_port(port_names, preferred=None):
    if preferred:
        preferred_lower = preferred.lower()
        for idx, name in port_names:
            if preferred_lower in name.lower():
                return idx, name
    for idx, name in port_names:
        lowered = name.lower()
        if any(token in lowered for token in ("mpd", "controller", "launchpad", "iac", "bus", "midi")):
            return idx, name
    return 0, port_names[0][1] if port_names else None


def decode_message(raw):
    if raw is None:
        return None
    if isinstance(raw, tuple):
        if len(raw) >= 3 and all(isinstance(v, (int,)) for v in raw[:3]):
            return list(map(int, raw[:3]))
        for candidate in raw:
            decoded = decode_message(candidate)
            if decoded is not None:
                return decoded
        return None
    if isinstance(raw, (list, bytes, bytearray)):
        data = list(raw)
        if len(data) >= 3 and all(isinstance(v, (int,)) for v in data[:3]):
            return [int(data[0]), int(data[1]), int(data[2])]
    if isinstance(raw, dict):
        if {"status", "data1", "data2"}.issubset(raw):
            return [int(raw["status"]), int(raw["data1"]), int(raw["data2"])]
    return None


def format_status(status):
    channel = status & 0x0F
    type_ = status & 0xF0
    names = {
        0x80: "Note Off",
        0x90: "Note On",
        0xA0: "Poly Aftertouch",
        0xB0: "Control Change",
        0xC0: "Program Change",
        0xD0: "Channel Pressure",
        0xE0: "Pitch Bend",
    }
    return f"{names.get(type_, f'0x{type_:02X}')} ch={channel}"


def listen(port_name=None, duration=20.0):
    try:
        midi_in = rtmidi.RtMidiIn()
    except Exception as exc:
        print(f"Unable to create RtMidiIn: {exc}")
        return 1

    ports = list_ports(midi_in)
    if not ports:
        print("No MIDI input ports found.")
        return 1

    print("Available MIDI input ports:")
    for idx, name in ports:
        print(f"  [{idx}] {name}")

    preferred = port_name or os.environ.get("MPD218_TEST_PORT")
    if preferred:
        print(f"Requested port filter: {preferred}")

    selected_idx, selected_name = select_port(ports, preferred)
    print(f"Opening: {selected_name} (index {selected_idx})")

    try:
        midi_in.openPort(selected_idx)
    except Exception as exc:
        print(f"Could not open port {selected_idx}: {exc}")
        return 1

    start = time.time()
    print(f"Listening for {duration:.1f} seconds. Move any knob or press a key...\n")
    got_any = False

    while time.time() - start < duration:
        try:
            message = midi_in.getMessage(10)
        except TypeError:
            try:
                message = midi_in.getMessage(10)
            except Exception:
                time.sleep(0.01)
                continue
        except Exception:
            time.sleep(0.01)
            continue

        if message is None:
            time.sleep(0.01)
            continue

        decoded = decode_message(message)
        got_any = True
        if decoded is None:
            print(f"Raw message (unparsed): {message!r}")
            continue

        status, data1, data2 = decoded[:3]
        print(
            f"RAW status=0x{status:02X} ({format_status(status)}) "
            f"data1={data1} data2={data2}"
        )

        if (status & 0xF0) == 0xB0:
            print(f"  -> CC message: controller={data1}, value={data2}")
        elif (status & 0xF0) == 0xE0:
            print(f"  -> Pitch bend: value={((data2 << 7) | data1)}")
        elif (status & 0xF0) == 0xD0:
            print(f"  -> Channel pressure: value={data1}")
        elif (status & 0xF0) in (0x90, 0x80):
            print(f"  -> Note {'on' if (status & 0xF0) == 0x90 else 'off'}: note={data1}, vel={data2}")

    if not got_any:
        print("No MIDI messages were received during the listening window.")
        print("This means the port is either wrong, the MPD218 is not sending, or it is set to a different MIDI mode.")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Standalone MPD218 MIDI input probe")
    parser.add_argument("--port", default=None, help="Partial port name to prefer, e.g. 'MPD218 Port A'")
    parser.add_argument("--seconds", type=float, default=20.0, help="How many seconds to listen before exiting")
    args = parser.parse_args()

    print("MPD218 MIDI probe")
    print("Tip: if your device has multiple ports, try --port 'MPD218 Port A' or set MPD218_TEST_PORT.")
    return listen(port_name=args.port, duration=args.seconds)


if __name__ == "__main__":
    sys.exit(main())
