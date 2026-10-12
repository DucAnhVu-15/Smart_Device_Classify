"""Run an SDC release ONNX model against nested device-observation JSONL.

The input JSONL contains one device/window per line.  This adapter converts the
nested DHCP/DNS/mDNS/TLS observations to the release's 12-column contract and
runs a paired hostname ablation (hostname present vs. ``<missing>``).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort


MISSING = "<missing>"
TOKEN_SPLIT = re.compile(r"[.\-]")
MODEL_RE = re.compile(r"(?:^|[;\s])model=([^;\s]+)", re.IGNORECASE)
COLUMNS = (
    "dhcp_prl",
    "tls_fp",
    "tls_version",
    "tls_alpn",
    "tls_ciphers",
    "dhcp_vci",
    "dns_tokens",
    "mdns_tokens",
    "tls_sni_tokens",
    "mac_oui",
    "dhcp_hostname",
    "mdns_model",
)
HEADS = ("make", "type", "model")


def mode(values):
    clean = [str(v) for v in values if v is not None and str(v).strip()]
    if not clean:
        return MISSING
    counts = Counter(clean)
    peak = max(counts.values())
    return sorted(v for v, count in counts.items() if count == peak)[0]


def comma_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    text = str(value).strip()
    return text or None


def domain_tokens(names, max_tokens=40):
    tokens = []
    for name in names:
        if name is None or not str(name).strip():
            continue
        normalized = str(name).lower().rstrip(".")
        tokens.extend(token for token in TOKEN_SPLIT.split(normalized) if token)
    unique = list(dict.fromkeys(tokens))[:max_tokens]
    return " ".join(unique) if unique else MISSING


def section_names(observations, section="questions"):
    return (
        entry.get("name")
        for observation in observations
        for entry in observation.get(section, [])
    )


def mdns_model(observations):
    for observation in observations:
        for section in ("answers", "authorities", "additionals"):
            for entry in observation.get(section, []):
                match = MODEL_RE.search(str(entry.get("value", "")))
                if match:
                    return match.group(1).strip().strip(".") or MISSING
    return MISSING


def mac_oui(mac):
    text = str(mac or "").strip().lower().replace("-", ":")
    if not re.fullmatch(r"[0-9a-f]{2}(?::[0-9a-f]{2}){5}", text):
        return MISSING
    # This is the convention used by the release notebook: randomized/local
    # MACs have no meaningful vendor OUI and are represented as missing.
    if int(text[:2], 16) & 0x02:
        return MISSING
    return text[:8]


def feature_row(observation):
    dhcp = observation.get("dhcp", [])
    dns = observation.get("dns", [])
    mdns = observation.get("mdns", [])
    tls = observation.get("tls", [])

    prl = mode(
        comma_list(record.get("parameter_request_list"))
        for record in dhcp
        if record.get("parameter_request_list")
    )
    vci = mode(record.get("vendor_class_id") for record in dhcp)
    tls_versions = [str(record.get("legacy_version", "")).lower() for record in tls]
    tls_ciphers = [comma_list(record.get("cipher_suites")) for record in tls]
    tls_alpn = [comma_list(record.get("alpn")) for record in tls]
    fingerprints = [
        f"{version}|{ciphers}|{alpn or ''}"
        for version, ciphers, alpn in zip(tls_versions, tls_ciphers, tls_alpn)
        if version and ciphers
    ]
    host = mode(record.get("hostname") for record in dhcp)
    if host == MISSING:
        host = str(observation.get("hostname") or "").strip() or MISSING

    return {
        "dhcp_prl": prl,
        "tls_fp": mode(fingerprints),
        "tls_version": mode(tls_versions),
        "tls_alpn": mode(value.lower() if value else None for value in tls_alpn),
        "tls_ciphers": mode(tls_ciphers),
        "dhcp_vci": vci,
        "dns_tokens": domain_tokens(section_names(dns)),
        "mdns_tokens": domain_tokens(section_names(mdns)),
        "tls_sni_tokens": domain_tokens(
            name for record in tls for name in record.get("server_names", [])
        ),
        "mac_oui": mac_oui(observation.get("mac")),
        "dhcp_hostname": host,
        "mdns_model": mdns_model(mdns),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observations = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    features = [feature_row(observation) for observation in observations]
    paired = []
    metadata = []
    for index, (observation, row) in enumerate(zip(observations, features)):
        for case in ("with_hostname", "without_hostname"):
            current = dict(row)
            if case == "without_hostname":
                current["dhcp_hostname"] = MISSING
            paired.append([current[column] for column in COLUMNS])
            metadata.append((index, case, observation, current))

    session = ort.InferenceSession(str(args.release / "sdc_os.onnx"))
    predictions = session.run(["output"], {"input": np.asarray(paired, dtype=object)})[0]

    output_rows = []
    for (index, case, observation, row), prediction in zip(metadata, predictions):
        output_rows.append(
            {
                "observation": index,
                "mac": observation.get("mac", ""),
                "source_hostname": observation.get("hostname", ""),
                "case": case,
                "input_dhcp_hostname": row["dhcp_hostname"],
                "dhcp_prl": row["dhcp_prl"],
                "dhcp_vci": row["dhcp_vci"],
                "mac_oui": row["mac_oui"],
                **dict(zip(HEADS, prediction.astype(str))),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    print(json.dumps(output_rows, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
