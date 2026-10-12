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
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pipeline.hostname_support import normalize_hostname, run_onnx


# ===== Sửa đường dẫn tại đây, sau đó chỉ cần: python test_release_observations.py =====
ROOT = Path(r"D:\01.AI_Security\02.SDC")
DATA_PATH = ROOT / "device_observations.jsonl"
MODEL_PATH = ROOT / "Models" / "20261012_142446_mdns_oui" / "release" / "sdc_os.onnx"
# Mỗi lần chạy tạo một thư mục mới: Reports\<YYYYmmdd_HHMMSS>\release_observations.csv
REPORT_DIR = ROOT / "Reports"
OUTPUT_PATH = REPORT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S") / "release_observations.csv"
# ======================================================================================

MISSING = "<missing>"
TOKEN_SPLIT = re.compile(r"[.\-]")
MODEL_RE = re.compile(r"(?:^|[;\s])model=([^;\s]+)", re.IGNORECASE)
# Xiaomi `_mi-connect._udp`: tên instance là JSON, `nm` là tên máy, vd.
# {"nm":"Redmi Note 10","as":"[8194]","ip":"122"}._mi-connect._udp.local
MI_CONNECT_RE = re.compile(r'\{.*?"nm"\s*:\s*"([^"]+)".*\}\._mi-connect\._udp\.local\.?', re.IGNORECASE)
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
    # Không có `model=` thì lấy tên máy Xiaomi, chuẩn hoá như hostname (bảng L0 ở 09 dùng chữ thường).
    for observation in observations:
        for section in ("questions", "answers", "authorities", "additionals"):
            for entry in observation.get(section, []):
                match = MI_CONNECT_RE.fullmatch(str(entry.get("name", "")))
                if match:
                    return normalize_hostname(match.group(1)) or MISSING
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
        "dhcp_hostname": normalize_hostname(host) or MISSING,
        "mdns_model": mdns_model(mdns),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    print(f"Data:   {args.data}")
    print(f"Model:  {args.model}")
    print(f"Output: {args.output}")

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

    session = ort.InferenceSession(str(args.model))
    predictions = run_onnx(session, ["output"], {"input": np.asarray(paired, dtype=object)})[0]

    output_rows = []
    for (index, case, observation, row), prediction in zip(metadata, predictions):
        output_rows.append(
            {
                "observation": index,
                "mac": observation.get("mac", ""),
                "source_hostname": observation.get("hostname", ""),
                "case": case,
                **dict(zip(HEADS, prediction.astype(str))),
                # Đủ 12 cột input model nhận (TLS, token DNS/mDNS/SNI, L0...), đúng thứ tự contract.
                **{("input_dhcp_hostname" if c == "dhcp_hostname" else c): row[c] for c in COLUMNS},
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
