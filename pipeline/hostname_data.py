"""Build a labeled hostname dataset from Akien and reviewed FIELD captures.

Never derive truth from a classifier, OUI, hostname rules or a device display name.
FIELD truth comes only from the verified taxonomy joined through verified sessions.
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from hostname_support import HEADS, normalize_hostname

MISSING_LABELS = {"", "null", "none", "nan", "<missing>", "<na>",
                  "__unknown__", "unknown"}
PSEUDO_MAKES = {"generic laptop", "camera", "linova/linux"}
# AKIEN gán cùng một máy hai nhãn theo cách hostname viết ("a51-cua-an" → "Samsung A51",
# "galaxy-a51-cua-an" → "Samsung Galaxy A51"); hostname không phân biệt được bản 5G.
# device_list.csv dùng đúng tên chuẩn bên phải.
MODEL_ALIAS = {
    "Samsung A10": "Samsung Galaxy A10", "Samsung A20": "Samsung Galaxy A20",
    "Samsung A21": "Samsung Galaxy A21", "Samsung A33": "Samsung Galaxy A33 5G",
    "Samsung A50": "Samsung Galaxy A50", "Samsung A51": "Samsung Galaxy A51",
    "Samsung A52s": "Samsung Galaxy A52s 5G", "Samsung A70": "Samsung Galaxy A70",
    "Samsung Galaxy Note20 5G": "Samsung Galaxy Note20",
    "Samsung Galaxy Note20 Ultra 5G": "Samsung Galaxy Note20 Ultra",
    "Samsung Galaxy S10 5G": "Samsung Galaxy S10", "Samsung Galaxy S20 FE 5G": "Samsung Galaxy S20 FE",
}


def clean_label(value, head):
    if pd.isna(value) or normalize_hostname(value) in MISSING_LABELS:
        return pd.NA
    label = str(value).strip()
    if head == "make":
        if label.casefold() in PSEUDO_MAKES or label.casefold().startswith("generic "):
            return pd.NA
        return {"VIVO": "Vivo"}.get(label, label)
    if head == "model":
        return MODEL_ALIAS.get(label, label)
    return {"Mobile Phone": "Smartphone"}.get(label, label) if head == "type" else label


def hostname_template(value):
    text = normalize_hostname(value)
    text = re.sub(r"(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}", "<mac>", text)
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<uuid>", text)
    text = re.sub(r"\b[0-9a-f]{10,}\b", "<hex>", text)
    return re.sub(r"\d+", "<n>", text)


def load_akien_hostnames(path):
    path = Path(path)
    raw = pd.read_csv(path, low_memory=False)
    columns = {"hostname": "hostname", "Brand": "make",
               "DeviceType": "type", "Modelname": "model"}
    missing = set(columns) - set(raw)
    if missing:
        raise ValueError(f"Thiếu cột Data_AKien: {sorted(missing)}")
    frame = raw[list(columns)].rename(columns=columns).copy()
    frame["source"] = "AKIEN"
    frame["source_file"] = str(path)
    frame["canonical_device"] = ""
    frame["device_group"] = ""
    return frame


def load_field_hostnames(sessions_path, labels_path, dhcp_paths):
    """Match actual DHCP option-12 to reviewed FIELD devices; unknown MACs excluded."""
    sessions = pd.read_parquet(sessions_path)
    labels = pd.read_csv(labels_path)
    required = {"scenario", "mac", "capture_file", "canonical_device"}
    if not required.issubset(sessions):
        raise ValueError(f"Thiếu cột sessions: {sorted(required - set(sessions))}")
    if not {"canonical_device", *HEADS}.issubset(labels):
        raise ValueError("Bảng nhãn verified cần canonical_device, make, type, model")
    labels = labels[["canonical_device", *HEADS]].drop_duplicates()
    if labels["canonical_device"].duplicated().any():
        raise ValueError("Bảng verified có nhiều bộ nhãn cho cùng canonical_device")
    field = sessions.loc[sessions["scenario"].eq("FIELD"),
                         ["mac", "capture_file", "canonical_device"]].drop_duplicates()
    field["mac"] = field["mac"].astype(str).str.strip().str.lower()
    field["capture_file"] = field["capture_file"].astype(str).str.strip()
    if field.duplicated(["mac", "capture_file"]).any():
        raise ValueError("MAC/capture FIELD trỏ tới nhiều canonical_device")
    # Inner join is intentional: sessions labels/predictions do not establish truth.
    reviewed = field.merge(labels, on="canonical_device", how="inner", validate="many_to_one")
    parts, raw_count, unmatched = [], 0, 0
    for raw_path in dhcp_paths:
        path = Path(raw_path)
        raw = pd.read_csv(path, low_memory=False)
        host_col = "hostname" if "hostname" in raw else "dhcp_hostname"
        if not {"mac", "capture_file", host_col}.issubset(raw):
            raise ValueError(f"DHCP cần mac/capture_file/hostname: {path}")
        raw_count += len(raw)
        dhcp = raw[["mac", "capture_file", host_col]].rename(columns={host_col: "hostname"}).copy()
        dhcp["mac"] = dhcp["mac"].astype(str).str.strip().str.lower()
        dhcp["capture_file"] = dhcp["capture_file"].astype(str).str.strip()
        joined = dhcp.merge(reviewed, on=["mac", "capture_file"], how="left",
                            validate="many_to_one", indicator=True)
        unmatched += int(joined["_merge"].ne("both").sum())
        joined = joined.loc[joined["_merge"].eq("both")].drop(columns="_merge")
        joined["source"] = "FIELD"
        joined["source_file"] = str(path)
        joined["device_group"] = joined["canonical_device"].map(
            lambda name: "FIELD:" + hashlib.sha256(str(name).encode()).hexdigest()[:20])
        parts.append(joined)
    if not parts:
        raise ValueError("Chưa cấu hình nguồn DHCP tự thu")
    frame = pd.concat(parts, ignore_index=True)
    return frame, {"dhcp_raw_rows": raw_count, "unmatched_or_unreviewed_rows": unmatched,
                   "matched_packet_rows": len(frame), "verified_devices": reviewed["canonical_device"].nunique()}


def prepare_hostname_dataset(akien, field):
    """Deduplicate across sources; mask only conflicting heads, not valid other labels."""
    raw = pd.concat([akien, field], ignore_index=True, sort=False)
    for col in ("device_group", "canonical_device", "source_file"):
        if col not in raw:
            raw[col] = ""
        raw[col] = raw[col].fillna("")
    raw["hostname"] = raw["hostname"].map(lambda value: "" if pd.isna(value) else normalize_hostname(value))
    for head in HEADS:
        raw[head] = raw[head].map(lambda value: clean_label(value, head))
    input_rows_by_source = {str(k): int(v) for k, v in raw["source"].value_counts().items()}
    valid = raw["hostname"].str.len().between(2, 253)
    invalid_rows = int((~valid).sum())
    raw = raw.loc[valid].copy()
    # Prefer FIELD metadata on an identical hostname without overriding conflicting truth.
    raw = raw.sort_values("source", ascending=False, kind="stable")
    grouped = raw.groupby("hostname", sort=False)
    clean = grouped[[*HEADS, "source", "source_file", "canonical_device"]].first()
    counts = grouped[list(HEADS)].nunique()
    conflicts = []
    for head in HEADS:
        names = counts.index[counts[head] > 1]
        for name in names:
            rows = raw.loc[raw["hostname"].eq(name)]
            conflicts.append({"hostname": name, "head": head,
                              "labels": json.dumps(sorted(rows[head].dropna().unique()), ensure_ascii=False),
                              "sources": "|".join(sorted(rows["source"].unique())),
                              "action": "mask_conflicting_head"})
        clean.loc[names, head] = pd.NA
    both = grouped["source"].nunique() > 1
    clean["source_members"] = clean["source"]
    clean.loc[both, "source_members"] = "AKIEN|FIELD"
    device_groups = raw.loc[raw["device_group"].ne("")].groupby("hostname")["device_group"].agg(
        lambda values: "|".join(sorted(set(values))))
    clean["device_group"] = device_groups.reindex(clean.index).fillna("")
    clean = clean.reset_index()
    no_truth = clean[list(HEADS)].isna().all(axis=1)
    no_truth_count = int(no_truth.sum())
    clean = clean.loc[~no_truth].reset_index(drop=True)
    clean["template"] = clean["hostname"].map(hostname_template)
    # Connected components keep ALL templates of a FIELD device together, even when
    # its hostname changes. A shared Akien template inherits that same component.
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            following = parent[key]
            parent[key] = root
            key = following
        return root
    def union(left, right):
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)
    for row in clean.loc[clean["device_group"].ne(""), ["template", "device_group"]].itertuples(index=False):
        for device in row.device_group.split("|"):
            union("template:" + row.template, "device:" + device)
    clean["group_id"] = clean["template"].map(lambda value: find("template:" + value))
    audit = {"input_rows_by_source": input_rows_by_source,
             "invalid_hostname_rows": invalid_rows, "clean_rows": len(clean),
             "clean_rows_by_source": {str(k): int(v) for k, v in clean["source"].value_counts().items()},
             "field_unique_hostnames": int(clean["source_members"].str.contains("FIELD", regex=False).sum()),
             "cross_source_shared_hostnames": int(clean["source_members"].eq("AKIEN|FIELD").sum()),
             "conflicting_heads": len(conflicts), "removed_all_labels_missing_hostnames": no_truth_count,
             "duplicate_hostname_rows_collapsed": len(raw) - grouped.ngroups}
    return clean, pd.DataFrame(conflicts, columns=["hostname", "head", "labels", "sources", "action"]), audit


def assert_hostname_split_isolation(splits):
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = splits[left], splits[right]
            for col in ("hostname", "template", "group_id"):
                if not set(a[col]).isdisjoint(b[col]):
                    raise AssertionError(f"Leak {col}: {left}/{right}")
            devices_a = {d for values in a["device_group"] for d in values.split("|") if d}
            devices_b = {d for values in b["device_group"] for d in values.split("|") if d}
            if not devices_a.isdisjoint(devices_b):
                raise AssertionError(f"Leak FIELD device: {left}/{right}")


def split_hostname_dataset(frame, seed=42, test_size=0.20, validation_fraction=0.20):
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(outer.split(frame, groups=frame["group_id"]))
    trainval = frame.iloc[trainval_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed + 1)
    train_idx, valid_idx = next(inner.split(trainval, groups=trainval["group_id"]))
    splits = {"train": trainval.iloc[train_idx].reset_index(drop=True),
              "validation": trainval.iloc[valid_idx].reset_index(drop=True),
              "test": frame.iloc[test_idx].reset_index(drop=True)}
    assert_hostname_split_isolation(splits)
    return splits
