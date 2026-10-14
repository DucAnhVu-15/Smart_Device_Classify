"""Rule A hostname từ catalog thiết bị đã duyệt (hostname_catalog.csv).

Mỗi dòng catalog là một thiết bị theo cách người dùng gõ trong hostname ("s24 ultra",
"reno 12 pro"). Regex sinh ra cần ranh giới ký tự hai đầu nên "s24" không khớp trong
"s24ultra". Cascade lấy rule khớp ĐẦU TIÊN, nên một dòng phải đứng trước mọi dòng khác
nhãn mà pattern cũng khớp hostname mẫu của nó (S24 Ultra trước S24, OPPO A18 trước A18).

Nhãn "__unknown__" là rule chặn: hostname khớp thì head đó dừng ở A, không trả nhãn và
không cho expert B đoán (camera "nomi-ipc-s21fe" không được thành Galaxy S21 FE); kết quả
gói tin giữ nguyên vì cascade chỉ ghi đè khi hostname có nhãn.

Mọi rule được đối chiếu nhãn thật train+validation của dataset hostname (05). Test chỉ để
báo cáo, không dùng chọn rule. Chạy:
    python -X utf8 Code/pipeline/hostname_catalog.py --split Models/<RUN>/train/hostname_dataset_split.csv
"""
import argparse
import ast
import csv
import hashlib
import heapq
import json
import re
from pathlib import Path

import pandas as pd

from hostname_support import HEADS, UNKNOWN, normalize_hostname

FORMAT = "sdc-hostname-catalog-v1"
SEP = r"[-_. ]?"
COLUMNS = ("group", "make", "type", "model", "names", "prefixes", "require_prefix", "pattern", "examples")
MIN_SCORED = 5
MIN_PRECISION = 0.95
MAX_UNION = 2000
# Nhãn model chung của AKIEN, không phải đời máy: dự đoán đời cụ thể cùng hãng không tính sai.
GENERIC_MODEL = {"Realme 1": "realme", "OPPO 1105": "oppo", "Xiaomi Mi Mix": "xiaomi mix"}
# Nhãn AKIEN gán cho hostname rõ ràng của hãng khác ("oneplus-9-pro" → Blu): không làm truth.
NOISY_MODEL = {"Blu Vivo One Plus"}
# AKIEN tách POCO thành hãng riêng; pipeline (rule mặc định, evidence) gộp vào Xiaomi.
MAKE_ALIAS = {"POCO": "Xiaomi"}


def split_list(value):
    return [v.strip() for v in str(value or "").split("|") if v.strip()]


def pieces(text):
    return re.findall(r"[a-z]+|[0-9]+", normalize_hostname(text))


def name_regex(name):
    return SEP.join(re.escape(p) for p in pieces(name))


def row_regex(row):
    if row["pattern"]:
        if not row["pattern"].startswith("(?i)"):
            raise ValueError(f"Pattern phải bắt đầu bằng (?i): {row['pattern']}")
        return row["pattern"]
    names = "|".join(name_regex(n) for n in row["names"])
    prefix = "|".join(name_regex(p) for p in row["prefixes"])
    lead = f"(?:{prefix}){SEP}" if prefix else ""
    if prefix and not row["require_prefix"]:
        lead = f"(?:{lead})?"
    return rf"(?i)(?:.*[^a-z0-9])?{lead}(?:{names})(?:[^a-z0-9].*)?"


def row_samples(row):
    """Hostname người dùng thiết bị này hay đặt; dùng để xếp thứ tự và trong test."""
    out = [normalize_hostname(e) for e in row["examples"]]
    prefix = "-".join(normalize_hostname(row["prefixes"][0]).split()) if row["prefixes"] else ""
    for name in row["names"]:
        words = normalize_hostname(name).split()
        bases = ["-".join(words), "".join(words)]
        if prefix:
            with_prefix = [f"{prefix}-{b}" for b in bases]
            bases = with_prefix if row["require_prefix"] else bases + with_prefix
        for base in bases:
            out += [base, f"{base}-cua-an", f"an-s-{base}"]
    return list(dict.fromkeys(out))


def load_catalog(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw or set(COLUMNS) - set(raw[0]):
        raise ValueError(f"Catalog cần cột {COLUMNS}: {path}")
    rows = []
    for line, item in enumerate(raw, start=2):
        row = {k: (item[k] or "").strip() for k in COLUMNS}
        row.update(line=line, names=split_list(row["names"]), prefixes=split_list(row["prefixes"]),
                   examples=split_list(row["examples"]), require_prefix=row["require_prefix"] in {"1", "true", "True"})
        if not (row["names"] or row["pattern"]) or not any(row[h] for h in HEADS):
            raise ValueError(f"Dòng {line}: cần names/pattern và ít nhất một nhãn make/type/model")
        if row["pattern"] and not row["examples"]:
            raise ValueError(f"Dòng {line}: pattern tự viết cần examples")
        row["regex"] = row_regex(row)
        re.compile(row["regex"])
        row["samples"] = row_samples(row)
        bad = [s for s in row["samples"] if not re.fullmatch(row["regex"], s)]
        if bad:
            raise ValueError(f"Dòng {line} ({row['model'] or row['make']}): pattern không khớp mẫu {bad[:3]}")
        rows.append(row)
    return rows


def order_rows(rows, head):
    """Thứ tự first-match-wins: A trước B khi B khớp mẫu của A mà nhãn khác nhau.

    Hòa thì pattern viết tay đi trước: chúng bám định dạng cố định (mã máy "m2006c3lg-",
    camera "nomi-ipc-", TV "lgwebos") nên tin hơn tên máy người dùng gõ phía sau
    ("m2006c3lg-redmi10a" là Redmi 9A). Sau đó (nhãn, dòng) cho make/type để rule cùng nhãn
    liền nhau rồi gộp được.
    """
    active = [r for r in rows if r[head]]
    compiled = [re.compile(r["regex"]) for r in active]
    after = {i: set() for i in range(len(active))}
    blockers = {i: 0 for i in range(len(active))}
    for i, a in enumerate(active):
        for j, b in enumerate(active):
            if i != j and a[head] != b[head] and any(compiled[j].fullmatch(s) for s in a["samples"]):
                after[i].add(j)
    for i in after:
        for j in after[i]:
            blockers[j] += 1
    def key(i):
        tier = 0 if active[i]["pattern"] else 1
        return (tier, i) if head == "model" else (tier, active[i][head], i)
    ready = [(key(i), i) for i, n in blockers.items() if n == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, i = heapq.heappop(ready)
        order.append(active[i])
        for j in after[i]:
            blockers[j] -= 1
            if blockers[j] == 0:
                heapq.heappush(ready, (key(j), j))
    if len(order) != len(active):
        stuck = [f"{active[i][head]} (dòng {active[i]['line']})" for i, n in blockers.items() if n]
        raise ValueError(f"Rule {head} chồng lấn vòng tròn, cần sửa names: {stuck[:10]}")
    return order


def build_rules(ordered):
    """Gộp các rule liền nhau cùng nhãn; giữ nguyên ngữ nghĩa first-match-wins.

    Giới hạn độ dài: pattern ~16k ký tự làm DFA của RE2 (ONNX/TF) hết bộ nhớ, rơi về NFA chậm.
    """
    rules = []
    for head, rows in ordered.items():
        for row in rows:
            body = row["regex"][len("(?i)"):]
            if (rules and rules[-1][1] == head and rules[-1][2] == row[head]
                    and len(rules[-1][0]) + len(body) < MAX_UNION):
                rules[-1][0] += f"|(?:{body})"
            else:
                rules.append([f"(?:{body})", head, row[head]])
    return [[f"(?i){p}", h, y] for p, h, y in rules]


def read_default_rules(notebook):
    """DEFAULT_HOSTNAME_RULES của 09, đọc literal bằng AST như phần C của 05."""
    document = json.loads(Path(notebook).read_text(encoding="utf-8"))
    for cell in document["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or "DEFAULT_HOSTNAME_RULES =" not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "DEFAULT_HOSTNAME_RULES" for t in node.targets):
                return [list(r) for r in ast.literal_eval(node.value)]
    raise RuntimeError(f"Không tìm thấy DEFAULT_HOSTNAME_RULES: {notebook}")


def model_tokens(label):
    """Bỏ 'Galaxy' và đuôi '5G': AKIEN viết cùng một máy thành 'Samsung A51'/'Samsung Galaxy A51'."""
    words = [w for w in normalize_hostname(label).replace("+", " plus").split() if w != "galaxy"]
    return words[:-1] if words[-1:] == ["5g"] else words


def judge(head, truth, predicted):
    """correct / compatible (nhãn truth chung hơn, cùng hãng) / wrong."""
    if truth == predicted:
        return "correct"
    if head == "make" and MAKE_ALIAS.get(truth) == predicted:
        return "compatible"
    if head == "model":
        if model_tokens(truth) == model_tokens(predicted):
            return "correct"
        parent, child = model_tokens(GENERIC_MODEL.get(truth, truth)), model_tokens(predicted)
        if parent == child[:len(parent)]:
            return "compatible"
        # 'OPPO Reno' là cha của 'OPPO Reno6' (chữ nối số), nhưng 'Redmi 9' không là cha của 'Redmi 9A'.
        a, b = "".join(parent), "".join(child)
        if len(b) > len(a) and b.startswith(a) and a[-1].isalpha() and b[len(a)].isdigit():
            return "compatible"
    return "wrong"


def row_hits(rows, hosts):
    """Host khớp từng dòng. Lọc trước bằng mảnh chữ/số dài nhất cho nhanh."""
    hits = {}
    for row in rows:
        anchors = [max(pieces(n), key=len) for n in row["names"]] if not row["pattern"] else []
        compiled = re.compile(row["regex"])
        if anchors:
            candidates = [i for i, h in enumerate(hosts) if any(a in h for a in anchors)]
        else:
            candidates = range(len(hosts))
        hits[row["line"]] = [i for i in candidates if compiled.fullmatch(hosts[i])]
    return hits


def score(rows, ordered, defaults, frame, hits):
    """Thống kê theo (dòng, head) với thứ tự cascade thật: rule mặc định trước, catalog sau."""
    hosts = frame["hostname"].tolist()
    truths = {h: frame[h].tolist() for h in HEADS}
    split = frame["split"].tolist()
    by_line = {row["line"]: row for row in rows}

    def verdict(head, i, row):
        if not pd.isna(truths[head][i]):
            return judge(head, truths[head][i], row[head])
        # Chưa có nhãn ở head này nhưng head khác mâu thuẫn ("nomi-ipc-s21fe" là camera):
        # model/make của dòng cũng sai vì hierarchy sẽ kéo type theo.
        if any(row[g] and row[g] != UNKNOWN and not pd.isna(truths[g][i])
               and judge(g, truths[g][i], row[g]) == "wrong" for g in HEADS if g != head):
            return "wrong"
        return "unlabeled"

    stats = []
    for head, order in ordered.items():
        owner = [None] * len(hosts)
        for pattern, h, _ in defaults:
            if h == head:
                compiled = re.compile(pattern)
                for i, host in enumerate(hosts):
                    if owner[i] is None and compiled.fullmatch(host):
                        owner[i] = "default"
        for row in order:
            for i in hits[row["line"]]:
                if owner[i] is None:
                    owner[i] = row["line"]
        by_row = {row["line"]: {"correct": 0, "compatible": 0, "wrong": 0, "unlabeled": 0,
                                "test_correct": 0, "test_wrong": 0, "wrong_examples": []} for row in order}
        for i, line in enumerate(owner):
            if line in (None, "default"):
                continue
            item = by_row[line]
            if by_line[line][head] == UNKNOWN:
                item["unlabeled"] += 1
                continue
            got = verdict(head, i, by_line[line])
            if got != "unlabeled" and split[i] == "test":
                item["test_" + ("wrong" if got == "wrong" else "correct")] += 1
                continue
            item[got] += 1
            if got == "wrong" and len(item["wrong_examples"]) < 5:
                truth = truths[head][i] if not pd.isna(truths[head][i]) else "/".join(
                    str(truths[g][i]) for g in HEADS if not pd.isna(truths[g][i]))
                item["wrong_examples"].append(f"{hosts[i]}={truth}")
        for row in order:
            item = by_row[row["line"]]
            scored = item["correct"] + item["wrong"]
            precision = item["correct"] / scored if scored else None
            weak = 0 < scored < MIN_SCORED and item["wrong"] > item["correct"] + item["compatible"]
            if row[head] == UNKNOWN:
                status = "stop"   # rule chặn: khớp thì dừng cascade head này, không trả nhãn
            elif (scored >= MIN_SCORED and precision < MIN_PRECISION) or weak:
                status = "rejected"
            else:
                status = "verified" if scored >= MIN_SCORED else "unverified"
            stats.append({"line": row["line"], "group": row["group"], "head": head, "label": row[head],
                          "status": status, "precision": precision, **item,
                          "wrong_examples": " ; ".join(item["wrong_examples"])})
    return pd.DataFrame(stats)


def label_key(label):
    return re.sub(r"[^a-z0-9]", "", normalize_hostname(label).replace("+", "plus"))


def label_hints(rows, frame):
    """Nhãn catalog trùng khóa với nhãn dataset nhưng viết khác (vd 'Samsung A51')."""
    known = {}
    for head in HEADS:
        for label in frame[head].dropna().unique():
            known.setdefault((head, label_key(label)), set()).add(str(label))
    hints = []
    for row in rows:
        for head in HEADS:
            if row[head]:
                same = known.get((head, label_key(row[head])), set()) - {row[head]}
                if same:
                    hints.append(f"dòng {row['line']} {head}={row[head]!r}; dataset viết {sorted(same)}")
    return hints


def build(catalog_path, split_path, default_notebook):
    rows = load_catalog(catalog_path)
    defaults = read_default_rules(default_notebook)
    frame = pd.read_csv(split_path, low_memory=False, keep_default_na=False, na_values=[""])
    frame = frame.loc[frame["hostname"].notna()].reset_index(drop=True)
    frame["hostname"] = frame["hostname"].map(normalize_hostname)
    frame.loc[frame["model"].isin(NOISY_MODEL), ["make", "model"]] = pd.NA
    hits =row_hits(rows, frame["hostname"].tolist())
    excluded, dropped = set(), []
    full = {h: order_rows(rows, h) for h in HEADS}
    for _ in range(5):
        # Bỏ một (dòng, head) có thể đẩy host sang dòng sau, nên chấm lại tới khi ổn định.
        kept = {h: [r for r in full[h] if (r["line"], h) not in excluded] for h in HEADS}
        report = score(rows, kept, defaults, frame, hits)
        rejected = report["status"].eq("rejected")
        if not rejected.any():
            break
        dropped.append(report.loc[rejected])
        excluded |= set(zip(report.loc[rejected, "line"], report.loc[rejected, "head"]))
        # Make/type sai nghĩa là host khớp là thiết bị khác ("nomi-ipc-s21fe" là camera),
        # nên model cùng dòng cũng không tin được dù host đó chưa có nhãn model.
        wrong_device = report.loc[rejected & report["head"].isin(["make", "type"]), "line"]
        dropped.append(report.loc[report["line"].isin(wrong_device) & ~rejected].assign(status="rejected_row"))
        excluded |={(line, h) for line in wrong_device for h in HEADS}
    else:
        raise RuntimeError("Loại rule không hội tụ sau 5 vòng")
    report = pd.concat([report, *dropped], ignore_index=True).sort_values(["head", "line"], kind="stable")
    hierarchy = {}
    for row in rows:
        if row["model"] not in ("", UNKNOWN) and (row["line"], "model") not in excluded:
            entry = {"types": [row["type"]] if row["type"] else []}
            if row["make"]:
                entry["make"] = row["make"]
            hierarchy[row["model"]] = entry
    return {"format": FORMAT,
            "catalog": project_path(catalog_path),
            "catalog_sha256": catalog_digest(catalog_path),
            "evidence": {"split_file": project_path(split_path), "used_splits": ["train", "validation"],
                         "min_scored": MIN_SCORED, "min_precision": MIN_PRECISION},
            "rules": build_rules(kept), "hierarchy": hierarchy,
            "label_hints": label_hints(rows, frame)}, report


ROOT = Path(__file__).resolve().parents[2]


def project_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def catalog_digest(path):
    """SHA-256 không phụ thuộc CRLF/LF (git autocrlf đổi kiểu xuống dòng khi checkout)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_catalog_rules(path):
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    if bundle.get("format") != FORMAT:
        raise ValueError(f"Catalog rule format không hỗ trợ: {bundle.get('format')}")
    for pattern, head, _ in bundle["rules"]:
        assert head in HEADS
        re.compile(pattern)
    return bundle


def main():
    root = ROOT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default=root / "Code/pipeline/hostname_catalog.csv", type=Path)
    parser.add_argument("--split", required=True, type=Path, help="hostname_dataset_split.csv của run 05")
    parser.add_argument("--defaults", default=root / "Code/pipeline/out_model/09_build_sdc_iden.ipynb", type=Path)
    parser.add_argument("--out", default=root / "Code/pipeline/hostname_catalog_rules.json", type=Path)
    parser.add_argument("--report", default=root / "Code/pipeline/hostname_catalog_report.csv", type=Path)
    args = parser.parse_args()
    bundle, report = build(args.catalog, args.split, args.defaults)
    args.out.write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    report.to_csv(args.report, index=False, encoding="utf-8-sig")
    print(report.groupby(["head", "status"]).size().unstack(fill_value=0))
    print(f"{len(bundle['rules'])} rule → {args.out}")
    for hint in bundle["label_hints"]:
        print("Nhãn viết khác dataset:", hint)


if __name__ == "__main__":
    main()
