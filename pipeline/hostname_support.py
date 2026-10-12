"""Hostname cascade shared by notebooks 05/09/12/13.

Caller supplies NFKC-normalized hostnames (normalize_hostname). Character aliases
also handle uppercase and whitespace in the graph. ONNX char tokenization:
https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Tokenizer
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np

HEADS = ("make", "type", "model")
UNKNOWN = "__unknown__"
NULLS = {"", "null", "none", "nan", "<missing>", "<na>", UNKNOWN}
# Packet fingerprints cannot tell an Android tablet from a phone, so a hostname model
# whose hierarchy type is Tablet may refine a packet Smartphone (never the reverse).
TYPE_REFINE = {"Smartphone": ["Tablet"]}


def run_onnx(session, outputs, feeds, batch_size=32):
    """Bound memory of the dense hostname encoder during notebook evaluation."""
    matrix = feeds["input"]
    if len(matrix) <= batch_size:
        return session.run(outputs, feeds)
    parts = []
    for start in range(0, len(matrix), batch_size):
        batch = dict(feeds)
        batch["input"] = matrix[start:start + batch_size]
        got = session.run(outputs, batch)
        if not parts:
            parts = [[] for _ in got]
        for values, output in zip(parts, got):
            values.append(output)
    return [np.concatenate(values, axis=0) for values in parts]


def normalize_hostname(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return "" if text in NULLS else re.sub(r"\s+", " ", text)


def load_hostname(run_dir, required=False):
    path = Path(run_dir) / "train" / "hostname_cascade.joblib"
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Thiếu {path}. Chạy phần C của 05 với HN_SAVE=True trên cùng run.")
        return None
    bundle = joblib.load(path)
    if bundle.get("format") != "sdc-hostname-cascade-v1":
        raise ValueError(f"Hostname format không hỗ trợ: {bundle.get('format')}")
    # Averaged-SGD training buffers are not used by predict_proba/export.
    for item in bundle["experts"].values():
        for attr in ("_standard_coef", "_standard_intercept", "_average_coef", "_average_intercept"):
            if hasattr(item["model"], attr):
                delattr(item["model"], attr)
    return bundle


def expanded_labels(base, bundle):
    if bundle is None:
        return base
    return {h: sorted(set(base[h]) | set(map(str, bundle["experts"][h]["model"].classes_))
                      | {str(y) for _, head, y in bundle["rules"] if head == h}) for h in HEADS}


def make_evidence_patterns(bundle):
    """Require an actual manufacturer/family token before accepting expert make."""
    aliases = {"Apple": ["iphone", "ipad", "macbook", "imac"],
               "Samsung": ["galaxy"], "Xiaomi": ["redmi", "poco"]}
    patterns = {}
    for label in bundle["experts"]["make"]["model"].classes_:
        name = str(label)
        terms = [name, *aliases.get(name, [])]
        variants = []
        for term in terms:
            letters = re.sub(r"[^a-z0-9]", "", normalize_hostname(term))
            if len(letters) >= 3:
                variants.append(r"[-_. ]*".join(re.escape(c) for c in letters))
            elif letters:
                variants.append(re.escape(letters))
        patterns[name] = r"(?i)(?:.*[^a-z0-9])?(?:" + "|".join(variants) + r")(?:[^a-z0-9].*)?"
    return patterns


def candidate_make_supported(hostnames, candidates, patterns):
    return np.array([bool(patterns.get(str(label))) and
                     re.fullmatch(patterns[str(label)], normalize_hostname(host)) is not None
                     for host, label in zip(hostnames, candidates)], dtype=bool)


def rule_hits(hostnames, rules, head):
    hosts = [normalize_hostname(h) for h in hostnames]
    return np.array([any(h == head and 0 < len(s) <= 253 and re.fullmatch(p, s) is not None
                         for p, h, _ in rules) for s in hosts], dtype=bool)


def hostname_reference(hostnames, bundle, return_details=False, policy=None):
    hosts = [normalize_hostname(h) for h in hostnames]
    X = bundle["vectorizer"].transform(hosts)
    usable = np.asarray(X.getnnz(axis=1)).ravel() > 0
    usable &= np.array([0 < len(h) <= 253 for h in hosts])
    output = np.full((len(hosts), 3), UNKNOWN, dtype=object)
    confidence = np.zeros((len(hosts), 3), dtype=float)
    source = np.full((len(hosts), 3), "abstain", dtype=object)
    # A supplied v1 contract preserves the old behavior for old-model parity.
    policy = {"expert_make_evidence": make_evidence_patterns(bundle)} if policy is None else policy
    for j, head in enumerate(HEADS):
        item = bundle["experts"][head]
        hit = np.zeros(len(hosts), dtype=bool)
        for pattern, h, label in bundle["rules"]:
            if h != head:
                continue
            take = ~hit & np.array([0 < len(s) <= 253 and re.fullmatch(pattern, s) is not None for s in hosts])
            output[take, j] = label
            confidence[take, j] = 1.0
            source[take, j] = "A_rule"
            hit |= take
        positions = np.flatnonzero(~hit & usable)
        if len(positions) and item["threshold"] <= 1:
            p = item["model"].predict_proba(X[positions])
            idx = p.argmax(axis=1)
            accepted = p[np.arange(len(p)), idx] >= item["threshold"]
            if head == "make" and policy.get("expert_make_evidence"):
                accepted &= candidate_make_supported([hosts[i] for i in positions],
                    item["model"].classes_[idx], policy["expert_make_evidence"])
            output[positions[accepted], j] = item["model"].classes_[idx[accepted]]
            confidence[positions[accepted], j] = p[np.arange(len(p)), idx][accepted]
            source[positions[accepted], j] = "B_expert"
    return (output, confidence, source) if return_details else output


def make_spec(bundle):
    vec = bundle["vectorizer"]
    if (vec.analyzer, vec.ngram_range, vec.norm, vec.sublinear_tf) != ("char_wb", (3, 5), "l2", True):
        raise ValueError("Exporter yêu cầu char_wb (3,5), l2, sublinear_tf=True")
    vocab = list(map(str, vec.get_feature_names_out()))
    alphabet = sorted(set("".join(vocab)) | {" "})
    ids = {c: i + 1 for i, c in enumerate(alphabet)}
    # An unseen character must not accidentally match a shorter n-gram.
    base = len(ids) + 1
    if 2 * base ** 5 >= np.iinfo(np.int64).max:
        raise ValueError("Alphabet quá lớn cho khóa n-gram int64")
    aliases = dict(ids)
    for c in alphabet:
        for raw in (c.upper(), c.title()):
            if len(raw) == 1:
                aliases[raw] = ids[c]
    for code in range(0x110000):
        raw = chr(code)
        norm = unicodedata.normalize("NFKC", raw).lower()
        if raw.isspace():
            aliases[raw] = ids[" "]
        elif len(norm) == 1 and norm in ids:
            aliases[raw] = ids[norm]
    pool, indexes, offsets, gram_keys = [], [], [], []
    for n in range(1, 6):
        offsets.append(len(pool))
        for index, gram in enumerate(vocab):
            if len(gram) != n:
                continue
            codes = [ids[c] for c in gram]
            pool.extend(codes)
            indexes.append(index)
            key = base ** n + sum(c * base ** (n - k - 1) for k, c in enumerate(codes))
            gram_keys.append((key, index))
    if len(indexes) != len(vocab):
        raise ValueError("Vocabulary có n-gram ngoài khoảng 3..5")
    spec = {"vocab": vocab, "idf": vec.idf_.astype(np.float32), "aliases": aliases,
            "base": base, "pool": pool, "indexes": indexes, "offsets": offsets,
            "gram_keys": dict(gram_keys), "rules": bundle["rules"], "experts": {}}
    for h in HEADS:
        item = bundle["experts"][h]
        model = item["model"]
        spec["experts"][h] = {"labels": list(map(str, model.classes_)),
                              "weight": model.coef_.T.astype(np.float32),
                              "bias": model.intercept_.astype(np.float32),
                              "threshold": float(item["threshold"])}
    return spec


def implied_labels(labels, hierarchy):
    """Make/type implied by each model label; type only when the model allows exactly one."""
    implied = {"make": [], "type": []}
    for name in labels["model"]:
        rule = hierarchy.get(name, {}) if name != UNKNOWN else {}
        make = rule.get("make")
        types = [t for t in rule.get("types", []) if t in labels["type"]]
        implied["make"].append(make if make in labels["make"] and make != UNKNOWN else UNKNOWN)
        implied["type"].append(types[0] if len(types) == 1 and types[0] != UNKNOWN else UNKNOWN)
    return implied


def hostname_contract(run_dir, bundle, base_labels, resolved, hierarchy):
    labels = expanded_labels(base_labels, bundle)
    merged = dict(hierarchy)
    for model, rule in bundle.get("hierarchy", {}).items():
        merged.setdefault(model, rule)
    masks = {}
    for h in ("make", "type"):
        mask = np.ones((len(labels["model"]), len(labels[h])), dtype=np.float32)
        for i, name in enumerate(labels["model"]):
            rule = merged.get(name, {})
            allowed = [rule.get("make")] if h == "make" else rule.get("types", [])
            allowed = [v for v in allowed if v in labels[h]]
            if allowed and name != UNKNOWN:
                mask[i] = 0
                for v in allowed + [UNKNOWN]:
                    mask[i, labels[h].index(v)] = 1
        masks[h] = mask
    # A packet type that a model may refine (Smartphone -> Tablet) is not a conflict.
    guard = masks["type"].copy()
    for coarse, fine in TYPE_REFINE.items():
        if coarse in labels["type"]:
            cols = [labels["type"].index(t) for t in fine if t in labels["type"]]
            if cols:
                guard[:, labels["type"].index(coarse)] = np.maximum(
                    guard[:, labels["type"].index(coarse)], masks["type"][:, cols].max(axis=1))
    masks["type_guard"] = guard
    path = Path(run_dir) / "train/hostname_cascade.joblib"
    return {"format": "sdc-hostname-graph-v3", "file": "train/hostname_cascade.joblib",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "base_labels": base_labels, "labels": labels,
            "rules": [list(r) for r in bundle["rules"]],
            "thresholds": {h: float(bundle["experts"][h]["threshold"]) for h in HEADS},
            "priority": "mDNS/OUI > hostname rule A > answered packet head > expert B; "
                        "hostname model implies make/type; hierarchy; OS cascade",
            "preserve_packet_answers": True, "guard_model_type": True,
            "guard_model_make": True, "model_implies": True,
            "type_refine": {k: [t for t in v if t in labels["type"]]
                            for k, v in TYPE_REFINE.items() if k in labels["type"]},
            "implied": implied_labels(labels, merged),
            "expert_make_evidence": make_evidence_patterns(bundle),
            "input_normalization": "dhcp_hostname: NFKC + strip + lowercase + collapse whitespace",
            "experimental": True, "gate": bundle.get("provenance", {}).get("test_gate", {}),
            "strong_mdns": list(resolved["mdns_model"]), "strong_oui": list(resolved["oui"]),
            "hierarchy": merged, "masks": {h: m.tolist() for h, m in masks.items()}}


def merge_reference(base, host, matrix, cols, contract):
    result = np.array(base, dtype=object, copy=True)
    host = np.array(host, dtype=object, copy=True)
    hits = {h: rule_hits(matrix[:, cols.index("dhcp_hostname")], contract.get("rules", []), h)
            for h in HEADS}
    labels = contract["labels"]
    masks = contract["masks"]
    index = lambda values, h: np.array([labels[h].index(v) for v in values])
    if contract.get("guard_model_type"):
        type_mask = np.asarray(masks.get("type_guard", masks["type"]))
        allowed = type_mask[index(host[:, 2], "model"), index(result[:, 1], "type")] > 0
        conflict = ~allowed & (result[:, 1] != UNKNOWN) & ~hits["model"]
        host[conflict, 2] = UNKNOWN
    if contract.get("guard_model_make"):
        # An expert model must agree with every make already known (packet or hostname).
        model_index = index(host[:, 2], "model")
        allowed = np.ones(len(host), dtype=bool)
        for values in (result[:, 0], host[:, 0]):
            allowed &= np.asarray(masks["make"])[model_index, index(values, "make")] > 0
        host[~allowed & ~hits["model"], 2] = UNKNOWN
    strong_mdns = np.isin(matrix[:, cols.index("mdns_model")], contract["strong_mdns"])
    strong_oui = np.isin(matrix[:, cols.index("mac_oui")], contract["strong_oui"])
    taken = {}
    for j, h in enumerate(HEADS):
        strong = strong_mdns | (strong_oui if h == "make" else False)
        if contract.get("preserve_packet_answers"):
            strong |= ~hits[h]
        keep_base = strong & (result[:, j] != UNKNOWN)
        take = (host[:, j] != UNKNOWN) & ~keep_base
        result[take, j] = host[take, j]
        taken[h] = take
    if contract.get("model_implies"):
        # Only a model taken from the hostname fills in; packet-only rows stay unchanged.
        model_index = index(result[:, 2], "model")
        make = np.asarray(contract["implied"]["make"], dtype=object)[model_index]
        fill = taken["model"] & (result[:, 0] == UNKNOWN) & (make != UNKNOWN)
        result[fill, 0] = make[fill]
        kind = np.asarray(contract["implied"]["type"], dtype=object)[model_index]
        refine = contract.get("type_refine", {})
        refinable = np.array([k in refine.get(t, []) for t, k in zip(result[:, 1], kind)], dtype=bool)
        fill = taken["model"] & (kind != UNKNOWN) & ((result[:, 1] == UNKNOWN) | refinable)
        result[fill, 1] = kind[fill]
    driver = np.array([labels["model"].index(v) for v in result[:, 2]])
    for j, h in enumerate(("make", "type")):
        idx = np.array([labels[h].index(v) for v in result[:, j]])
        ok = np.asarray(contract["masks"][h])[driver, idx] > 0
        result[~ok, j] = UNKNOWN
    return result


def attach_hostname(model, bundle, cols, contract, GB):
    """Append hostname branch to existing ONNX without changing packet classifiers."""
    import onnx
    from onnx import TensorProto as TP, helper
    spec = make_spec(bundle)
    gb = GB("sdc/hn")
    # Preserve the legacy output for probes and TensorFlow reconstruction.
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name == "output":
                node.output[i] = "sdc/packet_output"
    def const(value, stem, dtype=None):
        return gb.const(np.asarray(value), stem, dtype)
    def lookup(x, book, stem, default=-1):
        return gb.add("LabelEncoder", [x], stem, domain="ai.onnx.ml",
                      keys_strings=list(book) or ["\x00empty"],
                      values_int64s=list(book.values()) or [default], default_int64=default)
    axis = const(np.int64([1]), "axis")
    hostname = gb.add("Gather", ["input", const(np.int64([cols.index("dhcp_hostname")]), "hc")], "host", axis=1)
    padded = gb.add("StringConcat", [const([[" "]], "space", TP.STRING), hostname], "padded_left")
    padded = gb.add("StringConcat", [padded, const([[" "]], "space_right", TP.STRING)], "padded")
    flat = gb.add("Reshape", [padded, const(np.int64([-1]), "flatshape")], "flat")
    chars = gb.add("Tokenizer", [flat], "chars", domain="com.microsoft", separators=[""],
                   mark=0, mincharnum=1, pad_value="")
    char_ids = lookup(chars, spec["aliases"], "char_ids", default=0)
    counts = gb.add("TfIdfVectorizer", [char_ids], "counts", mode="TF", min_gram_length=3,
                    max_gram_length=5, max_skip_count=0, ngram_counts=spec["offsets"],
                    ngram_indexes=spec["indexes"], pool_int64s=spec["pool"],
                    weights=[1.0] * len(spec["vocab"]))
    positive = gb.add("Greater", [counts, const(np.float32(0), "zero")], "positive")
    tf_counts = gb.add("Where", [positive,
        gb.add("Add", [gb.add("Log", [gb.add("Max", [counts, const(np.float32(1), "one")], "safe")], "log"),
                       const(np.float32(1), "logone")], "sublinear"), const(np.float32(0), "z")], "tf")
    weighted = gb.add("Mul", [tf_counts, const(spec["idf"], "idf")], "weighted")
    norm = gb.add("Sqrt", [gb.add("ReduceSum", [gb.add("Mul", [weighted, weighted], "square"), axis], "sum", keepdims=1)], "norm")
    X = gb.add("Div", [weighted, gb.add("Max", [norm, const(np.float32(1e-12), "eps")], "safe_norm")], "X")
    usable = gb.add("Greater", [norm, const(np.float32(0), "zn")], "usable")
    bad = gb.add("RegexFullMatch", [hostname], "bad", pattern=r"(?i)[\s\p{Z}]*(?:null|none|nan|<missing>|<na>|__unknown__)?[\s\p{Z}]*")
    valid = gb.add("And", [usable, gb.add("Not", [bad], "notbad")], "valid")
    lengths = gb.add("ReduceSum", [gb.add("Cast", [gb.add("Not", [gb.add("Equal", [chars, const([[""]], "pad", TP.STRING)], "ispad")], "notpad")], "charsf", to=TP.FLOAT), axis], "length", keepdims=1)
    short = gb.add("LessOrEqual", [lengths, const(np.float32(255), "limit")], "short")
    valid = gb.add("And", [valid, short], "length_valid")
    rule_valid = gb.add("And", [gb.add("Not", [bad], "rule_notbad"), short], "rule_valid")
    base_cols, host_cols, rule_masks = [], [], []
    for j, h in enumerate(HEADS):
        base_cols.append(gb.add("Gather", ["sdc/packet_output", const(np.int64([j]), "base_" + h)], "basecol_" + h, axis=1))
        item = spec["experts"][h]
        score = gb.add("Add", [gb.add("MatMul", [X, const(item["weight"], "weight_" + h)], "score_" + h), const(item["bias"], "bias_" + h)], "linear_" + h)
        sigmoid = gb.add("Sigmoid", [score], "sigmoid_" + h)
        if len(item["labels"]) == 2:
            p = gb.add("Concat", [gb.add("Sub", [const(np.float32(1), "p1_" + h), sigmoid], "inverse_" + h), sigmoid], "p_" + h, axis=1)
        else:
            total = gb.add("ReduceSum", [sigmoid, axis], "psum_" + h, keepdims=1)
            p = gb.add("Div", [sigmoid, total], "p_" + h)
        index = gb.add("ArgMax", [p], "idx_" + h, axis=1, keepdims=1)
        conf = gb.add("ReduceMax", [p, axis], "conf_" + h, keepdims=1)
        label = gb.add("Gather", [const(np.asarray(item["labels"], dtype=object), "labels_" + h, TP.STRING), index], "candidate_" + h, axis=0)
        accepted = gb.add("And", [valid, gb.add("GreaterOrEqual", [conf, const(np.float32(item["threshold"]), "threshold_" + h)], "conf_ok_" + h)], "accepted_" + h)
        if h == "make" and contract.get("expert_make_evidence"):
            evidence = const(np.bool_(False), "no_make_evidence")
            for make, pattern in contract["expert_make_evidence"].items():
                match = gb.add("RegexFullMatch", [hostname], "make_evidence", pattern=pattern)
                same = gb.add("Equal", [label, const([[make]], "make_evidence_label", TP.STRING)], "same_make")
                evidence = gb.add("Or", [evidence, gb.add("And", [match, same], "make_supported")], "make_any")
            accepted = gb.add("And", [accepted, evidence], "make_accepted")
        answer = gb.add("Where", [accepted, label, const([[UNKNOWN]], "unknown_" + h, TP.STRING)], "expert_" + h)
        rule_mask = const(np.bool_(False), "no_rule_" + h)
        for pattern, head, rule_label in reversed(spec["rules"]):
            if head == h:
                match = gb.add("RegexFullMatch", [hostname], "rule_" + h, pattern=pattern)
                match = gb.add("And", [match, rule_valid], "valid_rule_" + h)
                rule_mask = gb.add("Or", [rule_mask, match], "any_rule_" + h)
                answer = gb.add("Where", [match, const([[rule_label]], "rule_label_" + h, TP.STRING), answer], "cascade_" + h)
        host_cols.append(answer)
        rule_masks.append(rule_mask)
    def member(x, keys, stem):
        hit = lookup(x, {k: 1 for k in keys}, stem)
        return gb.add("Equal", [hit, const(np.int64(1), stem + "_one")], stem + "_hit")
    mdns = gb.add("Gather", ["input", const(np.int64([cols.index("mdns_model")]), "mdns_col")], "mdns", axis=1)
    oui = gb.add("Gather", ["input", const(np.int64([cols.index("mac_oui")]), "oui_col")], "oui", axis=1)
    mdns_hit = member(mdns, contract["strong_mdns"], "strong_mdns")
    oui_hit = member(oui, contract["strong_oui"], "strong_oui")
    merged, taken = [], []
    unknown = const([[UNKNOWN]], "unknown", TP.STRING)
    def model_index(x, stem):
        idx = lookup(x, {v: i for i, v in enumerate(contract["labels"]["model"])}, stem, default=0)
        return gb.add("Reshape", [idx, const(np.int64([-1]), stem + "_shape")], stem + "_flat")
    def mask_allowed(mask, rows, values, head, stem):
        row = gb.add("Gather", [const(np.float32(mask), stem + "_mask"), rows], stem + "_row", axis=0)
        idx = lookup(values, {v: i for i, v in enumerate(contract["labels"][head])}, stem + "_idx", default=0)
        allowed = gb.add("GatherElements", [row, idx], stem + "_allowed", axis=1)
        return gb.add("Greater", [allowed, const(np.float32(0), stem + "_zero")], stem + "_ok")
    model_no_rule = gb.add("Not", [rule_masks[2]], "model_no_rule")
    if contract.get("guard_model_type"):
        masks = contract["masks"]
        allowed = mask_allowed(masks.get("type_guard", masks["type"]), model_index(host_cols[2], "model_candidate_idx"),
                               base_cols[1], "type", "candidate_type")
        known_type = gb.add("Not", [gb.add("Equal", [base_cols[1], unknown], "packet_type_unknown")], "packet_type_known")
        conflict = gb.add("And", [gb.add("Not", [allowed], "candidate_type_conflict"), known_type], "known_type_conflict")
        conflict = gb.add("And", [conflict, model_no_rule], "expert_model_conflict")
        host_cols[2] = gb.add("Where", [conflict, unknown, host_cols[2]], "safe_expert_model")
    if contract.get("guard_model_make"):
        rows = model_index(host_cols[2], "model_make_idx")
        allowed = gb.add("And", [mask_allowed(contract["masks"]["make"], rows, base_cols[0], "make", "packet_make"),
                                 mask_allowed(contract["masks"]["make"], rows, host_cols[0], "make", "host_make")], "model_make_ok")
        conflict = gb.add("And", [gb.add("Not", [allowed], "model_make_conflict"), model_no_rule], "expert_make_conflict")
        host_cols[2] = gb.add("Where", [conflict, unknown, host_cols[2]], "make_safe_expert_model")
    for j, h in enumerate(HEADS):
        strong = gb.add("Or", [mdns_hit, oui_hit], "strong_make") if h == "make" else mdns_hit
        if contract.get("preserve_packet_answers"):
            strong = gb.add("Or", [strong, gb.add("Not", [rule_masks[j]], "no_rule_" + h)], "preserve_packet_" + h)
        keep = gb.add("And", [strong, gb.add("Not", [gb.add("Equal", [base_cols[j], unknown], "base_unknown_" + h)], "base_known_" + h)], "keep_" + h)
        take = gb.add("And", [gb.add("Not", [gb.add("Equal", [host_cols[j], unknown], "host_unknown_" + h)], "host_known_" + h), gb.add("Not", [keep], "notkeep_" + h)], "take_" + h)
        merged.append(gb.add("Where", [take, host_cols[j], base_cols[j]], "merged_" + h))
        taken.append(take)
    if contract.get("model_implies"):
        # Gather [M] by [N,1] indexes keeps the [N,1] column shape.
        idx = lookup(merged[2], {v: i for i, v in enumerate(contract["labels"]["model"])}, "implied_idx", default=0)
        make = gb.add("Gather", [const(np.asarray(contract["implied"]["make"], dtype=object), "implied_make_table", TP.STRING), idx], "implied_make", axis=0)
        kind = gb.add("Gather", [const(np.asarray(contract["implied"]["type"], dtype=object), "implied_type_table", TP.STRING), idx], "implied_type", axis=0)
        make_known = gb.add("Not", [gb.add("Equal", [make, unknown], "implied_make_unknown")], "implied_make_known")
        fill = gb.add("And", [taken[2], gb.add("And", [gb.add("Equal", [merged[0], unknown], "fill_make_unknown"), make_known], "fill_make_ok")], "fill_make")
        merged[0] = gb.add("Where", [fill, make, merged[0]], "implied_merged_make")
        refinable = gb.add("Equal", [merged[1], unknown], "fill_type_unknown")
        for coarse, fine in contract.get("type_refine", {}).items():
            for target in fine:
                pair = gb.add("And", [gb.add("Equal", [merged[1], const([[coarse]], "refine_from", TP.STRING)], "refine_from_hit"),
                                      gb.add("Equal", [kind, const([[target]], "refine_to", TP.STRING)], "refine_to_hit")], "refine_pair")
                refinable = gb.add("Or", [refinable, pair], "refinable")
        kind_known = gb.add("Not", [gb.add("Equal", [kind, unknown], "implied_type_unknown")], "implied_type_known")
        fill = gb.add("And", [taken[2], gb.add("And", [kind_known, refinable], "fill_type_ok")], "fill_type")
        merged[1] = gb.add("Where", [fill, kind, merged[1]], "implied_merged_type")
    driver = lookup(merged[2], {v: i for i, v in enumerate(contract["labels"]["model"])}, "driver", default=0)
    driver = gb.add("Reshape", [driver, const(np.int64([-1]), "drivershape")], "driverflat")
    for j, h in enumerate(("make", "type")):
        row = gb.add("Gather", [const(np.float32(contract["masks"][h]), "mask_" + h), driver], "row_" + h, axis=0)
        idx = lookup(merged[j], {v: i for i, v in enumerate(contract["labels"][h])}, "merged_idx_" + h, default=0)
        allowed = gb.add("GatherElements", [row, idx], "allowed_" + h, axis=1)
        merged[j] = gb.add("Where", [gb.add("Greater", [allowed, const(np.float32(0), "maskzero_" + h)], "ok_" + h), merged[j], unknown], "checked_" + h)
    gb.nodes.append(helper.make_node("Concat", merged, ["sdc/hn/branch_output"], axis=1,
                                     name="sdc/hn/branch_output_node"))
    # Avoid allocating a dense [N,60000] vector for a batch with no hostname.
    guard = GB("sdc/hn_guard")
    guard_host = guard.add("Gather", ["input", guard.const(np.int64([cols.index("dhcp_hostname")]), "col")], "host", axis=1)
    invalid = guard.add("RegexFullMatch", [guard_host], "invalid",
                        pattern=r"(?i)[\s\p{Z}]*(?:null|none|nan|<missing>|<na>|__unknown__)?[\s\p{Z}]*")
    present = guard.add("Cast", [guard.add("Not", [invalid], "present")], "present_int", to=TP.INT64)
    total = guard.add("ReduceSum", [present], "total", keepdims=0)
    any_host = guard.add("Greater", [total, guard.const(np.int64(0), "zero")], "any")
    then_branch = helper.make_graph(gb.nodes, "hostname_present", [],
        [helper.make_tensor_value_info("sdc/hn/branch_output", TP.STRING, [None, 3])])
    else_branch = helper.make_graph(
        [helper.make_node("Identity", ["sdc/packet_output"], ["sdc/hn/pass_output"])],
        "hostname_absent", [], [helper.make_tensor_value_info("sdc/hn/pass_output", TP.STRING, [None, 3])])
    model.graph.node.extend(guard.nodes)
    model.graph.node.append(helper.make_node("If", [any_host], ["sdc/hn/result"],
                           then_branch=then_branch, else_branch=else_branch, name="sdc/hn/if"))
    parts = ["sdc/hn/output_" + h for h in HEADS]
    model.graph.node.append(helper.make_node("Split", ["sdc/hn/result"], parts, axis=1, num_outputs=3))
    model.graph.node.append(helper.make_node("Concat", parts, ["output"], axis=1, name="sdc/hn/output"))
    model.graph.initializer.extend(gb.inits)
    model.graph.initializer.extend(guard.inits)
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)
    return model


def tf_hostname_module(tf, bundle, contract):
    """Equivalent char_wb + SGD graph, with Flex ops for Unicode/table lookup."""
    spec = make_spec(bundle)
    class HostnameModule(tf.Module):
        def __init__(self):
            super().__init__()
            self.characters = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
                tf.constant(list(spec["aliases"])), tf.constant(list(spec["aliases"].values()), tf.int64)), 0)
            self.grams = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
                tf.constant(list(spec["gram_keys"]), tf.int64), tf.constant(list(spec["gram_keys"].values()), tf.int64)), -1)

        def features(self, hosts):
            chars = tf.strings.unicode_split(tf.strings.join([tf.fill(tf.shape(hosts), " "), hosts,
                                                           tf.fill(tf.shape(hosts), " ")]), "UTF-8").to_tensor("")
            ids = self.characters.lookup(chars)
            width = len(spec["vocab"])
            n, length = tf.shape(ids)[0], tf.shape(ids)[1]
            # Always enough columns for 3..5-grams, even for a batch of empty hostnames.
            ids = tf.pad(ids, [[0, 0], [0, tf.maximum(5 - length, 0)]])
            length = tf.shape(ids)[1]
            all_indexes = []
            for order in (3, 4, 5):
                count = length - order + 1
                key = tf.fill([n, count], tf.cast(spec["base"] ** order, tf.int64))
                for k in range(order):
                    key += ids[:, k:k + count] * tf.cast(spec["base"] ** (order - k - 1), tf.int64)
                all_indexes.append(self.grams.lookup(key))
            index = tf.concat(all_indexes, axis=1)
            values = tf.cast(index >= 0, tf.float32)
            segment = tf.maximum(index, 0) + tf.expand_dims(tf.range(n, dtype=tf.int64) * width, 1)
            counts = tf.reshape(tf.math.unsorted_segment_sum(tf.reshape(values, [-1]),
                                tf.reshape(segment, [-1]), n * width), [n, width])
            weighted = tf.where(counts > 0, 1 + tf.math.log(tf.maximum(counts, 1)), 0) * tf.constant(spec["idf"])
            norm = tf.sqrt(tf.reduce_sum(weighted * weighted, axis=1, keepdims=True))
            return tf.math.divide_no_nan(weighted, norm), tf.squeeze(norm > 0, 1)

        def predict(self, hosts):
            X, usable = self.features(hosts)
            bad = tf.strings.regex_full_match(hosts, r"(?i)[\s\p{Z}]*(?:null|none|nan|<missing>|<na>|__unknown__)?[\s\p{Z}]*")
            rule_valid = ~bad & (tf.strings.length(hosts, unit="UTF8_CHAR") <= 253)
            valid = usable & rule_valid
            outputs = []
            for h in HEADS:
                item = spec["experts"][h]
                p = tf.math.sigmoid(tf.matmul(X, tf.constant(item["weight"])) + tf.constant(item["bias"]))
                p = tf.concat([1 - p, p], axis=1) if len(item["labels"]) == 2 else tf.math.divide_no_nan(p, tf.reduce_sum(p, axis=1, keepdims=True))
                idx = tf.argmax(p, axis=1, output_type=tf.int32)
                candidate = tf.gather(tf.constant(item["labels"]), idx)
                accepted = valid & (tf.reduce_max(p, axis=1) >= item["threshold"])
                if h == "make" and contract.get("expert_make_evidence"):
                    evidence = tf.zeros_like(valid)
                    for make, pattern in contract["expert_make_evidence"].items():
                        evidence |= tf.strings.regex_full_match(hosts, pattern) & tf.equal(candidate, make)
                    accepted &= evidence
                answer = tf.where(accepted, candidate, UNKNOWN)
                for pattern, head, label in reversed(spec["rules"]):
                    if head == h:
                        answer = tf.where(tf.strings.regex_full_match(hosts, pattern) & rule_valid, label, answer)
                outputs.append(answer)
            return tf.stack(outputs, axis=1)

        def merge(self, base_indexes, hosts, col, labels):
            base = tf.stack([tf.gather(tf.constant(labels[h]), base_indexes[:, j]) for j, h in enumerate(HEADS)], axis=1)
            host = self.predict(hosts)
            def index(values, table):
                return tf.argmax(tf.cast(tf.equal(values[:, None], tf.constant(table)), tf.int32), axis=1, output_type=tf.int32)
            hits = {}
            for h in HEADS:
                hit = tf.zeros_like(hosts, dtype=tf.bool)
                for pattern, head, _ in spec["rules"]:
                    if head == h:
                        hit |= tf.strings.regex_full_match(hosts, pattern)
                hits[h] = hit
            masks = contract["masks"]
            def allowed(mask, rows, values, head):
                cells = tf.stack([rows, index(values, contract["labels"][head])], axis=1)
                return tf.gather_nd(tf.constant(np.float32(mask)), cells) > 0
            if contract.get("guard_model_type"):
                candidate_idx = index(host[:, 2], contract["labels"]["model"])
                ok = allowed(masks.get("type_guard", masks["type"]), candidate_idx, base[:, 1], "type")
                conflict = ~ok & tf.not_equal(base[:, 1], UNKNOWN) & ~hits["model"]
                host = tf.stack([host[:, 0], host[:, 1], tf.where(conflict, UNKNOWN, host[:, 2])], axis=1)
            if contract.get("guard_model_make"):
                candidate_idx = index(host[:, 2], contract["labels"]["model"])
                ok = (allowed(masks["make"], candidate_idx, base[:, 0], "make")
                      & allowed(masks["make"], candidate_idx, host[:, 0], "make"))
                host = tf.stack([host[:, 0], host[:, 1], tf.where(~ok & ~hits["model"], UNKNOWN, host[:, 2])], axis=1)
            mdns = tf.reduce_any(tf.equal(col["mdns_model"][:, None], tf.constant(contract["strong_mdns"], dtype=tf.string)), axis=1)
            oui = tf.reduce_any(tf.equal(col["mac_oui"][:, None], tf.constant(contract["strong_oui"], dtype=tf.string)), axis=1)
            merged, taken = [], []
            for j, h in enumerate(HEADS):
                strong = mdns | (oui if h == "make" else False)
                if contract.get("preserve_packet_answers"):
                    strong |= ~hits[h]
                keep = strong & tf.not_equal(base[:, j], UNKNOWN)
                take = tf.not_equal(host[:, j], UNKNOWN) & ~keep
                merged.append(tf.where(take, host[:, j], base[:, j]))
                taken.append(take)
            if contract.get("model_implies"):
                implied_idx = index(merged[2], contract["labels"]["model"])
                make = tf.gather(tf.constant(contract["implied"]["make"]), implied_idx)
                kind = tf.gather(tf.constant(contract["implied"]["type"]), implied_idx)
                fill = taken[2] & tf.equal(merged[0], UNKNOWN) & tf.not_equal(make, UNKNOWN)
                merged[0] = tf.where(fill, make, merged[0])
                refinable = tf.equal(merged[1], UNKNOWN)
                for coarse, fine in contract.get("type_refine", {}).items():
                    for target in fine:
                        refinable |= tf.equal(merged[1], coarse) & tf.equal(kind, target)
                fill = taken[2] & tf.not_equal(kind, UNKNOWN) & refinable
                merged[1] = tf.where(fill, kind, merged[1])
            driver = index(merged[2], contract["labels"]["model"])
            for j, h in enumerate(("make", "type")):
                ix = index(merged[j], contract["labels"][h])
                mask = tf.constant(np.float32(contract["masks"][h]))
                allowed = tf.gather_nd(mask, tf.stack([driver, ix], axis=1)) > 0
                merged[j] = tf.where(allowed, merged[j], UNKNOWN)
            return tf.stack([index(merged[j], contract["labels"][h]) for j, h in enumerate(HEADS)], axis=1)
    return HostnameModule()


def assert_hostname_source(run_dir, contract):
    if not contract:
        return
    path = Path(run_dir) / contract["file"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != contract["sha256"]:
        raise ValueError(f"Hostname bundle khác bản 09 đã dùng: {path}; chạy lại 09 → 12 → 13")
