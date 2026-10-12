import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from IPython.core.inputtransformer2 import TransformerManager
from onnx import TensorProto as TP, helper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from hostname_support import (HEADS, UNKNOWN, normalize_hostname, hostname_reference,
    hostname_contract, attach_hostname, merge_reference, assert_hostname_source,
    make_evidence_patterns, candidate_make_supported)


class GB:
    def __init__(self, prefix):
        self.prefix, self.nodes, self.inits, self.n = prefix, [], [], 0
    def name(self, stem):
        self.n += 1
        return f"{self.prefix}/{stem}_{self.n}"
    def const(self, value, stem, dtype=None):
        name = self.name(stem)
        value = np.asarray(value)
        if dtype == TP.STRING:
            self.inits.append(helper.make_tensor(name, TP.STRING, value.shape,
                                                [str(v).encode() for v in value.ravel()]))
        else:
            self.inits.append(onnx.numpy_helper.from_array(value, name))
        return name
    def add(self, op, inputs, stem, **attrs):
        name = self.name(stem)
        domain = attrs.pop("domain", "")
        self.nodes.append(helper.make_node(op, inputs, [name], domain=domain, **attrs))
        return name


class HostnameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hosts = ["iphone", "apple-phone", "galaxy-a51", "samsung-phone", "huyền-iphone",
                 "a a", "galaxy a51", "iphone pro"] * 10
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                              dtype=np.float32)
        X = vec.fit_transform(hosts)
        cls.bundle = {"format": "sdc-hostname-cascade-v1", "vectorizer": vec,
                      "rules": [(r"(?i)iphone", "make", "Apple"),
                                (r"(?i)iphone", "model", "iPhone")], "experts": {}}
        for h in HEADS:
            labels = ["Apple", "Apple", "Samsung", "Samsung", "Apple", "Samsung", "Samsung", "Apple"]
            if h == "type":
                labels = ["Smartphone"] * 5 + ["Tablet", "Smartphone", "Tablet"]
            if h == "model":
                labels = ["iPhone", "iPhone", "Galaxy", "Galaxy", "iPhone", "Other", "Galaxy", "Other"]
            m = SGDClassifier(loss="log_loss", random_state=42, max_iter=1000, tol=1e-4).fit(X, labels * 10)
            cls.bundle["experts"][h] = {"model": m, "threshold": 0.55}
        cls.base_labels = {h: [UNKNOWN] for h in HEADS}
        cls.cols = ["mac_oui", "dhcp_hostname", "mdns_model"]

    def graph(self, bundle, base_values=None, hierarchy=None):
        base_values = base_values or [UNKNOWN] * 3
        base_labels = {h: sorted(set(self.base_labels[h]) | {base_values[j]})
                       for j, h in enumerate(HEADS)}
        shape = helper.make_node("Shape", ["input"], ["shape"])
        n = helper.make_node("Gather", ["shape", "zero"], ["n"], axis=0)
        dims = helper.make_node("Concat", ["n", "three"], ["dims"], axis=0)
        expand = helper.make_node("Expand", ["unknown", "dims"], ["output"])
        model = helper.make_model(helper.make_graph([shape, n, dims, expand], "base",
            [helper.make_tensor_value_info("input", TP.STRING, [None, 3])],
            [helper.make_tensor_value_info("output", TP.STRING, [None, 3])],
            [onnx.numpy_helper.from_array(np.int64([0]), "zero"),
             onnx.numpy_helper.from_array(np.int64([3]), "three"),
             helper.make_tensor("unknown", TP.STRING, [1, 3], [v.encode() for v in base_values])]),
            opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("ai.onnx.ml", 3)])
        model.ir_version = 10
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train"
            path.mkdir()
            joblib.dump(bundle, path / "hostname_cascade.joblib")
            contract = hostname_contract(temp, bundle, base_labels,
                                         {"mdns_model": {}, "oui": {}}, hierarchy or {})
            assert_hostname_source(temp, contract)
        model = attach_hostname(model, bundle, self.cols, contract, GB)
        return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]), contract

    def test_onnx_matches_sklearn_rules_and_expert(self):
        sess, contract = self.graph(self.bundle)
        hosts = [normalize_hostname(v) for v in ["IPHONE", "galaxy-a51", "Huyền-iPhone", "a a",
                  "galaxy   a51", "iphone pro", "", "None", "<missing>", "💡💡", "ＦＵＬＬ-iphone"]]
        matrix = np.array([["", h, ""] for h in hosts], dtype=object)
        expected = merge_reference(np.full((len(hosts), 3), UNKNOWN, dtype=object),
                                   hostname_reference(hosts, self.bundle), matrix, self.cols, contract)
        actual = sess.run(None, {"input": matrix})[0]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[0, 0], "Apple")
        self.assertTrue((actual[6:10] == UNKNOWN).all())

    def test_disabled_expert_keeps_rule(self):
        bundle = dict(self.bundle)
        bundle["experts"] = {h: {**v, "threshold": 1.01} for h, v in self.bundle["experts"].items()}
        sess, _ = self.graph(bundle)
        matrix = np.array([["", "iphone", ""], ["", "galaxy-a51", ""]], dtype=object)
        actual = sess.run(None, {"input": matrix})[0]
        self.assertEqual(actual[0, 0], "Apple")
        self.assertTrue((actual[1] == UNKNOWN).all())

    def test_make_requires_manufacturer_evidence(self):
        supported = candidate_make_supported(
            ["laptop-random123", "arbitrary-personal-name", "tivi-sam-sung-7", "galaxy-a51"],
            ["Samsung"] * 4, make_evidence_patterns(self.bundle))
        np.testing.assert_array_equal(supported, [False, False, True, True])
        sess, _ = self.graph(self.bundle)
        actual = sess.run(None, {"input": np.array([["", "a a", ""]], dtype=object)})[0]
        self.assertEqual(actual[0, 0], UNKNOWN)

    def test_packet_laptop_survives_phone_expert_and_model_hierarchy(self):
        base = ["Generic Laptop", "Laptop", UNKNOWN]
        hierarchy = {"Galaxy": {"make": "Samsung", "types": ["Smartphone"]}}
        sess, contract = self.graph(self.bundle, base_values=base, hierarchy=hierarchy)
        matrix = np.array([["", "galaxy-a51", ""], ["", "arbitrary-personal-name", ""]], dtype=object)
        # Force B predictions to clear threshold; the guards must still preserve packet Laptop.
        actual = sess.run(None, {"input": matrix})[0]
        np.testing.assert_array_equal(actual, np.array([base, base], dtype=object))
        host = np.array([["Samsung", "Smartphone", "Galaxy"]] * 2, dtype=object)
        expected = merge_reference(np.array([base, base], dtype=object), host, matrix, self.cols, contract)
        np.testing.assert_array_equal(expected, actual)

    def test_strong_evidence_and_hierarchy(self):
        labels = {"make": [UNKNOWN, "Apple", "Samsung"],
                  "type": [UNKNOWN, "Smartphone", "Tablet"],
                  "model": [UNKNOWN, "iPhone", "Galaxy"]}
        contract = {"labels": labels, "strong_mdns": ["iPhone17,1"], "strong_oui": ["00:11:22"],
                    "masks": {"make": [[1, 1, 1], [1, 1, 0], [1, 0, 1]],
                              "type": [[1, 1, 1], [1, 1, 0], [1, 1, 1]]}}
        matrix = np.array([["00:11:22", "iphone", "iPhone17,1"],
                           ["", "iphone", ""]], dtype=object)
        base = np.array([["Apple", "Smartphone", "iPhone"], [UNKNOWN] * 3], dtype=object)
        host = np.array([["Samsung", "Tablet", "Galaxy"], ["Samsung", "Tablet", "iPhone"]], dtype=object)
        merged = merge_reference(base, host, matrix, self.cols, contract)
        np.testing.assert_array_equal(merged[0], base[0])
        np.testing.assert_array_equal(merged[1], [UNKNOWN, UNKNOWN, "iPhone"])

    def test_notebook_syntax(self):
        for path in Path(__file__).parent.glob("**/*.ipynb"):
            if path.name not in {"05_train.ipynb", "09_build_sdc_iden.ipynb", "12_build_release_onnx.ipynb", "13_export_release_tflite.ipynb"}:
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
            for i, cell in enumerate(document["cells"]):
                if cell["cell_type"] == "code":
                    source = TransformerManager().transform_cell("".join(cell["source"]))
                    compile(source, f"{path.name}:{i}", "exec")


if __name__ == "__main__":
    unittest.main()
