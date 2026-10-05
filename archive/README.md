# archive/

Notebook không thuộc pipeline chạy tuần tự, giữ lại để tham khảo. Lịch sử git vẫn đầy đủ.

| File | Vì sao ở đây |
|---|---|
| `02_1_export_preprocessed.ipynb` | Chỉ để **xem** dữ liệu ở đúng điểm trước khi train (40 cột → 516 cột số). Không sinh artifact nào bước sau cần. Không còn chạy được như cũ: nó nạp `pipeline/00_common.ipynb`, notebook đã bỏ (lấy lại bằng `git show 869d471:pipeline/00_common.ipynb`). |
| `06_train_openset.ipynb`, `07_evaluate_openset.ipynb`, `08_test_openset.ipynb`, `10_activate_openset.ipynb`, `11_export_onnx_openset.ipynb` | Nhánh open-set (contract `sdc-tiered-v2`: L1 vân tay + ngưỡng theo tổ hợp nguồn, `Predictor`, `DeviceTracker`). Bản deploy không dùng; pipeline giờ chỉ train một chỗ (`05_train`). **Không chạy được nữa**: chúng nạp `pipeline/00_common.ipynb` (đã bỏ) và cần các hàm open-set (`Predictor`, `DeviceTracker`, `FingerprintTable`, `train_main`, `evaluate_main`, `select_run_dir`…). Muốn chạy lại thì lấy `00_common` cũ về đúng chỗ: `git show 869d471:pipeline/00_common.ipynb > pipeline/00_common.ipynb`. |
