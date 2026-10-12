# Hostname cascade Data_AKien — bản ngày 12/10/2026

## 1. Phạm vi bản này

Bổ sung nhánh hostname vào pipeline fingerprint và OS hiện có, train trong notebook 05,
đóng gói vào cùng ONNX ở 09/12 và cung cấp triển khai TensorFlow/TFLite ở 13.
Đầu ra release vẫn là ba trường `make`, `type`, `model`; thiếu bằng chứng trả `__unknown__`.

Đây là bản thử nghiệm. Không diễn giải điểm confidence hoặc parity giữa các runtime
thành độ chính xác trên mọi thiết bị thực tế.

| Thành phần | Nguồn dữ liệu | Artifact |
|---|---|---|
| Fingerprint nền | `Data/sessions_verified.parquet`: FIELD tự thu và nền CIC | `train/iden.joblib` |
| OS | Dữ liệu session và nhãn OS của pipeline hiện có | `train/os.joblib` |
| Hostname expert | `Data_AKien/Device_finger print/host_name_tutorial/data/all_data_1604.csv` | `train/hostname_cascade.joblib` |

Hostname expert hiện **chỉ học Data_AKien**, không phải dữ liệu train hợp nhất hai nguồn.
Pipeline kết hợp quyết định của các nhánh. Chạy riêng phần C sẽ sao chép `iden.joblib`
và `os.joblib` từ run nền, không retrain hai bundle đó.
`UA_data.csv` thiếu nhãn cần thiết và SQLite taxonomy chưa được dùng làm mẫu supervised.

## 2. A, B, C là gì?

- **A — hostname rules:** luật regex cho từng head; khớp thì lấy nhãn của luật.
- **B — hostname expert:** TF-IDF `char_wb`, n-gram 3–5, tối đa 60.000 feature,
  sublinear TF, chuẩn hóa L2; SGDClassifier averaged/log-loss riêng cho mỗi head.
- **C — cascade:** toàn bộ luồng A → B → abstain, không phải model thứ ba.

Mỗi head chạy độc lập. B chỉ được gọi khi A không khớp và hostname hợp lệ.
Ngưỡng được chọn trên validation tại các mẫu luật của head đó không khớp,
với target precision 0,90 và tối thiểu 30 câu trả lời validation.
Không có ngưỡng khả thi thì dùng ngưỡng trên 1 để vô hiệu hóa expert của head.
Các lớp cần ít nhất 25 mẫu train để được học.

Hostname được chuẩn hóa NFKC, strip, lowercase và gộp whitespace.
Giá trị thiếu/rỗng và hostname dài quá giới hạn bị từ chối.
Dữ liệu được xử lý nhãn thiếu, loại hostname xung đột nhãn, deduplicate và
chia theo nhóm hostname template thay vì chia ngẫu nhiên từng dòng.
Điều này không thay thế được phép đánh giá riêng trên thiết bị thực chưa thấy.

Trace trong 05 gồm `prediction`, `source`, `confidence`, `threshold`, `rule_pattern`,
`expert_called`, `expert_candidate`, `reason`. `confidence=1` của A chỉ là quy ước
cho rule hit, không phải xác suất đúng đã hiệu chuẩn. Khi abstain, trace vẫn có thể
hiển thị confidence cao của ứng viên bị chặn bởi điều kiện bằng chứng.

## 3. Chính sách ghép với fingerprint

Nhánh hostname được ghép theo từng head, sau đó kiểm hierarchy và chạy cascade OS:

1. Kết quả nền đã trả lời từ mDNS được ưu tiên; OUI được ưu tiên ở head make.
2. Luật hostname A có thể bổ sung/ghi đè kết quả nền không thuộc ưu tiên mạnh trên.
3. **Expert B chỉ bổ sung khi head nền chưa trả lời**, không ghi đè head đã có nhãn.
4. Expert make phải có token hãng/family tương ứng trong hostname; confidence cao
   đơn thuần không đủ để đoán hãng từ `thang` hoặc `LAPTOP-...`.
5. Model B xung đột với type nền đã biết bị loại trước khi áp hierarchy,
   để không làm mất kết quả Laptop vì một dự đoán model điện thoại.
6. Hierarchy loại các nhãn make/type xung đột với model theo bảng cho phép.

Contract `sdc-hostname-graph-v3` (release `20261012_142446`, xem `releases/20261012_142446.md`) bổ sung:

7. Packet `Smartphone` không còn coi là xung đột với model hostname có type `Tablet`
   (`masks.type_guard`, `type_refine`): fingerprint không tách được tablet khỏi điện thoại.
8. Model expert B xung đột với make đã biết (packet hoặc hostname) bị loại (`guard_model_make`),
   ví dụ `samsung-galaxy-a34` không còn ghép make Samsung với model `OPPO A34`.
9. Khi model lấy từ hostname, make/type còn trống được điền theo hierarchy của model
   (`model_implies`, bảng `implied`); type chỉ điền khi model có đúng một type,
   và `Smartphone` của packet được nâng lên `Tablet`. Dòng chỉ có packet không bị điền.

Ở release (12/13), `TYPE_OK` của iOS/Android chấp nhận `Tablet`, và output gộp `Smartphone`
và `Tablet` thành `Smartphone/Tablet` (`output_type_alias` trong contract release).
`labels` trong contract vẫn là nhãn nội bộ chưa gộp.

Mọi chính sách được ghi trong contract cùng SHA-256 của bundle, nhãn mở rộng, thresholds,
provenance/gate và trạng thái experimental. Contract cũ vẫn được hỗ trợ để kiểm parity
theo chính sách của model cũ.

## 4. Các file thay đổi

| File | Vai trò |
|---|---|
| `pipeline/train_model/05_train.ipynb` | Thêm phần C: load/clean/split, train/calibrate, trace, đánh giá và lưu bundle cùng run |
| `pipeline/hostname_support.py` | Dùng chung normalization, inference tham chiếu, contract, merge policy, ONNX exporter, TF module, batch inference |
| `pipeline/out_model/09_build_sdc_iden.ipynb` | Nung hostname vào graph iden, mở rộng nhãn, kiểm parity và ghi contract |
| `pipeline/out_model/12_build_release_onnx.ipynb` | Ghép OS với nhãn cuối của C; giữ phụ thuộc graph If khi prune; kiểm hồi quy |
| `pipeline/out_model/13_export_release_tflite.ipynb` | Tái dựng nhánh hostname trong TF, giữ nhãn rừng nền riêng với nhãn mở rộng, kiểm nguồn bundle và thêm probes |
| `test_release_observations.py` | Chạy không cần tham số trên cấu hình local, hỗ trợ `--data/--model/--output`, test có/không hostname và xuất CSV |
| `pipeline/test_hostname_support.py` | Sáu bài kiểm tra syntax, parity, rules, abstain, bằng chứng hãng và bảo toàn type nền |

ONNX dùng encoder n-gram ký tự tương đương sklearn. Nhánh hostname chỉ được thực thi
khi batch có hostname; helper `run_onnx` chia batch mặc định 32 để hạn chế bộ nhớ
tensor hostname dense. Bundle tải cho inference bỏ các buffer SGD chỉ dùng lúc train.
Phần hiển thị bảng so sánh C không phụ thuộc Pandas Styler/Jinja2.

## 5. Cách chạy

Thư mục làm việc là project root chứa `Code/`, `Data/`, `Data_AKien/`, `Models/`.
Các dataset và artifact nặng không nằm trong repo Git; cần chuẩn bị riêng.

1. Chạy notebook **05** để tạo run có đủ ba bundle train. Lấy tên run từ
   `run_dir` hoặc `hn_output_dir.parent.name` nếu chạy riêng phần C.
2. Đặt **cùng `RUN`** trong **09, 12, 13**, không phụ thuộc `None = mới nhất`
   khi có nhiều run hoặc chạy notebook không theo thứ tự.
3. Chạy **09** với `HOSTNAME_MODE="required"` → `Models/<RUN>/iden/sdc_iden.onnx`.
4. Chạy **12** → `Models/<RUN>/release/sdc_os.onnx`, contract và manifest.
5. Test observations; chạy **13** nếu cần TFLite và môi trường TensorFlow phù hợp.

Run ONNX đã test tại workspace ngày 12/10/2026: `20261012_104055`.
09 và script observations hiện tham chiếu run này; 12/13 vẫn mặc định `RUN=None`,
vì vậy nên đặt tường minh trước khi chạy. Với run khác, sửa cấu hình hoặc dùng CLI:

```powershell
python Code/test_release_observations.py --data device_observations.jsonl --model Models/20261012_104055/release/sdc_os.onnx --output Reports/manual_hostname_test/release_observations.csv
```

Mặc định script tạo `Reports/<timestamp>/release_observations.csv`.
Các đường dẫn local ở đầu script cần chỉnh nếu clone sang máy khác.

## 6. Kiểm tra và kết quả ghi nhận

Chạy unit tests từ project root:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -X utf8 -m unittest discover -s Code/pipeline -p test_hostname_support.py -v
```

Sáu tests đã pass khi chuẩn bị commit, gồm syntax notebook, ONNX/sklearn parity,
rule/threshold/Unicode/missing inputs, maker evidence, ưu tiên nguồn mạnh và
bảo toàn packet Laptop khi expert dự đoán điện thoại/model xung đột.

Manifest local của run `20261012_104055` ghi nhận:

- 62 dòng capture và 5.010 probe OS: không lệch giữa ONNX và reference release.
- Kiểm 7.280 tổ hợp cascade OS.
- `capture_broken=0`, `iot_general_type=0` trong kiểm hồi quy.

Báo cáo observations local `Reports/20261012_104830/release_observations.csv`:

| Hostname | make | type | model |
|---|---|---|---|
| `LAPTOP-VAF70SQ6` | Generic Laptop | Laptop | `__unknown__` |
| `LAPTOP-49956CF1` | Generic Laptop | Laptop | `__unknown__` |
| `thang` | `__unknown__` | Laptop | `__unknown__` |

Kết quả giống nhau ở hai chế độ có/không hostname: ba thiết bị, sáu lượt suy luận,
không phải sáu thiết bị độc lập. Đã hết lỗi hostname tự đoán Samsung/làm mất type Laptop
trên các mẫu này. Chưa chứng minh tăng coverage hoặc biết hãng/model chính xác hơn.
`Generic Laptop` là nhãn legacy, không phải hãng thực tế.

Gate hostname-only của run này: make/type pass, **model fail**; artifact vẫn được lưu
ở trạng thái experimental. Gate pass không bảo đảm chất lượng trên miền data tự thu.
Code TFLite đã bổ sung nhưng **chưa xác nhận export/parity TFLite cho run này**;
không coi kiểm tra ONNX là kiểm chứng TFLite.
Warning ONNX initializer `thresholds` còn trong graph inputs liên quan tối ưu hóa,
không phải nguyên nhân của lỗi đoán hãng đã sửa.

## 7. Giới hạn còn lại và việc chưa triển khai

- Luật legacy `desktop-* → Laptop` vẫn tồn tại; không coi đây là luật chắc chắn đúng
  và cần rà lại taxonomy trước khi sửa nhãn Desktop/Laptop.
- Từ v3, model hostname điền hãng qua hierarchy (`a34-cua-nga` → Samsung). Head model
  vẫn chưa đạt gate (novel false answers), nên một model sai ở ngưỡng 0,3 giờ kéo theo
  cả make/type sai. Cần benchmark có nhãn trước khi coi đây là cải thiện.
- Chưa hợp nhất hostname có nhãn từ data tự thu vào tập train Data_AKien.
- Chưa thực hiện benchmark ba cấu hình: fingerprint nền, expert Akien-only,
  expert train hợp nhất, trên cùng tập thiết bị chưa dùng train.
- Cần tách dữ liệu theo thiết bị/phiên thu và kiểm template trùng giữa các tập,
  đo riêng precision/coverage/novel false answers và số kết quả đúng nền bị làm sai.
- Không dùng dự đoán của model làm nhãn thật, không điền hãng/model còn thiếu.

Các mục trên là kế hoạch tiếp theo, **không phải tính năng đã hoàn thành trong commit này**.
Dataset, báo cáo CSV và model ONNX/joblib/TFLite nằm ngoài repo và không được push
cùng bản code này; notebook có thể giữ các output kiểm tra của lần chạy local.
