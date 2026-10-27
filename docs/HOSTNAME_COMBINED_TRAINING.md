# Train hostname expert bằng Data_AKien + data tự thu

## Phạm vi

Phần C của `pipeline/train_model/05_train.ipynb` mặc định dùng `HN_DATA_MODE="combined"`.
Đây là hợp nhất dữ liệu train hostname thật sự, không chỉ ghép quyết định của hai model.
Không thay đổi bộ feature/train của fingerprint và OS. Luật legacy và chính sách
nhất quán model/hãng chưa được mở rộng trong thay đổi tập trung vào dữ liệu này.

## Nguồn và nhãn

| Nguồn | Đọc từ | Cách lấy truth |
|---|---|---|
| AKIEN | `Data_AKien/Device_finger print/host_name_tutorial/data/all_data_1604.csv` | Brand/DeviceType/Modelname có sẵn |
| FIELD hostname | `Data/features/dhcp_features_capture.csv` | Hostname thực trong DHCP, không phải tên hiển thị thiết bị |
| FIELD mapping | `Data/sessions_verified.parquet`, chỉ `scenario=FIELD` | Nối MAC + capture_file tới canonical_device |
| FIELD truth | `Data/device_labels_verified.csv` | Nối canonical_device tới make/type/model đã review |

Sessions hiện không có hostname, nên cần phép nối nói trên. Nhãn make/type/model trong
sessions không thay thế bảng nhãn verified; loader không sử dụng nhãn dự đoán.
`device_observations.jsonl` (router MQTT) vào train qua `predata/06_add_router_observations.ipynb`.
06 tự gán nhãn fingerprint (FIELD verified trùng hostname → luật `device_list.csv` → Windows chung),
nhưng `Data/features/dhcp_features_router.csv` cho expert này chỉ gồm thiết bị split train có nhãn
từ FIELD verified hoặc nhãn tay (`verified=1`): nhãn sinh từ luật hostname mà dạy lại expert hostname
thì vòng tròn. Thiết bị split test giữ riêng ở `Data/router_obs/`. Không tự suy ra hãng/model từ OUI.

Chuẩn hóa hostname NFKC/lower/whitespace; `Mobile Phone` → `Smartphone`, `VIVO` → `Vivo`.
Nhãn thiếu, `Null`, `__unknown__` không dùng làm truth. Head make loại các nhãn giả
`Generic Laptop`, `Camera`, `Linova/Linux`, nhưng vẫn giữ type/model của mẫu nếu hợp lệ.
Không tự đổi taxonomy Desktop/Laptop hoặc đoán hãng HP/DELL từ nhãn model legacy.

## Hợp nhất, audit và chống leak

1. Loại hostname rỗng/ngoài giới hạn. Nối FIELD chỉ khi MAC/capture và canonical_device
   được xác nhận trong hai bảng verified; không đưa Unknown-* chưa review vào train.
2. Gộp một hostname thành một mẫu, kể cả xuất hiện nhiều packet hoặc ở cả hai nguồn.
   Nhãn không xung đột được giữ; chỉ mask head xung đột và ghi `hostname_conflicts.csv`.
   Không tự cho nguồn nào thắng khi hai nguồn có nhãn khác nhau.
3. Lưu `source_members` cho mẫu trùng nguồn. `source` chính là FIELD nếu hostname
   thuộc FIELD, để báo cáo không đếm một mẫu hai lần. Các nhãn không thiếu được
   hợp nhất từ cả hai nguồn khi không xung đột.
4. Chia connected components của **template hostname + thiết bị FIELD verified**.
   Mọi hostname của một thiết bị và các template liên quan trong AKIEN ở cùng split.
5. Assert không trùng hostname/template/component/thiết bị giữa train, validation, test.
   Vectorizer và classifier chỉ fit train; threshold chỉ chọn validation.

Seed mặc định 42, test 20% nhóm và validation 20% nhóm trainval. Split tỷ lệ nhóm
không bảo đảm tỷ lệ dòng hoặc từng nguồn đúng 20%. FIELD test hiện rất nhỏ.
Đây là isolation cho hostname expert; fingerprint nền có thể đã học thiết bị FIELD,
vì vậy không gọi metrics hostname là benchmark thiết bị chưa thấy của cả release.

## Cấu hình và cách chạy

```python
HN_DATA_MODE = "combined"   # "akien_only" để thử mốc riêng
HN_FIELD_SESSIONS = HN_ROOT / "Data/sessions_verified.parquet"
HN_FIELD_LABELS = HN_ROOT / "Data/device_labels_verified.csv"
HN_FIELD_DHCP = [HN_ROOT / "Data/features/dhcp_features_capture.csv"]
HN_FIELD_WEIGHT = 1.0
```

Thêm các CSV DHCP tự thu khác vào `HN_FIELD_DHCP` khi chúng có MAC/capture_file
được ánh xạ trong sessions verified. FIELD trọng số 1 như AKIEN; không tự nhân bản
packet để tăng số mẫu. Giữ `HN_MIN_TRAIN=25` hostname train duy nhất cho mỗi lớp;
lớp FIELD-only hiếm có thể chưa đủ mẫu và sẽ không được classifier của head đó học.
`hostname_class_support.csv` và dòng `FIELD actually fitted` cho biết cụ thể.
Nhãn FIELD thiếu không được tự điền hoặc đổi sang lớp phổ biến.

Chạy toàn bộ 05 hoặc riêng các cell C1–C6 trong kernel mới. Chạy riêng C tạo một
run `<timestamp>_hostname_combined`, sao chép iden/os nền vào cùng run, không sửa
run ONNX cũ. Lấy tên run được in cuối phần C, đặt cùng `RUN` ở 09/12/13 và export lại.
Chạy `akien_only` là mốc nguồn riêng nhưng không tự tạo benchmark A/B matched-split;
muốn quy kết mức cải thiện do FIELD cần so sánh trên một tập đánh giá chung cố định.

## Báo cáo

- `hostname_dataset_split.csv`: dataset hợp nhất, nguồn, nhãn, template, device group, split.
- `hostname_field_examples.csv`: tất cả mẫu FIELD sau làm sạch, kèm split và nhãn.
- `hostname_conflicts.csv`: các head bị mask do xung đột.
- `hostname_split_summary.csv`: số mẫu AKIEN/FIELD và số nhóm từng split.
- `hostname_class_support.csv`: lớp đủ/thiếu mẫu, số hostname train từng nguồn.
- `hostname_source_comparison.csv`: metrics A/B/C riêng cho AKIEN và FIELD test.
- `hostname_meta.json`: chế độ, SHA-256 mọi nguồn, audit, sample weight, split counts,
  số FIELD thực sự được fit cho từng head, thresholds và gate.

Các báo cáo trên nằm trong `Models/<RUN>/train/`, cùng các báo cáo A/B/C hiện có.
Gate gộp không phải chứng nhận chất lượng FIELD; luôn đọc n_truth/answered theo nguồn.
Model head có thể vẫn không đạt gate, artifact tiếp tục mang `experimental=True`.

## Kiểm tra

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -X utf8 -m unittest discover -s Code/pipeline -p 'test_hostname*.py' -v
```

11 tests: sáu tests inference/export có sẵn và năm tests dữ liệu, bao gồm nhãn thật
từ verified thay vì sessions prediction, dedup xuyên nguồn, partial labels, mask
xung đột đúng head, component liên thông qua thiết bị/template và phát hiện leak.

Lần kiểm dữ liệu tại workspace ngày 12/10/2026: 41 bản ghi DHCP, 37 bản ghi match
verified, 9 hostname FIELD duy nhất; hai hostname cũng có trong AKIEN. Hợp nhất được
261.029 hostname; split hiện có FIELD 6 train / 2 validation / 1 test, tương ứng
165.453 / 39.612 / 55.964 mẫu tổng. Những con số này sẽ đổi khi bổ sung dữ liệu.
Tập test FIELD một hostname không đủ để kết luận độ nhận diện tổng thể.

## Kết quả train thử đã hoàn tất

Run mới: `20261012_111031_653539_hostname_combined`. Đã chạy C1–C6, lưu bundle/báo cáo
và kiểm tra reload cho cùng kết quả. Không ghi đè run ONNX `20261012_104055`;
run mới hiện có bundle train, **chưa export ONNX/TFLite**.

| Head | Tổng mẫu được fit | Lớp được học | Mẫu FIELD thực sự được fit |
|---|---:|---:|---:|
| make | 63.059 | 20 | 3 |
| type | 165.453 | 8 | 6 |
| model | 61.849 | 179 | 1 |

Các lớp model FIELD-only chưa có 25 hostname train duy nhất nên bị loại khỏi fit;
không phải mọi nhãn của 6 mẫu FIELD train đều được học ở cả ba head.
Gate gộp: make/type pass, model fail do novel false answers. FIELD test `lee_kingdom`
có truth Laptop nhưng expert/C hostname-only trả Smartphone với score khoảng 0,750;
make/model abstain. Đây là **mẫu đoán sai**, không báo cáo như cải thiện nhận diện.
Chính sách giữ head packet đã trả lời vẫn nằm trong pipeline release, nhưng chưa
chạy kiểm release của run combined này. Cần bổ sung FIELD có nhãn và benchmark
chung trước khi coi bản train hợp nhất là bản nâng cấp chất lượng.

Outputs cũ của sáu cell C trong notebook đã được xóa để tránh nhầm kết quả Akien-only
với combined; outputs A/B giữ nguyên. Kết quả train thật được lưu trong thư mục run mới.
