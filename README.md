# Smart Device Classification (SDC)

## Bản hostname Data_AKien — 12/10/2026

Xem [mô tả đầy đủ bản hostname cascade](docs/HOSTNAME_DATA_AKIEN_20261012.md): nguồn dữ liệu,
kiến trúc A/B/C, chính sách chống ghi đè, cách chạy 05 → 09 → 12 → 13, kiểm tra và giới hạn.
Bản ONNX `20261012_104055` trong tài liệu trên dùng expert Akien-only.
Phần C của 05 hiện mặc định **train chung Data_AKien + hostname tự thu có nhãn verified**;
xem [cách train hai nguồn và đọc audit](docs/HOSTNAME_COMBINED_TRAINING.md).
Data/model vẫn nằm ngoài repo; cần train/export run mới để áp dụng thay đổi này.

Các mục bên dưới mô tả pipeline nền; tài liệu bản hostname giải thích phần mở rộng.

Nhận dạng thiết bị trong mạng LAN từ traffic thụ động: DHCP, DNS, mDNS và TLS ClientHello.
Đầu ra là ba nhãn — `make` (hãng), `type` (loại thiết bị), `model` (dòng sản phẩm) — đóng gói
thành **một file ONNX duy nhất** chạy được trên router.

> Repo này chỉ chứa notebook. `Data/` và `Models/` nằm ngoài repo (xem [.gitignore](.gitignore)):
> pcap, parquet, `.joblib` và `.onnx` không được push.

---

## 1. Dữ liệu

Toàn bộ pipeline đọc từ một file: `Data/sessions_verified.parquet` — **8.189 dòng**, mỗi dòng
là một *cửa sổ* quan sát của một MAC trong một file pcap.

| `scenario` | số dòng | thiết bị | nguồn |
|---|---:|---:|---|
| `IDLE` | 8.022 | 39 | CIC IoT 2022 |
| `POWER` | 117 | | CIC IoT 2022 |
| `FIELD` | **50** | **12** | tự thu, 14/09/2026 |

Rừng học trên 40 cột đặc trưng mỗi dòng (28 cột DHCP, 3 cờ nguồn, 5 cột TLS, 4 cột text
token hoá). Model deploy chỉ nhận **12 cột chuỗi**: 9 cột chuỗi trong số đó cộng 3 cột khoá
tra L0. 31 cột số còn lại (`has_*`, `dhcp_opt_*`, `dhcp_prl_len`) được tính ngay trong graph
từ các chuỗi. Bảng đầy đủ nằm ở `FEATURES.md` ngoài repo.

### Vai trò hai tập

Đây là bài toán **closed-set**: 12 thiết bị FIELD là các lớp thật, còn 8.139 dòng CIC-2022
**không** làm lớp thật mà làm **nền `__unknown__`** — dạy cho rừng biết từ chối thiết bị lạ
thay vì ép chúng vào một lớp đã học.

Nhãn FIELD (50 dòng):

```
make   Generic Laptop 22 | Camera 6 | Linova/Linux 6 | Samsung 6
       Raspberry Pi 3 | Xiaomi 3 | Apple 3 | OPPO 1
type   Laptop 28 | Smartphone 13 | IP Camera 6 | Single-board Computer 3
model  Windows Desktop DELL 11 | Generic IP Camera 6 | Windows Laptop HP 6
       Linova Laptop HP 6 | Samsung Galaxy 6 | Windows Desktop HP 5
       Raspberry Pi 3 | iPhone 3 | Redmi Note 14 Pro 2 | OPPO A92 1 | Redmi Note 10 1
```

### Hai chỗ đã sửa so với bản train ngây thơ

**FIX 1 — bỏ trần `TF-IDF max_features`.** Trần cũ (200/100/150/100) được đặt khi tập train
có hàng trăm phiên mỗi lớp. Với 50 dòng FIELD chọi 8.139 dòng CIC, vocab bị CIC chiếm chỗ và
trần cắt mất 71–75% token nhận dạng của FIELD. Bỏ trần: input tensor nở 518 → **1.088 feature**.

**FIX 2 — không ép dòng CIC trùng nhãn thành `__unknown__`.** `np.where(is_field, nhãn, UNKNOWN)`
gán tất cả dòng CIC thành `__unknown__`, kể cả dòng mang đúng nhãn mà một lớp FIELD đang giữ.
Head `type` dính nặng nhất: 2.471 dòng CIC có `type == "IP Camera"` thật, chọi 6 dòng FIELD
cùng nhãn — rừng bị dạy hai điều ngược nhau trên cùng một nhãn, tỉ lệ 412:1. Các dòng đó bị
**loại khỏi tập train của head đang xét** thay vì dán nhãn sai.

Số dòng thực sự vào train: `make` 8.189 · `type` **5.718** · `model` 8.189.

### Tập test

`Data/features/*_capture.csv` — **62 cửa sổ / 15 MAC / 6 file pcap**, tách theo hai phần:

- **50 cửa sổ có nhãn — IN-SAMPLE.** 12 MAC này *chính là* tập FIELD đã train. Sai ở đây là
  lỗi thật, nhưng đúng ở đây không chứng minh được gì.
- **12 cửa sổ / 3 MAC `Unknown-*` — held-out thật.** Không có trong bảng nhãn nên bị loại khỏi
  train; model chưa từng thấy. Đây mới là phép đo có ý nghĩa.

Đây cũng là bộ duy nhất có `dhcp_hostname` thật (47/62 dòng), tức lần đầu tầng L0/hostname
được chạy trên dữ liệu không phải bịa ra.

---

## 2. Kiến trúc quyết định

```
12 cột  ──►  tự tính 31 cột số (has_* · dhcp_opt_* · dhcp_prl_len)
             │
             ▼
             L0 luật tất định     (OUI · DHCP hostname · mDNS model)   conf = 1.0
             │ trượt
             ▼
             L1 vân tay exact     (dhcp_prl + dhcp_vci + tls_fp)
             │ trượt / nhập nhằng
             ▼
             L2 RandomForest      ──► conf < ngưỡng ──► __unknown__
             │
             ▼
             kiểm tra hierarchy   ──► mâu thuẫn ──► __unknown__
             │
             ▼
             make · type · model
```

Cả ba tầng nằm **trong cùng một graph ONNX**. Không có tầng policy nào bên ngoài — trừ việc
gộp nhiều cửa sổ của cùng một MAC (`device_log`), là state xuyên lời gọi nên graph stateless
không giữ được.

**Ngưỡng abstain** tra theo `n_sources` (số nguồn quan sát được trong cửa sổ):

| head | 0 nguồn | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `make` | 1.01 | 0.75 | 0.65 | 0.60 | 0.55 |
| `type` | 1.01 | 0.70 | 0.60 | 0.52 | 0.48 |
| `model` | 1.01 | 0.80 | 0.70 | 0.62 | 0.56 |

Giá trị `1.01` là cố ý và không thể đạt: **một nguồn duy nhất thì luôn abstain**.

**Classifier:** RandomForest, `n_estimators=250`, `class_weight="balanced"`, `random_state=42`,
ba head độc lập dùng chung một encoder.

---

## 3. Cấu trúc và thứ tự chạy

```
Code/
├── pipeline/
│   ├── predata/                 1. dựng dữ liệu
│   │   ├── 01_extract_cic.ipynb           pcap CIC-2022 → Data/features/*_all.csv
│   │   ├── 02_extract_capture.ipynb       pcap tự thu → Data/features/*_capture.csv
│   │   ├── 02b_extract_capture_by_ip.ipynb pcap gắn thiết bị theo IP → Data/features/test_pcap_collection/
│   │   ├── 03_build_dataset.ipynb         *_all.csv → Data/sessions_verified.parquet
│   │   └── 04_add_capture_devices.ipynb   gộp thiết bị FIELD → sessions(_verified).parquet
│   ├── train_model/             2. train — MỘT file
│   │   └── 05_train.ipynb                 → Models/<ts>/train/  iden.joblib (make/type/model) + os.joblib (head OS)
│   └── out_model/               3. đóng gói model
│       ├── 09_build_sdc_iden.ipynb        iden.joblib → Models/<ts>/iden/sdc_iden.onnx
│       ├── 12_build_release_onnx.ipynb    sdc_iden.onnx + os.joblib → Models/<ts>/release/sdc_os.onnx  (BẢN DEPLOY)
│       └── 13_export_release_tflite.ipynb release → cùng thư mục: sdc_os.tflite (cần TensorFlow; chạy cần Flex)
├── router/hostname_sources.py   đặc tả nguồn hostname cho agent trên router (stdlib, chưa nối vào pipeline)
└── archive/                     notebook cũ (gồm nhánh open-set 06/07/08/10/11) — xem archive/README.md
```

Mở từng notebook theo số thứ tự rồi Run All. Không có notebook thư viện chung: mỗi notebook tự chứa đủ code nó cần (mục **Thư viện** ở đầu notebook).

### Khâu model: một chuỗi thẳng

Một lần chạy pipeline = một thư mục `Models/<ts>/` (05 tạo), các bước sau ghi vào thư mục con của nó:

```
05  train ─► <ts>/train/  iden.joblib + os.joblib + meta.json
09  đóng gói iden ─► <ts>/iden/sdc_iden.onnx              (contract ghi source_run = <ts>)
12  ghép iden + OS ─► <ts>/release/  sdc_os.onnx, sdc_os_contract.json, manifest.json
13  xuất TFLite    ─► cùng <ts>/release/  sdc_os.tflite, sdc_os_tflite_contract.json, parity_inputs.npz
```

- **Train ở đúng một chỗ:** `05_train.ipynb` — phần A head iden, phần B head OS, cùng ghi vào một run.
- **12 không chọn head OS riêng:** nó lấy `train/os.joblib` của cùng run và kiểm `source_run` của `sdc_iden.onnx`
  trỏ đúng run đó, nên không thể ghép iden của lần train này với OS của lần train khác.
- **Bản deploy = `release/` của run mới nhất.** Chạy lại 09 hoặc 12 trên cùng run thì ghi đè bước đó của run
  (12 xoá cả `release/` cũ, kể cả TFLite); muốn giữ bản cũ thì chạy 05 để có run mới.
  `manifest.json` ghi run nguồn, sha256 và kết quả kiểm của từng file; 13 thêm `sdc_os.tflite` vào đó.
- Thư mục dạng cũ `<ts>_train/`, `<ts>_iden/`, `<ts>_release/` trong `Models/` không còn được chọn tự động.
- **`sdc_os.onnx` và `sdc_os.tflite` là cùng một model:** 13 kiểm từng nhãn TFLite khớp `sdc_os.onnx`.

### Chạy riêng một notebook

Mỗi notebook tự chứa đủ code (mục **Thư viện** ở đầu), đọc input từ đĩa và ghi output ra đĩa. Mở notebook nào rồi
Run All cũng được, miễn input ở cột 2 đã có. Ngoại lệ duy nhất là `04`: nó tự `%run` lại `02` để lấy bản ghi
pcap trong bộ nhớ.

Run đầu vào chọn bằng biến `RUN` ở đầu notebook 09, 12, 13: tên thư mục `Models/<ts>/`, hoặc `None` = run mới
nhất đã có đầu ra của bước trước (hàm `pick_run`). Đặt cùng một giá trị `RUN` ở cả ba notebook để làm việc trên
một run cụ thể.

| Notebook | Cần có sẵn | Ghi ra | Chọn run |
|---|---|---|---|
| `01` | pcap CIC-2022 `Data/1-Power`, `Data/2-Idle` (chỉ server) | `Data/features/*_all.csv` | — |
| `02` | pcap tự thu trong `CAPTURE_DIR`, `sessions_verified.parquet` | `Data/features/*_capture.csv` | — |
| `02b` | `CAPTURE_DIR/test_pcap_collection.pcap`, `device_labels_capture.csv` | `Data/features/test_pcap_collection/` | — |
| `03` | `Data/features/*_all.csv`, bảng nhãn | `Data/sessions_verified.parquet` | — |
| `04` | pcap tự thu (chạy lại `02`), `Data/sessions.parquet` | `sessions.parquet`, `sessions_verified.parquet` | — |
| `05` | `sessions_verified.parquet`; đánh giá cần `test_model/data_test/out/windows.jsonl` | run mới `Models/<ts>/train/` | — |
| `09` | `<ts>/train/`, `sessions_verified.parquet`, `*_capture.csv` | `<ts>/iden/sdc_iden.onnx` + báo cáo | `RUN` |
| `12` | `<ts>/iden/` + `<ts>/train/`; mục 5 cần `sessions_verified.parquet`, `*_capture.csv` | `<ts>/release/` | `RUN` |
| `13` | `<ts>/release/` + `iden/`, `train/` cùng run | thêm `sdc_os.tflite` vào `<ts>/release/` | `RUN` |

### Muốn sửa gì thì sửa ở đâu

| Muốn đổi | Sửa ở | Chạy lại |
|---|---|---|
| Cách train head iden (FIX 1/2, tập train) | `05_train` phần A | `05` → `09` → `12` → `13` |
| Feature, nhãn OS hay cách train head OS | `05_train` phần B (`FIELD_OS`, `os_feature_cols`, `fit_os`) | `05` → `09` → `12` → `13` |
| Luật L0 của head iden (hostname, mDNS `model=`) | `09`, `DEFAULT_HOSTNAME_RULES` / `DEFAULT_MDNS_MODEL_MAP` | `09` → `12` → `13` |
| Ngưỡng abstain của head iden | `09`, `DEFAULT_THRESHOLDS` (hoặc input `thresholds` lúc chạy) | `09` → `12` → `13` |
| Hàm trong mục **Thư viện** (trích xuất pcap, encoder, contract 12 cột, luật OS, `cascade`…) | **mọi** notebook có bản sao — bảng dưới | các bước từ notebook sớm nhất bị sửa |

### Hàm có bản sao ở đâu

Mục **Thư viện** ở đầu mỗi notebook là bản chép **nguyên văn** của cùng một code; mỗi notebook chỉ chứa phần nó
dùng. Sửa một hàm ở đây thì sửa y hệt ở mọi notebook trong cột 3, nếu không các bước sẽ lệch nhau (ví dụ luật OS
ở `05` khác `12` → model train một kiểu, file deploy chạy một kiểu).

| Nhóm (dòng đầu cell Thư viện) | Hàm chính | Có bản sao ở |
|---|---|---|
| Đường dẫn dự án, `ROOT`, lược đồ dữ liệu | `load_sessions`, `feature_sets`, `normalize_heads`, `assert_serveable` | 01, 02, 02b, 03, 04, 05, 09, 12, 13 |
| Trích xuất feature từ pcap và bản ghi raw | `aggregate`, `extract_dhcp`, `extract_dns`, `TLSStreamReassembler`, `domain_tokens`… | 01, 02, 02b, 03, 04, 05, 12 |
| Dựng dataset đã xác minh | `build_verified_dataset`, `validate_taxonomy` | 04 |
| Bộ phân lớp L2 | `make_model` | 05, 09, 12 |
| Bộ mã hoá feature (TF-IDF + ordinal) | `fit_encoder`, `apply_encoder`, `prepare_frame`… | 05, 09, 12, 13 |
| Contract `sdc_iden` — input 12 cột | `INPUT_COLUMNS`, `add_l0_columns`, `derive_numeric`, `to_input_matrix`, `mac_oui`, `build_hierarchy`, `capture_frame`… | 05, 09, 12, 13 |
| Head OS | `HOST_OS`, `l0_os`, `l1_os`, `os_decide`, `TYPE_OK`, `MAKE_OK`, `cascade` | 05, 12, 13 |
| Khối dựng graph ONNX | `GB`, `relink`, `str_present`, `set_locale`… | 09, 12 |
| Xuất rừng sang mảng phẳng và TFLite | `flatten_forest`, `rename_io`, `embed_metadata`, `read_metadata` | 12, 13 |

- **Bản deploy** cần `01 → 05 → 09 → 12` (thêm `02`–`04` khi có pcap tự thu mới; `13` khi cần TFLite).
- `01` cần pcap thô CIC-2022 (`Data/1-Power`, `Data/2-Idle`), chỉ có trên server.
- Thư mục pcap tự thu là `CAPTURE_DIR` (mục Thư viện): tự chọn `Data/data_pcap` (server) hoặc
  `Data/data_pcap_capture` (máy local).
- ⚠️ **Đừng chạy `02` / `04` trên máy local.** Ở đây `Data/data_pcap_capture/` chỉ có `test_pcap_collection.pcap`; 6 file pcap FIELD chỉ có trên server. Chạy `02` ở local sẽ **ghi đè** `Data/features/*_capture.csv` (bộ test 62 cửa sổ) bằng dữ liệu khác, rồi `04`, `09`, `12` hỏng theo. Khâu model (`05` → `13`) chạy local bình thường.

⚠️ **Nguồn dữ liệu chưa khép kín.** `03` ghi `sessions_verified.parquet` từ `*_all.csv`, nhưng `04`
dựng lại chính file đó từ `Data/sessions.parquet` — tức kết quả của `03` bị ghi đè. Phần CIC trong
`sessions.parquet` hiện không bước nào sinh ra; nó là file có sẵn từ phiên bản cũ.

`09_build_sdc_iden` lấy run `Models/<ts>/` mới nhất có `train/iden.joblib` (hoặc `RUN`); chưa có run nào của `05` thì nó dừng bằng AssertionError.

**Môi trường:** cần `nbformat` (cho `04` `%run` notebook `02`), `scikit-learn`,
`skl2onnx`, `onnx`, `onnxruntime`, `pandas`, `pyarrow`, `joblib`, `matplotlib`, `scapy`.
Riêng `13` cần thêm `tensorflow` (Python 3.14: `pip install --pre tensorflow`).
File của `13` dùng op Flex: `tensorflow` ≥ 2.20 (mọi OS) **không** chạy được nó vì đã bỏ Flex delegate khỏi
`tf.lite.Interpreter` — notebook khi đó đối chiếu bằng graph TF và ghi `parity_inputs.npz`; kiểm file `.tflite`
bằng `tensorflow==2.16.2` trên Linux (Python 3.12), bản này còn Flex (đoạn code mẫu ở mục 5 của 13).

**Kiểm chứng bản dựng.** Contract nhúng trong ONNX mang hai hash; dựng lại đúng thì phải khớp:

```
columns_sha256 = 4537a2bbdec921acbe56d10f4f984c2bbeab0b19fd8e67fa71f0ace2ceba6261   (sdc-iden-onnx-v2, 12 cột)
labels_sha256  = db44cab1f7ab501193ef0526d4886f3985cc0d3d65237df65b146517c9b17a07
```

---

## 4. Những gì bản này **không** làm được

Phần này không phải TODO. Đây là các giới hạn đã đo, cần đọc trước khi tin vào bất kỳ con số nào.

**Đánh giá FIELD là in-sample.** 50 dòng train, 1 ngày capture, 1 MAC mỗi thiết bị. Các lớp
dưới `RARE_THRESHOLD=10` — Raspberry Pi 3 phiên, iPhone 3, Redmi Note 14 Pro 2, OPPO A92 1,
Redmi Note 10 1 — là ghi nhớ thuộc lòng, không phải khái quát hoá.

**`Linova Laptop` không có đặc trưng nội tại nào khác 4 laptop FIELD còn lại:** chung `tls_fp`,
không mDNS, không `dhcp_vci`, `dhcp_prl` trùng hệt Raspberry Pi. Thứ duy nhất tách được nó là
tập token DNS/SNI của người dùng — nên nó sẽ trượt khi chạy thật.

**Thiếu DHCP không làm model trả sai, nhưng làm nó im lặng.** Đo bằng cách xoá sạch mọi cột
DHCP trên 50 cửa sổ capture có nhãn: trong 150 quyết định, 5 quyết định đổi và **cả 5 đều là
`đúng → __unknown__`**, không có ca nào thành nhãn sai. Nguyên nhân là `dhcp_hostname`
(= DHCP option 12) nuôi tầng L0, mất DHCP là tầng đó chết. Ở quy mô dataset, chỉ **10,9%**
cửa sổ có DHCP trong khi **48/51 thiết bị** có ít nhất một cửa sổ DHCP — đây là bài toán cắt
cửa sổ, không phải bài toán thiếu dữ liệu.

**Ngưỡng tra theo *số* nguồn chứ không theo *tổ hợp* nguồn.** Một cửa sổ `DNS+TLS` và một cửa
sổ `DHCP+mDNS` đều là `n_sources=2` và nhận cùng ngưỡng, dù `dns_tokens` + `tls_sni_tokens`
chiếm 76,8% importance còn toàn bộ 28 cột DHCP dưới 2%.

**Dựng lại không cho ra model giống hệt.** Chạy lại đúng code này trên một bản Python khác
(cùng `scikit-learn 1.9.0`, `numpy` lệch 2.5.1 ↔ 2.5.2) cho ra rừng khác: số nút
`[19258, 16548, 18636]` → `[16570, 13196, 19486]`, dù `random_state=42`. Encoder thì giống hệt
(cùng TF-IDF pool `[32, 575, 118, 327]`, cùng 1.088 feature), và `columns_sha256` /
`labels_sha256` vẫn khớp. Chênh lệch đầu ra rất nhỏ và một chiều: **9/24.567 ô** trên toàn bộ
8.189 dòng, toàn bộ theo hướng `__unknown__` → nhãn thật, không có chiều ngược lại. Nguyên
nhân chưa xác định; nghi do khác bản numpy. Nếu cần tái lập chính xác thì phải ghim cả phiên
bản numpy, không chỉ `random_state`.
