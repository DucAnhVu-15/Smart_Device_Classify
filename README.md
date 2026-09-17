# Smart Device Classification (SDC)

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

44 cột đặc trưng mỗi dòng (28 cột DHCP, 3 cờ nguồn, 5 cột TLS, 4 cột text token hoá,
4 cột khoá tra L0). Bảng đầy đủ nằm ở `FEATURES.md` ngoài repo.

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
44 cột  ──►  L0 luật tất định     (OUI · DHCP hostname · mDNS model)   conf = 1.0
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

## 3. Chạy

Thứ tự bắt buộc:

```
1. Code_data_collection/closedset_field_model.ipynb
      └─► Models/<timestamp>_field_closedset/{model.joblib, meta.json}

2. Code_SDC_V1/08_build_sdc_iden.ipynb
      └─► Models/<timestamp>_iden/sdc_iden.onnx
```

Notebook 2 gọi `latest_run("*_field_closedset")`; không có bước 1 thì nó dừng bằng
AssertionError. `closedset_field_model.ipynb` tự `%run -i` notebook `03_train_model.ipynb`
để lấy `fit_encoder` / `apply_encoder` / `make_model` — dò đường dẫn bằng glob, nên
`03` nằm ở đâu dưới `Code/` cũng được.

**Môi trường:** cần `nbformat` (cho `%run -i` file `.ipynb`), `scikit-learn`, `skl2onnx`,
`onnx`, `onnxruntime`, `pandas`, `pyarrow`, `joblib`, `matplotlib`.

**Kiểm chứng bản dựng.** Contract nhúng trong ONNX mang hai hash; dựng lại đúng thì phải khớp:

```
columns_sha256 = 7be7e5d0deca35db6b34d343a041bf19183199aec58671975dea78b1b2b05773
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
