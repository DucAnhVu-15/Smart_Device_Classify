# Rule hostname từ catalog thiết bị

## Phạm vi

Bổ sung rule A (regex, first-match-wins) cho cascade hostname từ danh sách thiết bị hiện nay
(`Search/device-list.md`). Không đổi feature/train của fingerprint, OS hay expert B.

| File | Vai trò |
|---|---|
| `pipeline/hostname_catalog.csv` | Nguồn sửa tay: mỗi dòng một thiết bị và cách người dùng gõ nó trong hostname |
| `pipeline/hostname_catalog.py` | Sinh regex, sắp thứ tự, đối chiếu nhãn thật, ghi rule đã nhận |
| `pipeline/hostname_catalog_rules.json` | Rule + hierarchy mà phần C của 05 nạp sau rule mặc định của 09 |
| `pipeline/hostname_catalog_report.csv` | Thống kê từng (dòng, head), không commit (`*.csv`) |

## Cột catalog

| Cột | Ý nghĩa |
|---|---|
| `make`, `type`, `model` | Nhãn trả ra; để trống head nào thì dòng không chốt head đó |
| `names` | Tên trong hostname, cách nhau `\|`: `s24 ultra\|s24u`. Dấu cách/chữ-số cho phép `-_. ` hoặc dính liền |
| `prefixes`, `require_prefix` | Tiền tố hãng (`galaxy\|samsung`). `1` = bắt buộc, dùng khi tên trần trùng hãng khác (`oppo a18`, `itel s24`) |
| `pattern`, `examples` | Regex viết tay (bắt đầu `(?i)`, cú pháp RE2) và hostname mẫu bắt buộc đi kèm |
| `group` | Nhóm báo cáo: phone, tablet, laptop, tv, gaming_xr, printer, camera, model_code |

Tên cần ranh giới ký tự hai đầu: `s24` khớp `s24-cua-an`, `an-s-s24`, `galaxy-s24` nhưng không khớp
`s240` hay bên trong `s24ultra`. `model = __unknown__` là rule chặn: hostname khớp thì head đó dừng ở A,
không trả nhãn và không gọi expert, nên kết quả gói tin giữ nguyên. Camera `nomi-ipc-s21fe` dùng
cách này để không thành Galaxy S21 FE.

Dùng lại đúng chính tả nhãn đã có trong dataset (`Samsung A51`, `Samsung Galaxy A15 5G`,
`Redmi Note 10`), nếu không sẽ thành lớp mới. Script in `Nhãn viết khác dataset` khi phát hiện.
Type chỉ dùng 8 type hiện có; gaming/XR/máy in không có type phù hợp nên chỉ chốt make/model.

## Thứ tự và kiểm tra

Dòng A phải chạy trước dòng B khác nhãn nếu pattern B khớp hostname mẫu của A (S24 Ultra trước
S24, OPPO A18 trước A18). Script sắp tô-pô theo quan hệ này; chồng lấn vòng tròn thì báo lỗi.
Hòa thì pattern viết tay (mã máy, tiền tố camera, TV OS) đi trước tên do người dùng gõ.

Mỗi (dòng, head) được chấm trên `train + validation` của `hostname_dataset_split.csv`, theo đúng
cascade: rule mặc định 09 trước, catalog sau. Test chỉ để báo cáo, không dùng chọn rule.

- `correct`; `compatible` khi nhãn thật là nhãn chung hơn cùng hãng (`Google Pixel` với
  `Google Pixel 7`, `OPPO Reno` với `OPPO Reno6`); `Samsung A51` = `Samsung Galaxy A51`.
- `wrong` cả khi head đang chấm chưa có nhãn nhưng head khác mâu thuẫn (host là IP Camera).
- Loại khi có ≥ 5 mẫu chấm và precision < 0,95, hoặc < 5 mẫu mà sai nhiều hơn đúng.
  Make/type bị loại thì loại cả dòng (`rejected_row`).
- `unverified`: chưa có mẫu, thường là máy mới (S26, Fold7). Vẫn được dùng.

Nhãn `Blu Vivo One Plus` của AKIEN gán cho `oneplus-9-pro...` không được dùng làm truth.

## Cập nhật catalog

```powershell
python -X utf8 Code/pipeline/hostname_catalog.py --split Models/<RUN>/train/hostname_dataset_split.csv
python -X utf8 -m unittest discover -s Code/pipeline -p "test_hostname_catalog.py" -v
```

Đọc `hostname_catalog_report.csv` (cột `status`, `wrong_examples`) trước khi train. Sau đó chạy phần C
của 05, rồi 09 → 12 → 13 cùng `RUN`. Test kiểm từng hostname mẫu của mọi dòng ra đúng nhãn của chính
nó qua cả cascade, pattern không dùng cú pháp ngoài RE2, và JSON khớp SHA của catalog.

## Kết quả lần đầu (13/10/2026, split của run `20261012_142446`)

784 dòng → 881 rule (gộp rule liền nhau cùng nhãn, mỗi pattern < 2.000 ký tự để DFA RE2 không
hết bộ nhớ; 881 + 17 rule mặc định khớp Python/onnxruntime trên 29.255 hostname). Model: 126 verified, 625 unverified (chưa có mẫu), 26 bị loại. Trên test
giữ riêng: model 17.260 đúng / 23 sai (0,9987), make 0,9986, type 0,9998.

Bị loại chủ yếu là tên Samsung trần: `a12`, `a31`, `a52`… cũng là hostname OPPO trong AKIEN, và
AKIEN gán `s24`/`s25` trần cho bản Ultra. Dòng bắt buộc tiền tố (`galaxy-a12`) vẫn được giữ;
hostname trần do expert B xử lý như trước. Hostname Xiaomi có mã máy (`m2006c3lg-…`) được 14 rule
mã máy lấy từ train+validation (≥ 5 mẫu, một nhãn ≥ 98%) bắt trước phần tên người dùng gõ.

Laptop Windows (`DESKTOP-xxxx`), Android 10+ (`android-<hex>` hoặc không gửi hostname) và iPhone
(chỉ "iPhone") không chứa đời máy; catalog không giúp được các trường hợp này.
