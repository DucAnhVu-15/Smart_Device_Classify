# Rule hostname từ danh sách thiết bị

## Phạm vi

Rule A (regex, first-match-wins) của cascade hostname được sinh từ **một file danh sách thiết bị**
`pipeline/device_list.csv`. File gộp thiết bị của Data_AKien (`all_data_1604.csv`) và danh sách
thiết bị hiện nay (`device-list.md`). Không đổi feature/train của fingerprint, OS hay expert B.

| File | Vai trò |
|---|---|
| `pipeline/device_list.csv` | File chung, sửa tay: mỗi dòng một thiết bị |
| `pipeline/hostname_catalog.py` | Sinh regex, sắp thứ tự, đối chiếu nhãn thật, giữ rule đạt |
| `05_train.ipynb` phần C | `HN_DEVICE_CATALOG` trỏ tới file CSV; sinh rule ở C2 mỗi lần chạy |
| `Models/<RUN>/train/hostname_catalog*.{csv,json}` | Bản sao CSV, rule đã sinh và report của run đó |

## Cột

| Cột | Ý nghĩa |
|---|---|
| `make`, `type`, `model` | Nhãn trả ra; để trống head nào thì dòng không chốt head đó |
| `hostname_names` | Tên người dùng gõ trong hostname, cách nhau `\|`: `s24 ultra\|s24u`. Để trống thì tự sinh từ model |
| `source` | Ghi chú nguồn (`AKien`, `device-list`), không dùng khi train |

- Dấu cách giữa chữ và số tùy ý: `s24 ultra`, `s24ultra`, `s 24 ultra` sinh cùng regex, cho phép
  `-_. ` hoặc dính liền. Cần ranh giới hai đầu: `s24` khớp `s24-cua-an`, `an-s-s24`, `galaxy-s24`,
  không khớp `s240`.
- Tên viết kèm hãng (`vivo y27`, `galaxy a12`) chỉ khớp khi hostname có hãng; tên trần (`a12`)
  khớp cả có lẫn không có hãng. Tên trần ngắn dễ trùng hãng khác thì ghi kèm hãng.
- Tự sinh: bỏ tên hãng và `Galaxy` ở đầu model, thêm bản bỏ `5G/4G` (`Samsung Galaxy A15 5G` →
  `a15 5g|a15`). Không sinh khi chỉ còn từ chung (`ZTE Router`) hoặc trùng tên của dòng khác nhãn.
- `re:<regex>` (đứng cuối ô) là regex viết tay, cú pháp RE2: máy in `re:br[nw][0-9a-f]{12}`.
- `model = __unknown__` là rule chặn: hostname khớp thì head đó dừng ở A, không gọi expert, kết quả
  gói tin giữ nguyên. Camera `nomi-ipc-s21fe` dùng cách này để không thành Galaxy S21 FE.
- Dùng đúng chính tả nhãn đã có (`Samsung Galaxy A51`, `Samsung Galaxy A15 5G`), nếu không sẽ thành lớp mới;
  05 in `Nhãn viết khác dataset` khi phát hiện. AKIEN viết một máy thành hai nhãn (`Samsung A51` /
  `Samsung Galaxy A51`, `Galaxy S10` / `Galaxy S10 5G`); `MODEL_ALIAS` trong `hostname_data.py` gộp
  về tên chuẩn khi nạp dữ liệu, và danh sách báo lỗi nếu một dòng dùng tên alias. Type chỉ dùng 8 type hiện có; gaming/XR/máy in để trống.

## Thứ tự và kiểm tra

Mỗi tên là một rule riêng. Rule A phải chạy trước rule B khác nhãn nếu B khớp hostname mẫu của A
(S24 Ultra trước S24); script sắp tô-pô theo quan hệ này. Hai thiết bị khác nhãn ghi cùng một tên
thì báo lỗi, chỉ ra dòng cần sửa. Hòa thì regex viết tay đi trước tên.

Mỗi (tên, head) được chấm trên `train + validation` của split hostname trong 05, đúng thứ tự cascade:
rule mặc định 09 trước, danh sách sau. Test chỉ để báo cáo.

- `correct`; `compatible` khi nhãn thật chung hơn cùng hãng (`Google Pixel` với `Google Pixel 7`);
  `Samsung A51` = `Samsung Galaxy A51`. Bảng `hostname_comparison.csv` của 05 cũng có cột
  `compatible` và `selective_accuracy_compatible` theo cùng cách chấm này; gate vẫn dùng khớp chính xác.
- C_cascade của 05 chạy như release khi gói tin chưa biết make: model suy ra make còn trống
  (`source = model_implies`), model expert trái make bị bỏ, make trái make của model rule bị bỏ.
- `wrong` cả khi head đang chấm chưa có nhãn nhưng head khác mâu thuẫn (host là IP Camera).
- Loại khi ≥ 5 mẫu chấm và precision < 0,95, hoặc < 5 mẫu mà sai nhiều hơn đúng. Make/type bị loại
  thì loại cả tên đó (`rejected_row`). Tên khác cùng dòng không bị ảnh hưởng.
- `unverified`: chưa có mẫu, thường là máy mới (S26, Fold7). Vẫn được dùng.

## Cập nhật danh sách thiết bị

1. Sửa/thêm dòng trong `Code/pipeline/device_list.csv` (Excel lưu dạng CSV UTF-8).
2. (Tùy chọn) xem trước rule nào bị loại, khoảng 1,5 phút:
   ```powershell
   python -X utf8 Code/pipeline/hostname_catalog.py --split Models/<RUN>/train/hostname_dataset_split.csv
   ```
   rồi đọc `Code/pipeline/hostname_catalog_report.csv` (cột `line` = `<dòng CSV>:<tên>`, `status`,
   `wrong_examples`).
3. Chạy phần C của 05 → ghi `RUN` mới vào 09 và 12 (13 nếu cần TFLite) → chạy lần lượt.

Laptop Windows (`DESKTOP-xxxx`), Android 10+ (`android-<hex>` hoặc không gửi hostname) và iPhone
(chỉ "iPhone") không chứa đời máy; danh sách không giúp được các trường hợp này.

## Lịch sử

- 13/10/2026, run `20261013_094051`: catalog 784 dòng (9 cột) → 881 rule. Model: 126 verified,
  625 unverified, 26 bị loại; test giữ riêng model 0,9987, make 0,9986, type 0,9998.
- 14/10/2026: chuyển sang `device_list.csv` (849 thiết bị = 143 chỉ AKien + 578 chỉ device-list.md
  + 128 cả hai), dựng lại từ rule trong contract `20261013_094051` cộng nhãn AKien. Trên split của
  run đó: 964 rule; rule A trả lời model cho 18.740 hostname test (trước 17.554), precision 0,9984.
