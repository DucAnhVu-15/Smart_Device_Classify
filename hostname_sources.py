"""Nguồn hostname ngoài DHCP option 12, cho cột `dhcp_hostname`.

Cột `dhcp_hostname` mang tên DHCP nhưng tầng L0 chỉ regex trên chuỗi — `add_l0_columns()`
nạp nó thẳng từ frame, KHÔNG gác theo `has_dhcp`. Nên đổ tên lấy từ giao thức khác vào
cột đó không đổi contract 44 cột, không phải train lại, không phải build lại ONNX.

Lý do cần: DHCP chỉ xuất hiện lúc join hoặc lúc renew (T1 = 50% lease). Một cửa sổ thu
thập vài phút thường không có gói DHCP nào, và khi đó luật hostname — đường L0 phủ rộng
nhất — không bao giờ nổ. mDNS và NetBIOS thì quảng bá liên tục.

Thứ tự ưu tiên khi gộp: opt12 > mDNS > NetBIOS. opt12 là thứ thiết bị khai với server nên
nó thắng khi có mặt; hai nguồn kia chỉ lấp chỗ trống.

⚠️ Đừng bật `has_dhcp` khi tên đến từ mDNS/NetBIOS. Cờ đó nuôi bảng ngưỡng theo số nguồn
bằng chứng và nuôi cổng `fp_usable()` của L1 — bịa cờ là làm hỏng hai chỗ khác để cứu một
chỗ.

Stdlib thuần, không numpy/pandas: file này vừa chạy được trong pipeline Python vừa là bản
đặc tả để port sang agent C trên router.
"""

import json
import re

MISSING = "<missing>"

# Service mà TÊN INSTANCE theo quy ước chính là hostname của máy. Danh sách đóng, có chủ ý:
# `_googlecast`, `_companion-link`, `_airplay` đặt tên instance là tên người dùng tự đặt
# hoặc chuỗi ngẫu nhiên, lấy vào chỉ tổ sinh rác cho regex L0.
HOSTNAME_SERVICES = frozenset({
    "_workstation", "_dosvc", "_smb", "_device-info",
    "_ssh", "_sftp-ssh", "_udisks-ssh", "_rfb",
})

# `raspberrypi [e4:5f:01:a3:c6:0d]._workstation._tcp.local` -> `raspberrypi`
_WORKSTATION_MAC = re.compile(r"\s*\[[0-9a-f]{2}(?::[0-9a-f]{2}){5}\]\s*$", re.I)

# Nhãn hợp lệ của một hostname. Cố tình chặt: cái gì không giống tên máy thì bỏ, vì rác
# lọt vào đây sẽ được regex L0 trả lời ở confidence 1.0 — không tầng nào phía sau chặn được.
_PLAUSIBLE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$", re.I)

# Tên NetBIOS mã hoá first-level: mỗi byte -> 2 ký tự 'A'..'P'. 32 ký tự + nhãn scope.
_NB_ENCODED = re.compile(r"^[A-P]{32}$")


def _plausible(name):
    """Lọc rác. Trả tên đã strip, hoặc None."""
    if not name:
        return None
    name = str(name).strip().strip(".").strip()
    if not name or not _PLAUSIBLE.match(name):
        return None
    # `localhost`, `wpad`, `isatap` là tên hạ tầng, không phải tên thiết bị.
    if name.lower() in ("localhost", "wpad", "isatap", "lb", "local"):
        return None
    return name


def from_mdns_qname(qname):
    """Hostname suy từ một qname mDNS. Trả None nếu qname không mang tên máy.

    Hai dạng nhận:
      `raspberrypi.local`                     -> raspberrypi   (A record, tên máy đúng nghĩa)
      `desktop-ducanh._dosvc._tcp.local`      -> desktop-ducanh (instance của service whitelist)

    Dạng bỏ:
      `_googlecast._tcp.local`                -> None (truy vấn service, không có instance)
      `lb._dns-sd._udp.local`                 -> None (meta-query)
      `i1rcvkn8n14aaa._fc9f5ed42c8a._tcp.local` -> None (service không nằm trong whitelist)
    """
    if not qname:
        return None
    labels = str(qname).strip().strip(".").split(".")
    if len(labels) < 2 or labels[-1].lower() != "local":
        return None
    head = labels[0]
    if head.startswith("_"):
        return None

    if len(labels) == 2:                      # <host>.local
        return _plausible(head)

    service = labels[1].lower()               # <instance>._<service>._tcp.local
    if service not in HOSTNAME_SERVICES:
        return None
    return _plausible(_WORKSTATION_MAC.sub("", head))


def from_mdns_mi_connect(qname):
    """Xiaomi `_mi-connect._udp`: tên instance là một JSON, `nm` là tên thiết bị.

    `{"nm":"redmi note 10","as":"[8194]","ip":"102"}._mi-connect._udp.local`
        -> `redmi note 10`

    Tách riêng khỏi `from_mdns_qname` vì nó trả TÊN SẢN PHẨM chứ không phải hostname —
    khớp luật `(?i)(redmi|xiaomi|poco)[-_ ].*` nhưng không phải thứ người dùng đặt.
    """
    if not qname or "_mi-connect" not in str(qname):
        return None
    head = str(qname).split("._mi-connect", 1)[0]
    try:
        name = json.loads(head).get("nm")
    except (ValueError, AttributeError):
        return None
    if not name:
        return None
    name = str(name).strip()
    # `redmi note 10` có dấu cách -> _PLAUSIBLE từ chối; luật L0 lại cần đúng dạng đó.
    return name if re.match(r"^[\w][\w .-]{1,62}$", name) else None


def decode_netbios_name(encoded):
    """Giải mã tên NetBIOS first-level encoding (RFC 1001 §4.1).

    Mỗi byte gốc tách hai nibble, mỗi nibble cộng `ord('A')`:

        'E' 'B' 'F' 'C' ...  ->  (4<<4)|1 = 0x41 = 'A' ...

    Byte thứ 16 là suffix loại tên (0x00 workstation, 0x20 file server), bỏ đi.
    Trả tên đã strip, hoặc None nếu không đúng dạng.
    """
    if not encoded or not _NB_ENCODED.match(str(encoded)):
        return None
    raw = bytearray()
    text = str(encoded)
    for i in range(0, 32, 2):
        hi, lo = ord(text[i]) - 65, ord(text[i + 1]) - 65
        raw.append((hi << 4) | lo)
    name = bytes(raw[:15]).decode("ascii", "ignore").strip()
    return _plausible(name)


def from_nbns_question(qname):
    """Hostname từ một question NBNS (UDP 137). Nhận cả dạng đã mã hoá lẫn đã giải.

    Agent bắt gói thô sẽ đưa vào chuỗi 32 ký tự; bên nào đã giải sẵn thì đưa tên thường.
    """
    if not qname:
        return None
    text = str(qname).strip().strip(".")
    if _NB_ENCODED.match(text):
        return decode_netbios_name(text)
    return _plausible(text)


def mine_hostname(records):
    """Gộp mọi bản ghi của MỘT MAC trong MỘT cửa sổ -> một hostname, hoặc MISSING.

    `records` là danh sách dict theo contract raw-input: `{"proto": ..., ...}`.
    Ưu tiên opt12 > mDNS > NetBIOS; trong cùng một nguồn thì lấy tên gặp nhiều nhất, hoà
    thì lấy tên xuất hiện trước — để cùng một cửa sổ luôn cho cùng một kết quả.
    """
    found = {"dhcp": [], "mdns": [], "nbns": []}
    for record in records or ():
        proto = str(record.get("proto", "")).lower()
        if proto == "dhcp":
            name = _plausible(record.get("opt12") or record.get("hostname"))
            if name:
                found["dhcp"].append(name)
        elif proto == "mdns":
            qname = record.get("qname")
            name = from_mdns_qname(qname) or from_mdns_mi_connect(qname)
            if name:
                found["mdns"].append(name)
        elif proto in ("nbns", "netbios"):
            name = from_nbns_question(record.get("qname") or record.get("name"))
            if name:
                found["nbns"].append(name)

    for source in ("dhcp", "mdns", "nbns"):
        names = found[source]
        if names:
            return max(dict.fromkeys(names), key=names.count)
    return MISSING
