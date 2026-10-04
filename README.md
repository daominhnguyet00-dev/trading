# 📈 Hệ Thống Kiểm Định Chiến Lược Giao Dịch Định Lượng: Kết Hợp SMA & OBV (Streamlit Web App)

Ứng dụng web trực quan xây dựng trên nền tảng **Streamlit**, chuyển đổi toàn bộ quy trình kiểm định và tối ưu hóa tham số từ Notebook `SMA_+_OBV.ipynb` sang giao diện tương tác cao cấp. Hệ thống cho phép nhà đầu tư và chuyên viên phân tích định lượng (Quant Trader) kiểm định hiệu quả của sự kết hợp giữa **Xu hướng giá (SMA)** và **Động lượng dòng tiền (OBV)** trên dữ liệu lịch sử cổ phiếu **ACB** (giai đoạn 2014 – 2023) hoặc tải lên bất kỳ tập dữ liệu cổ phiếu nào khác.

---

## 🌟 Tính Năng Nổi Bật

1. **Kiểm định 4 chiến lược đồng thời**:
   - **SMA Crossover**: Giao cắt giữa đường trung bình động ngắn hạn (`ma_short`) và dài hạn (`ma_long`).
   - **OBV Crossover**: Tín hiệu khối lượng cân bằng dòng tiền OBV cắt đường trung bình động của nó (`obv_window`).
   - **SMA + OBV (AND)**: Tín hiệu đồng thuận nghiêm ngặt — chỉ MUA khi cả SMA và OBV cùng báo MUA; chỉ BÁN khi cả hai cùng báo BÁN.
   - **SMA + OBV (OR)**: Tín hiệu linh hoạt — MUA khi một trong hai chỉ báo báo MUA; BÁN khi một trong hai báo BÁN.
2. **Quy trình chuẩn mực Train / Test Split (Chống Data Snooping Bias)**:
   - **Tập Train (2014 – 2019)**: Dùng để khám phá mẫu hình và tìm kiếm tham số tối ưu.
   - **Tập Test (2020 – 2023)**: Dữ liệu mẫu ngoài (Out-of-sample) để kiểm định tính bền vững thực tế.
3. **Tích hợp Tối ưu hóa Tham số Tự động (Bayesian Optimization via Hyperopt)**:
   - Tìm kiếm bộ tham số tối ưu của SMA và OBV ngay trên ứng dụng web bằng thuật toán TPE (Tree-structured Parzen Estimator).
4. **Biểu đồ Tương tác Đa tầng (Interactive Plotly Charts)**:
   - Biểu đồ nến kỹ thuật (Candlestick) tích hợp vị trí các điểm vào/ra lệnh mua bán (Buy/Sell Markers).
   - Biểu đồ đường cong tăng trưởng vốn (Equity Curves) đối sánh cùng lúc cả 4 chiến lược với chuẩn thị trường (Buy & Hold).
   - Biểu đồ mức sụt giảm tài khoản (Underwater Drawdown) đo lường rủi ro thời gian thực.
5. **Nhật ký Giao dịch Chi Tiết (Trade Log)**:
   - Hiển thị ngày mở/đóng vị thế, giá thực hiện, tỷ suất sinh lời từng lệnh, và hỗ trợ tải về định dạng `.csv`.
6. **Linh hoạt dữ liệu đầu vào**:
   - Tự động nạp sẵn dữ liệu cổ phiếu ACB (`ACB.csv`) hoặc hỗ trợ tải lên file CSV bất kỳ có đủ các cột: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

---

## 📐 Cơ Sở Lý Thuyết Chiến Lược

### 1. Đường Trung Bình Động Giản Đơn (SMA)
- **Điểm MUA (Golden Cross)**: $SMA_{short} > SMA_{long}$ và $SMA_{short}(t-1) \le SMA_{long}(t-1)$.
- **Điểm BÁN (Death Cross)**: $SMA_{short} < SMA_{long}$ và $SMA_{short}(t-1) \ge SMA_{long}(t-1)$.

### 2. Chỉ Báo Cân Bằng Khối Lượng (OBV)
- Đo lường áp lực tích lũy/phân phối qua khối lượng giao dịch.
- Tạo đường tín hiệu $OBV\_MA$ với chu kỳ `obv_window`:
  - **Điểm MUA**: $OBV > OBV\_MA$ và $OBV(t-1) \le OBV\_MA(t-1)$.
  - **Điểm BÁN**: $OBV < OBV\_MA$ và $OBV(t-1) \ge OBV\_MA(t-1)$.

### 3. Nguyên Tắc Ghép Tín Hiệu (Combination)
| Điều kiện | Chiến lược AND (Khắt khe) | Chiến lược OR (Linh hoạt) |
| :--- | :--- | :--- |
| **Tín hiệu MUA (BUY = 1)** | Cả SMA và OBV cùng phát tín hiệu MUA | SMA **hoặc** OBV phát tín hiệu MUA |
| **Tín hiệu BÁN (SELL = -1)** | Cả SMA và OBV cùng phát tín hiệu BÁN | SMA **hoặc** OBV phát tín hiệu BÁN |
| **Đặc điểm** | Lọc nhiễu cao, ít lệnh, tránh bẫy bull-trap | Bắt nhạy xu hướng sớm, giao dịch nhiều hơn |

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py               # Mã nguồn chính của ứng dụng Streamlit Dashboard
├── requirements.txt     # Danh sách các thư viện Python cần thiết
├── README.md            # Tài liệu hướng dẫn sử dụng và triển khai
├── ACB.csv              # Dữ liệu mẫu lịch sử giá ACB (2014 - 2023)
└── SMA_+_OBV.ipynb      # File Jupyter Notebook phân tích gốc
```

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### Bước 1: Yêu cầu môi trường
Đảm bảo máy tính đã cài đặt **Python 3.9, 3.10 hoặc 3.11**.

### Bước 2: Tạo môi trường ảo và cài đặt thư viện
Mở Terminal / PowerShell tại thư mục dự án và thực hiện:

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv

# Kích hoạt môi trường ảo:
# Trên Windows:
venv\Scripts\activate
# Trên macOS / Linux:
source venv/bin/activate

# Cài đặt các thư viện phụ thuộc:
pip install -r requirements.txt
```

### Bước 3: Khởi chạy ứng dụng Web
```bash
streamlit run app.py
```
Sau khi chạy lệnh, trình duyệt web sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Triển Khai (Deploy) Lên Streamlit Cloud

Để đưa ứng dụng lên mạng internet hoàn toàn miễn phí thông qua **Streamlit Community Cloud**:

### Bước 1: Đẩy mã nguồn lên GitHub
1. Khởi tạo Git repository trong thư mục chứa 3 file (`app.py`, `requirements.txt`, `README.md`) cùng file dữ liệu `ACB.csv`:
   ```bash
   git init
   git add app.py requirements.txt README.md ACB.csv
   git commit -m "Deploy SMA + OBV Quantitative Trading Web App"
   ```
2. Tạo một Repository mới trên [GitHub](https://github.com/new) (ví dụ đặt tên: `sma-obv-trading-dashboard`).
3. Đẩy mã nguồn lên GitHub:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-github-cua-ban>/sma-obv-trading-dashboard.git
   git push -u origin main
   ```

### Bước 2: Deploy trên Streamlit Community Cloud
1. Truy cập vào **[share.streamlit.io](https://share.streamlit.io)** và đăng nhập bằng tài khoản GitHub của bạn.
2. Bấm vào nút **"New app"**.
3. Điền thông tin ứng dụng:
   - **Repository**: `<tai-khoan-github-cua-ban>/sma-obv-trading-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Bấm **"Deploy!"**.
5. Đợi 1-2 phút để hệ thống tự động cài đặt thư viện từ `requirements.txt`. Khi hoàn tất, bạn sẽ nhận được đường link web trực tuyến để chia sẻ cho bạn bè, giảng viên hoặc đồng nghiệp!

---

## 📊 Hướng Dẫn Sử Dụng Giao Diện

- **Thanh bên trái (Sidebar)**:
  - Chọn dữ liệu nguồn (Mẫu ACB có sẵn hoặc tải file `.csv` từ máy).
  - Chọn khoảng ngày cho tập **Train** và tập **Test**.
  - Tùy chỉnh vốn ban đầu, tỷ lệ phí giao dịch.
  - Điều chỉnh thanh trượt tham số `ma_short`, `ma_long`, `obv_window` hoặc bấm nút **"Mẫu Notebook"** để gán nhanh giá trị tối ưu.
  - Bấm **"🚀 Chạy Hyperopt trên Train"** để hệ thống tự động tìm kiếm tham số tối ưu bằng thuật toán thông minh.
- **Khu vực trung tâm**:
  - **Tab 1 (Bảng So Sánh Hiệu Quả)**: Đánh giá Return, Sharpe Ratio, Max Drawdown, Win Rate giữa Train và Test.
  - **Tab 2 (Đường Vốn Tài Khoản)**: Trực quan hóa tốc độ tăng trưởng vốn giữa các chiến lược và biểu đồ sụt giảm tài khoản (Drawdown).
  - **Tab 3 (Phân Tích Kỹ Thuật)**: Soi từng phiên giao dịch với biểu đồ nến, đường SMA, OBV và các mũi tên tín hiệu MUA/BÁN.
  - **Tab 4 (Chi Tiết Giao Dịch)**: Xem toàn bộ lịch sử lệnh vào/ra và nút xuất file Excel/CSV.
  - **Tab 5 (Lý Thuyết & Quy Trình)**: Tóm tắt công thức toán học và phương pháp luận của chiến lược.

---

## ⚖️ Tuyên Bố Miễn Trừ Trách Nhiệm (Disclaimer)
Ứng dụng được xây dựng phục vụ mục đích nghiên cứu học thuật, giáo dục và kiểm định định lượng. Kết quả kiểm thử trong quá khứ không đảm bảo cho tỷ suất sinh lời trong tương lai. Người dùng cần tự chịu trách nhiệm đối với các quyết định đầu tư thực tế trên thị trường tài chính.
