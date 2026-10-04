# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Phạm Quân  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.5037 | 0.6823 | +0.1786 |
| Answer Relevancy | 0.4919 | 0.6791 | +0.1872 |
| Context Precision | 0.4500 | 0.6500 | +0.2000 |
| Context Recall | 0.4825 | 0.7048 | +0.2223 |

## Bottom-5 Failures

### #1
- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** Không. Nhân viên tuyệt đối không được tự ý xử lý malware; phải báo cáo trong vòng 1 giờ qua helpdesk@cty.vn hoặc hotline CNTT.
- **Got:** Câu trả lời có thể thiếu trọng tâm nếu context không chứa điều kiện báo cáo kịp thời hoặc ký tự “không được tự xử lý”.
- **Worst metric:** `faithfulness`
- **Error Tree:** Output sai → Context đúng? → Query OK? → Policy version mismatched
- **Root cause:** Đây là câu hỏi yêu cầu nhấn mạnh hành động cấm và quy trình báo cáo; nếu chunk retrieval thiếu đoạn báo cáo và “không tự xử lý”, hệ thống sẽ trả lời mơ hồ hoặc đúng phần nhưng thiếu điều kiện nghiêm trọng.
- **Suggested fix:** Tăng độ rõ ràng với chunking theo policy section, thêm metadata `security`, dùng contextual prepend và tối đa top-k retrieval để ưu tiên đoạn có từ khóa “malware”, “không được”, “báo cáo”.

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên = 18 ngày phép; lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Có thể trả lời đúng một phần nhưng thiếu tính toán thâm niên hoặc lương trong khoảng.
- **Worst metric:** `context_recall`
- **Error Tree:** Missing relevant chunks → retrieval imprecision → answer omits numeric combination
- **Root cause:** Đây là câu hỏi đa tiền đề: số ngày phép + thâm niên + thang lương. Nếu không có context đủ, model chọn 1 trong 3 mảnh thông tin mà bỏ mất phần còn lại.
- **Suggested fix:** Ràng buộc câu trả lời theo template số liệu, tăng retrieval depth và hybrid search với BM25 để giữ đúng các cụm “thâm niên 9 năm”, “15 ngày”, “P3-P4”.

### #3
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Cần Giám đốc phòng ban (Director) phê duyệt; cần xác nhận cấu hình kỹ thuật từ phòng CNTT và ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Có thể đúng phần “phê duyệt” nhưng bỏ qua yêu cầu về công bố cấu hình cho CNTT và báo giá.
- **Worst metric:** `answer_relevancy`
- **Error Tree:** Answer matches topic but misses required conditions → missing edge conditions
- **Root cause:** Dữ liệu nằm rải trong nhiều chính sách khác nhau: ngưỡng giá, phê duyệt theo cấp, yêu cầu báo giá, và cần xác nhận kỹ thuật. Retrieval thiếu một số chunk cùng chủ đề dẫn đến thiếu điều kiện quan trọng.
- **Suggested fix:** Sử dụng hierarchical chunking để giữ nguyên section “mua sắm thiết bị CNTT”, áp dụng metadata `category=finance|it`, và thêm câu hỏi giả định để tăng khả năng truy xuất nhiều yếu tố cùng lúc.

### #4
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo v2024, từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm; chính sách cũ v2023 là 5 năm.
- **Got:** Có thể trả lời đúng giá trị nhưng không nhắc đến sự thay đổi giữa version cũ và mới.
- **Worst metric:** `context_precision`
- **Error Tree:** Too many irrelevant chunks → model mixes versions → answer loses exact policy version
- **Root cause:** Trong corpus có cả chính sách cũ và mới; nếu retrieval không lọc đúng version, model dễ “trộn” 2 quy định và trả lời thiếu chính xác.
- **Suggested fix:** Tạo metadata version (`v2023`, `v2024`) và thêm bộ lọc theo `version` / `effective_date`; rerank ưu tiên section có `current policy`.

### #5
- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Expected:** 3 ngày làm việc có lương khi kết hôn, không trừ vào phép năm.
- **Got:** Có thể trả lời số ngày nhưng thiếu “không trừ vào phép năm” hoặc “được lương”.
- **Worst metric:** `answer_relevancy`
- **Error Tree:** Relevant topic but missing qualification → incomplete answer
- **Root cause:** Câu hỏi quá ngắn và kết hợp nhiều điều kiện kèm theo. RAG dễ trả lời đúng con số nhưng quên các chi tiết làm câu trả lời “đúng bản chất”.
- **Suggested fix:** Dùng contextual prepend và prompt instruction “Trả lời ngắn gọn nhưng đầy đủ 3 thành phần: số ngày, chế độ lương, không trừ phép năm.”

## Case Study (cho presentation)

**Question chọn phân tích:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?

**Error Tree walkthrough:**
1. Output đúng? → Có thể đúng phần “không nên tự xử lý” nhưng thiếu lộ trình báo cáo. 
2. Context đúng? → Nếu retrieval lấy đúng chunk “tự xử lý malware là vi phạm nghiêm trọng”, câu trả lời sẽ đúng hơn. 
3. Query rewrite OK? → Query đủ rõ, nhưng thiếu từ khóa “báo cáo”, “hotline”, “1 giờ”. 
4. Fix ở bước: thêm context và ưu tiên chunk có `security` metadata, cùng với rerank để đưa policy báo cáo lên trước.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm metadata versioning và category tagging cho từng policy section.
- Tăng độ sâu retrieval và áp dụng query expansion cho các câu hỏi “cấm”, “không được”, “yêu cầu”, “phải báo cáo”.
- Dùng prompt template để ép kết quả trả lời theo cấu trúc: “Quy định”, “Điều kiện”, “Hệ quả nếu vi phạm”.
