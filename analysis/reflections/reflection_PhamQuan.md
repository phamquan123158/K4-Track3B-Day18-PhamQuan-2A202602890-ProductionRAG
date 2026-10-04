# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Phạm Quân  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 2026-10-04

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 giúp nhóm các câu có cùng chủ đề lại với nhau, giảm tình trạng cắt giữa ý so với `chunk_basic()`. Trong thực tế, chunking theo ngữ nghĩa phù hợp hơn khi tài liệu có nhiều section và điều kiện, ví dụ quy định nghỉ phép, chính sách bảo mật và lương. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF kết hợp điểm xếp hạng từ lexical search (BM25) và semantic search (dense/vector), giúp tìm được cả văn bản khớp từ khóa chính xác lẫn câu mang nghĩa tương đồng. Đây là kỹ thuật quan trọng khi tài liệu mang nhiều từ khóa dài, số và điều kiện thời gian. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Cross-encoder rerank lại top-k candidate từ bộ lọc ban đầu để ưu tiên tài liệu đúng chủ đề nhất. Trong lab, lợi ích rõ nhất là tăng độ chính xác ở top 3 kết quả và giảm nhiễu khi query tương tự nhiều loại nội dung. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Pipeline được đánh giá qua 4 chỉ số: Faithfulness, Answer Relevancy, Context Precision và Context Recall. Tôi nhận thấy các metric này phản ánh đúng các điểm yếu chính của RAG: độ trung thành câu trả lời, sự phù hợp với câu hỏi, độ tinh lọc context và mức độ phủ sóng thông tin. |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Việc thêm ngữ cảnh trước chunk giúp cải thiện retrieval bởi vì hệ thống không chỉ nhìn vào nội dung chunk mà còn hiểu chunk đó thuộc phần nào của tài liệu. This reduces retrieval failure on policy-heavy or long-form docs, especially in internal knowledge bases. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `ModuleNotFoundError: No module named 'dotenv'`
- **Nguyên nhân gốc rễ & Cách debug:**
  - Lỗi xảy ra khi import `config.py` trước khi môi trường Python chưa cài đầy đủ dependencies. Trong trường hợp này, module `python-dotenv` chưa được install, nên `from dotenv import load_dotenv` bị lỗi ngay khi import. Tôi đã debug bằng cách chạy lại `pytest` và xác định stack trace, sau đó cài đặt đầy đủ dependency từ `requirements.txt` bằng: `python -m pip install -r requirements.txt`.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Ban đầu, tôi chưa nhận ra rằng project này cần dependency ổn định và cảnh báo offline-safe trong môi trường grading. Sau đó, tôi hiểu rằng production RAG cần có fallback khi thiếu API key/model hoặc khi tải model từ HuggingFace không khả dụng. Vì vậy, tôi bổ sung logic graceful fallback trong M2–M5 để hệ thống vẫn chạy được dù không có OpenAI key hay model được cache sẵn. Đây là kỹ năng quan trọng để tăng độ ổn định của ứng dụng trong production và CI.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Production RAG cho tài liệu nội bộ công ty

#### 1. Hiện trạng
- **Pipeline hiện tại:** Một hệ thống RAG cơ bản gồm load docs → chunking → hybrid search → rerank → answer generation → evaluate metrics.
- **Vấn đề / Bottlenecks đang gặp:** Các vấn đề chính là độ chính xác retrieval khi các câu hỏi có từ khóa thay đổi, độ trùng lặp giữa nhiều policy version, và nguy cơ hallucination khi không có context đủ rõ.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Chọn `hierarchical + semantic` để giữ cả ngữ cảnh lớn và độ tập trung của chunk. Với tài liệu chính sách nội bộ, việc chia theo section và parent-child giúp giữ được mối quan hệ giữa quy định và chi tiết. 
2. **Search retrieval:** Dùng `Hybrid Search` (BM25 + Dense + RRF) vì nó kết hợp tốt cả keyword match và semantic match. Đây là lựa chọn hợp lý cho dữ liệu tiếng Việt có nhiều cụm từ và số liệu. 
3. **Reranking:** Bật `CrossEncoderReranker` nếu model có sẵn, vì reranking ở top-K giúp lọc thông tin nhiễu. Khi offline hoặc model chưa cache, dùng fallback heuristic để tránh block pipeline. 
4. **Evaluation:** Dùng `RAGAS 4 metrics` sebagai baseline chuẩn, đồng thời bổ sung custom check cho câu hỏi có nhiều version policy / câu hỏi cần logic số liệu. 
5. **Enrichment:** Áp dụng `contextual_prepend` và `summary` trước khi index để giảm retrieval failure và nâng chất lượng câu trả lời. Nếu cần thêm, dùng `HyQA` để tạo câu hỏi giả định từ chunk.

#### 3. Timeline triển khai
- **Tuần 1:** Hoàn thiện chunking và retrieval baseline; kiểm tra recall trên các câu hỏi quan trọng. 
- **Tuần 2:** Bổ sung reranker và enrichment; tối ưu lại context quality cho các policy/FAQ nội bộ. 
- **Tuần 3:** Chạy evaluation RAGAS và phân tích lỗi; tinh chỉnh prompt/chunking cho các case khó. 
- **Tuần 4:** Chuẩn hóa deploy pipeline, thêm logging và đánh giá latency trước khi dùng trong production.

---

### Kết luận

Lab Production RAG giúp tôi thấy rõ rằng thành công của một hệ thống RAG không nằm ở việc “đưa model lớn vào” mà nằm ở việc tối ưu trực tiếp từng tầng: chunking, retrieval, reranking, evaluation và enrichment. Từ đó, tôi hiểu rằng production RAG cần vừa mạnh về chất lượng, vừa ổn định về kỹ thuật và khả năng chạy trong môi trường giới hạn tài nguyên.
