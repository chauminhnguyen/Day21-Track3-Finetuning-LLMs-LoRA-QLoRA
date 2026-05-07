# Lab 21 — Evaluation Report
**Học viên**: Nguyễn Minh Châu — 2A202600179
**Ngày nộp**: 2026-07-05
**Submission option**: B

---

## 1. Setup

| Mục | Chi tiết |
|-----|---------|
| **Base model** | `unsloth/Qwen2.5-3B-bnb-4bit` (3B params, pre-quantized NF4) |
| **Dataset** | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval) |
| **max_seq_length** | 1024 (p95 ≈ 512, rounded up & capped tại 1024 theo T4 profile) |
| **GPU** | Tesla T4, 15 GB VRAM |
| **Training cost** | ~$0.04 (≈ 11.6 min tổng @ $0.35/hr) |
| **Target modules** | `q_proj`, `v_proj` |
| **Epochs** | 3, cosine LR, lr = 2e-4, effective batch = 8 |
| **HF Hub link** | crylake/qwen2.5-3b-vi-lab21-r16 |

---

## 2. Rank Experiment Results

### 2.1 Bảng so sánh 4 chiều

| Rank | Alpha | Trainable Params | % of Total | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|-----------------|-----------|------------|-----------|-----------|------------|
| **8** | 16 | 1,843,200 | 0.0623% | 3.86 min | 7.22 GB | 1.5574 | **4.747** |
| **16** ⬅ baseline | 32 | 3,686,400 | 0.1245% | 3.79 min | 6.62 GB | 1.5161 | **4.554** |
| **64** | 128 | 14,745,600 | 0.4982% | 3.93 min | 8.00 GB | 1.4766 | **4.378** |
| Base (no FT) | — | — | — | — | — | — | ~100+ |

### 2.2 Nhận xét nhanh về từng chiều

- **Thời gian train**: Ba rank có thời gian train rất gần nhau (~3.8–3.9 min), cho thấy với dataset nhỏ (200 samples), bottleneck không nằm ở số lượng params mà ở data throughput và overhead I/O.
- **VRAM**: r=16 thực sự dùng ít VRAM nhất (6.62 GB), trong khi r=64 tốn nhiều nhất (8.00 GB). Đây là điểm bất ngờ — r=8 (7.22 GB) dùng nhiều hơn r=16 có thể do cách Unsloth cấp phát buffer nội bộ.
- **Perplexity**: Giảm dần khi rank tăng: 4.747 → 4.554 → 4.378, nhưng mức cải thiện có dấu hiệu diminishing returns (xem phân tích Section 5).

---

## 3. Loss Curve Analysis

### 3.1 Biểu đồ

![Loss Curve r=16](loss_curve.png)

*Loss Curve — Baseline r=16, 3 epochs, 200 samples Vietnamese Alpaca*

### 3.2 Quan sát

**Không có overfitting đáng kể.** Cụ thể:

- Train loss giảm liên tục từ ~1.62 (step 5) xuống ~1.39 (step 70), tổng mức giảm ~14%.
- Đường loss không phẳng đơn điệu mà dao động nhẹ (đặc biệt bump ở step 35–40), phản ánh effective batch size nhỏ (=8) và noisy gradient khi training trên 180 samples.
- Eval loss (dấu chấm đỏ ở step 70) = **1.516**, gần với train loss cuối (~1.39). Gap ~0.12 là hợp lý cho dataset 180 samples — không đủ nhỏ để kết luận overfit, không đủ lớn để lo ngại.
- Bump tại step 35–40 nhiều khả năng do cosine LR scheduler còn cao ở giữa epoch 2, kết hợp với một vài training samples có sequence length dài hơn trung bình.

**Kết luận**: Model đang học tốt trong phạm vi 3 epochs với dataset nhỏ này. Nếu muốn kiểm tra overfitting kỹ hơn cần bật `eval_strategy="steps"` và có eval set lớn hơn 20 samples.

---

## 4. Qualitative Comparison — 5 Examples

> **Phương pháp chọn mẫu**: Chọn đại diện đa dạng task: giải thích khái niệm, viết code, danh sách, so sánh kỹ thuật, phân loại phương pháp. Bao gồm cả case win lẫn case gần như tương đương.

---

### Example 1 — Giải thích khái niệm

| | Nội dung |
|---|---|
| **Prompt** | Giải thích khái niệm machine learning cho người mới bắt đầu. |
| **Base** | "Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc phân loại mới. Tính toán này không dựa trên các quy tắc cứng nhắc, mà thay vào đó, nó sử dụng các thuật toán và các thuật toán để học tập, họ..." _(bị cắt, lặp "thuật toán")_ |
| **Fine-tuned (r=16)** | "Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu và từ đó đưa ra các dự đoán..." |
| **Nhận xét** | ✅ **Improved** — Fine-tuned version bỏ được lỗi lặp từ ("thuật toán và các thuật toán"), câu văn mạch lạc hơn. Cả hai đều thiếu ví dụ cụ thể, nhưng FT ổn định hơn về mặt ngữ pháp. |

---

### Example 2 — Viết code

| | Nội dung |
|---|---|
| **Prompt** | Viết đoạn code Python tính số Fibonacci thứ n. |
| **Base** | Dùng đệ quy, định nghĩa `fib(1)=0, fib(2)=1`, trả về string `"N phải là một số dương"` khi `n<=0`. |
| **Fine-tuned (r=16)** | Dùng vòng lặp iterative (`a, b = 0, 1; for _ in range...`), raise `ValueError` khi input âm, convention chuẩn hơn (`fib(0)=0, fib(1)=1`). |
| **Nhận xét** | ✅ **Improved** — FT chọn iterative thay vì đệ quy (tránh stack overflow với n lớn), xử lý edge case đúng hơn (`n<0` raise ValueError thay vì return string). Cho thấy model đã học được coding best practice từ dataset. |

---

### Example 3 — Danh sách / Liệt kê

| | Nội dung |
|---|---|
| **Prompt** | Liệt kê 5 nguyên tắc thiết kế UI/UX. |
| **Base** | Liệt kê 5 nguyên tắc có giải thích dài, ví dụ "Thân thiện với người dùng", "Trực quan", v.v. — ngôn ngữ tự nhiên, có context. |
| **Fine-tuned (r=16)** | Liệt kê ngắn gọn hơn: "1. Chuyển đổi, 2. Thích ứng, 3. Đơn giản, 4. Tương thích, ..." — thiếu giải thích cho từng điểm. |
| **Nhận xét** | ⚠️ **Mixed** — Base model cho output đầy đủ nội dung hơn. FT model bị ảnh hưởng bởi style ngắn gọn của Alpaca format, đưa ra danh sách quá tóm tắt. Đây là case FT không cải thiện chất lượng nội dung, chỉ thay đổi style. |

---

### Example 4 — So sánh kỹ thuật AI

| | Nội dung |
|---|---|
| **Prompt** | Tóm tắt sự khác biệt giữa LoRA và QLoRA. |
| **Base** | Giải thích đúng hướng nhưng nhầm tên: "NLU (NLP)" không liên quan, mô tả LoRA như "phép biến đổi nhỏ hơn" — thiếu chính xác. |
| **Fine-tuned (r=16)** | Giải thích sai nghiêm trọng: gọi LoRA là "Layer-wise Adaptive Regularization Optimization" — tên hoàn toàn sai (LoRA = Low-Rank Adaptation). Mô tả như kỹ thuật regularization thay vì parameter-efficient fine-tuning. |
| **Nhận xét** | ❌ **Degraded** — Đây là case quan trọng để ghi nhận: FT model đưa ra hallucination tự tin hơn và sai hơn base model. Với domain-specific knowledge, fine-tuning trên dataset chung (Vietnamese Alpaca GPT-4) không đảm bảo factual accuracy về các khái niệm kỹ thuật mới. |

---

### Example 5 — Phân loại phương pháp

| | Nội dung |
|---|---|
| **Prompt** | Phân biệt prompt engineering, RAG, và fine-tuning. |
| **Base** | Giải thích đúng hướng: prompt engineering = cách viết câu lệnh, RAG = retrieval-based, fine-tuning = training thêm — có phân biệt rõ. |
| **Fine-tuned (r=16)** | Giải thích prompt engineering khá đúng ("xây dựng câu lệnh để giúp hệ thống AI"), phần RAG và fine-tuning bị cắt trong 300 ký tự nhưng bắt đầu ổn. |
| **Nhận xét** | ✅ **Similar / Slightly improved** — Cả hai đều nắm hướng đúng. FT version câu văn mạch lạc hơn và ít vòng vo. Không có sự khác biệt lớn trong case này. |

---

## 5. Conclusion về Rank Trade-off

### Rank nào cho ROI tốt nhất trên dataset này?

Dựa trên kết quả thực nghiệm, **r=16 cho ROI tốt nhất** trên bài toán này. Lý do:

Khi tăng từ r=8 lên r=16, số trainable params tăng gấp đôi (1.84M → 3.69M, +100%), nhưng perplexity cải thiện 4.05% (4.747 → 4.554). Đây là mức cải thiện xứng đáng với chi phí tính toán gần như không đổi (~3.86 min vs 3.79 min). Ngược lại, khi tăng từ r=16 lên r=64, params tăng gấp 4 lần (3.69M → 14.75M, +300%), nhưng perplexity chỉ cải thiện thêm 3.87% (4.554 → 4.378) — tức là đầu tư params lớn hơn 4x nhưng chỉ thu về mức cải thiện tương đương lần tăng trước.

Quan trọng hơn, với dataset chỉ 200 samples (180 train), tăng rank cao không có nhiều ý nghĩa: mô hình không đủ data để khai thác capacity của r=64. Điều này thể hiện qua việc perplexity của r=64 (4.378) vẫn còn cao — không phải do rank thấp mà do data ít. Trong bối cảnh này, r=16 là điểm cân bằng lý tưởng giữa expressiveness và regularization tự nhiên từ việc rank bị giới hạn.

### Khi nào diminishing returns xuất hiện?

Dấu hiệu diminishing returns xuất hiện rõ ở **bước nhảy r=16 → r=64**. Cụ thể: mỗi triệu params thêm vào mang lại cải thiện perplexity ít hơn. Từ r=8 → r=16 (+1.84M params), perplexity giảm 0.192 đơn vị, tức ~0.104 perplexity point per million params. Từ r=16 → r=64 (+11.06M params), perplexity chỉ giảm 0.176 đơn vị, tức ~0.016 perplexity point per million params — hiệu suất per-param giảm 85%.

Về mặt lý thuyết LoRA, điều này nhất quán với bản chất low-rank decomposition: với một task nhất định, tồn tại một "intrinsic rank" mà sau đó tăng r thêm chỉ thêm noise, không thêm signal có ý nghĩa. Với Vietnamese instruction following trên 200 samples, intrinsic rank có vẻ nằm ở khoảng 16–32, và r=64 đã vượt qua ngưỡng này.

### Recommendation cho production

Nếu deploy production, tôi chọn **r=16** vì:

1. **Stability**: r=8 có thể thiếu capacity với prompt phức tạp; r=64 tốn VRAM và thời gian hơn khi fine-tune lại trên data mới.
2. **Generalization**: Rank cao hơn trên ít data hơn có nguy cơ overfit patterns cụ thể của training set (dù trong 3 epochs chưa thấy rõ).
3. **Operational cost**: VRAM r=16 (6.62 GB) thấp nhất trong 3 config, dễ serve hơn trên hardware phổ thông.
4. **Extensibility**: r=16 dễ scale — nếu sau này có thêm data, adapter có thể được tiếp tục huấn luyện mà không cần thay đổi config.

---

## 6. What I Learned

- **LoRA rank không phải cứ cao là tốt.** Tôi ban đầu giả định r=64 sẽ cải thiện đáng kể hơn, nhưng thực tế chỉ +3.87% perplexity với chi phí params gấp 4 lần. Dataset size và task complexity mới là yếu tố quyết định, không phải rank.

- **Hallucination sau fine-tuning vẫn là rủi ro thực.** Example 4 (LoRA vs QLoRA) cho thấy FT model đưa ra câu trả lời sai tự tin hơn base model. Fine-tuning trên general instruction dataset không inject factual knowledge — nó chỉ thay đổi style và format output. Muốn model biết LoRA là gì, cần fine-tune trên data có nội dung đó.

- **VRAM không tăng tuyến tính với rank.** r=16 dùng ít VRAM hơn cả r=8 (6.62 vs 7.22 GB), điều này nhắc nhở rằng memory profiling thực tế luôn cần thiết — intuition đơn giản "rank cao = tốn VRAM nhiều hơn" không phải lúc nào cũng đúng tuyệt đối do các yếu tố như gradient checkpointing, buffer allocation, và optimizer state.