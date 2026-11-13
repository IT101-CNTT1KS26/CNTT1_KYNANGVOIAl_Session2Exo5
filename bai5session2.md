# PHÂN TÍCH CƠ CHẾ IN-CONTEXT LEARNING VÀ THIẾT KẾ PROMPT FEW-SHOT CHỐNG MỈA MAI — TECHSHOP



## PHẦN 1: PHÂN TÍCH CƠ CHẾ IN-CONTEXT LEARNING (HỌC TRONG NGỮ CẢNH)

Hiện tượng AI bị lừa bởi các từ khóa tích cực ở bề mặt (như *"đẹp mê ly"*, *"khỏi cần bật lò sưởi"*) trong câu lệnh Zero-shot xảy ra do Zero-shot chỉ đánh giá **nghĩa hiển ngôn (Surface Lexical Meaning)** bằng cách tra cứu xác suất từ vựng đơn thuần.

Để giải quyết vấn đề này mà **không cần can thiệp huấn luyện lại (fine-tuning) hay cập nhật trọng số (weights) của mô hình**, chúng ta ứng dụng kỹ thuật **In-Context Learning (ICL)** thông qua các ví dụ mẫu (Few-shot Prompting). Cơ chế hoạt động diễn ra như sau:

1. **Cơ chế dự đoán token tiếp theo (Next-Token Prediction):** Các Mô hình Ngôn ngữ Lớn (LLMs) hoạt động bằng cách dự đoán từ (token) tiếp theo dựa trên toàn bộ chuỗi ngữ cảnh được cung cấp trong cửa sổ ngữ cảnh (Context Window). Khi đưa các cặp ví dụ mẫu (Input ➔ Output) vào prompt, mô hình không hề "học" theo nghĩa lưu trữ dài hạn vào bộ nhớ của neural network, mà nó dùng chính cấu trúc đó làm **khuôn mẫu suy luận tức thời (in-context activation)**.
2. **Kích hoạt tri thức tiềm ẩn (Latent Knowledge Priming):** Bản thân LLM đã được huấn luyện với lượng dữ liệu khổng lồ trên Internet nên nó hoàn toàn có khả năng hiểu các sắc thái phức tạp như mỉa mai, châm biếm, khen đểu. Tuy nhiên, ở dạng Zero-shot, mô hình mặc định chọn cách hiểu đơn giản nhất. Khi cung cấp các ví dụ mẫu Few-shot, ta đang thiết lập một **"bản lề ngữ nghĩa" (Semantic Priming)**, buộc cơ chế tự chú ý (Attention Mechanism) của AI phải chuyển hướng sang việc phân tích tầng nghĩa ẩn ý bên dưới thay vì chỉ bám vào từ ngữ bề mặt.



## PHẦN 2: THIẾT KẾ PROMPT FEW-SHOT HOÀN CHỈNH CHO TECHSHOP

Dưới đây là một Prompt Few-shot hoàn chỉnh, tích hợp sẵn **3 cặp ví dụ mẫu** dạy AI cách bóc trần lớp vỏ bọc mỉa mai đối với 3 vấn đề cốt lõi tại TechShop: **Nhiệt độ CPU (quá nhiệt), Dung lượng Pin (hao nguồn) và Thời gian giao hàng (chậm trễ)**.

```text
[Context / Character]
Bạn là Trợ lý AI chuyên phân tích cảm xúc đánh giá khách hàng (Sentiment Analysis) tại hệ thống thương mại điện tử TechShop. Khách hàng của TechShop thường xuyên sử dụng văn phong mỉa mai, "khen đểu" (sarcasm) đối với các lỗi kỹ thuật nghiêm trọng. Bạn cần nhìn thấu nghĩa ẩn ý thay vì chỉ đánh giá dựa trên từ ngữ tích cực ở bề mặt.

[Examples / Few-shot Learning]
Dưới đây là các ví dụ mẫu phân tích cảm xúc thực tế tại TechShop:

---
Ví dụ 1 (Lỗi Nhiệt độ / Quá nhiệt):
- Input: "Máy thiết kế đẹp mê ly, cầm rất đầm tay, mùa đông ôm máy chơi game thì khỏi cần bật lò sưởi vì nhiệt độ CPU lên hẳn 98 độ C!"
- Output: {
    "sentiment": "TIÊU CỰC",
    "hidden_issue": "Quá nhiệt (Overheating)",
    "reason": "Sử dụng từ ngữ khen hình thức nhưng thực chất mỉa mai mức nhiệt độ CPU cực kỳ nguy hiểm (98 độ C)."
  }

Ví dụ 2 (Lỗi Pin / Hao nguồn):
- Input: "Pin trâu khủng khiếp thật sự, rút sạc 100% từ sáng đến trưa không đụng vào máy mà đã tự sập nguồn ngắt điện, quá tuyệt vời cho một thiết bị di động!"
- Output: {
    "sentiment": "TIÊU CỰC",
    "hidden_issue": "Lỗi hao pin / Sập nguồn bất thường",
    "reason": "Dùng từ 'pin trâu' và 'tuyệt vời' để mỉa mai việc máy hao nguồn nặng và sập nguồn khi không sử dụng."
  }

Ví dụ 3 (Lỗi Vận chuyển / Giao hàng):
- Input: "Shop giao hàng nhanh như vận tốc ánh sáng, đặt hàng từ năm ngoái mà năm nay mới nhận được hàng, dịch vụ quá đỉnh cao!"
- Output: {
    "sentiment": "TIÊU CỰC",
    "hidden_issue": "Giao hàng chậm trễ nghiêm trọng",
    "reason": "Dùng cụm từ phóng đại 'nhanh như vận tốc ánh sáng' đi kèm khoảng thời gian vô lý để mỉa mai sự chậm trễ của đơn vị vận chuyển."
  }
---

[Request]
Hãy áp dụng phương pháp phân tích sâu sắc từ các ví dụ mẫu trên để đánh giá cảm xúc của đánh giá khách hàng mới nhất dưới đây.

[Input Data cần xử lý]
"Màn hình hiển thị sắc nét thật đấy, chỉ tiếc là cứ mở ứng dụng lên là đơ máy, khởi động lại 5 lần vẫn đứng hình, đúng là hàng cao cấp có khác!"

[Type of Output & Format]
Kết quả trả về PHẢI ở dạng mã JSON thuần túy (được đặt trong khối ```json ... ```) tuân thủ chính xác cấu trúc như các ví dụ mẫu.

```