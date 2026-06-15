# Hướng Dẫn Train BBDM Virtual Staining trên Google Colab

---

## 1. Mở Notebook

Vào [Google Colab](https://colab.research.google.com), chọn:

```
File → Open notebook → GitHub → dán link:
https://github.com/hongphuoc6104/bbdm_paired_virtual_staining
```

Chọn file `colab_train_bbdm_paired_png_virtual_staining.ipynb`.

Hoặc bấm nút **Open in Colab** trong README trên GitHub.

---

## 2. Bật GPU

```
Runtime → Change runtime type → GPU → Save
```

Chạy cell đầu tiên `!nvidia-smi` — phải hiện thông tin GPU.

---

## 3. Chạy Từng Cell Từ Trên Xuống

| # | Cell | Mô tả |
|---|---|---|
| 1 | Check GPU | Kiểm tra GPU |
| 2 | Cấu hình | Đặt tham số (sửa nếu cần) |
| 3 | Install | Cài thư viện |
| 4 | Clone GitHub | Tải code về Colab |
| 5 | Tải HuggingFace & chia train/test | Tải dataset, chia 80/20 |
| 6 | Kiểm tra dataset | Xác nhận ảnh đã tải đúng |
| 7 | Smoke test | Kiểm tra loader & encoder |
| 8 | Smoke train (20 bước) | Kiểm tra pipeline chạy được |
| 9 | Hiển thị kết quả | Xem ảnh validation |
| 10 | Long train | Train thật sự (bật thủ công) |

---

## 4. Cấu Hình

Các tham số chính trong cell **Cấu Hình**:

```python
MAX_SAMPLES   = None    # None = dùng toàn bộ; đặt 200 để chạy nhanh
TRAIN_RATIO   = 0.8     # 80% train, 20% test
SAVE_TO_DRIVE = True    # Lưu checkpoint vào Drive (khuyến nghị)
CROP_SIZE     = 256
BATCH_SIZE    = 1
```

---

## 5. Kết Quả Mong Đợi

### Sau tải dataset:

```
📊 Chia: 4800 train / 1200 test
✅ Train: 4800 | Test: 1200
```

### Sau kiểm tra dataset:

```
--- TRAIN --- input: 4800 | target: 4800 | paired: 4800
--- TEST --- input: 1200 | target: 1200 | paired: 1200
  00000.png | input L (512, 512) | target RGB (512, 512)
```

### Sau smoke test:

```
target: (2, 3, 256, 256) | cond: (2, 1, 256, 256)
encoded: (2, 3, 256, 256)
✅ OK!
```

---

## 6. Long Train

Chỉ chạy **sau khi smoke train thành công**.

Đổi trong cell Long Train:

```python
RUN_LONG_TRAIN = True   # bật train
LONG_STEPS     = 5000   # số bước (giảm xuống 1000 nếu Colab hết giờ)
```

### Lưu checkpoint vào Drive

Khi `SAVE_TO_DRIVE = True` (mặc định), checkpoint lưu **trực tiếp vào Drive** mỗi 1000 bước:

```
MyDrive/bbdm_outputs/bbdm_runs/
  bbdm_20260516_030000/
    models/
      model001000.pt            ← trọng số bước 1000
      model001000_compress.pt   ← trọng số encoder
      model002000.pt            ← bước 2000
      ...
    log/
      0_LR.png, 0_SR.png, 0_HR.png   ← ảnh validation
```

Nếu Colab mất kết nối, checkpoint gần nhất vẫn còn trên Drive.

> ⚠️ Nếu `SAVE_TO_DRIVE = False`, checkpoint chỉ ở Colab runtime — mất kết nối = mất toàn bộ.

---

## 7. Lỗi Thường Gặp

| Lỗi | Cách sửa |
|---|---|
| `CUDA is not available` | Runtime → Change runtime type → GPU |
| `ConnectionError` khi tải HF | Chạy lại cell download |
| `CUDA out of memory` | Giữ `BATCH_SIZE = 1`; giảm `num_channels` xuống 64 |
| Smoke train chậm | Bình thường, diffusion cần 5–10 phút cho 20 bước |

---

## 8. Lệnh Train Thủ Công (Nếu Cần)

Sau khi notebook đã clone project và tải dataset:

```bash
python train.py \
  --lr_data_dir /content/data/train/input \
  --hr_data_dir /content/data/train/target \
  --val_lr_data_dir /content/data/test/input \
  --val_hr_data_dir /content/data/test/target \
  --batch_size 1 \
  --large_size 256 --small_size 256 \
  --num_channels 128 --num_res_blocks 2 \
  --diffusion_steps 1000 --lr_anneal_steps 5000 \
  --log_interval 10 --val_interval 500 --save_interval 1000
```
