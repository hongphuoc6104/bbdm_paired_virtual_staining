# BBDM Paired Virtual Staining — Colab Training

Train mô hình **BBDM** cho bài toán **virtual H&E staining** trên Google Colab.

| | |
|---|---|
| **Input** | Ảnh unstained (grayscale), 512×512 |
| **Target** | Ảnh H&E stained (RGB), 512×512 |
| **Dataset** | Tải tự động từ [HuggingFace](https://huggingface.co/datasets/mezeidragos-lateral/bach-breast-histopathology-he-staining-patches-512x512), chia 80% train / 20% test |
| **Checkpoint** | Lưu trực tiếp vào Google Drive mỗi 1000 bước (không mất khi Colab ngắt) |

> Dựa trên paper: [Super-resolved Virtual Staining of Label-free Tissue Using Diffusion Models](https://arxiv.org/pdf/2410.20073)

---

## 🚀 Bắt Đầu

1. **Mở notebook trên Colab**:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hongphuoc6104/bbdm_paired_virtual_staining/blob/main/colab_train_bbdm_paired_png_virtual_staining.ipynb)

2. **Bật GPU**: `Runtime → Change runtime type → GPU → Save`

3. **Chạy từng cell từ trên xuống** — notebook tự động tải code, dataset, chia dữ liệu, và bắt đầu train.

**Không cần upload file nào lên Drive.**

---

## 📁 Cấu Trúc Repo

```
├── colab_train_bbdm_paired_png_virtual_staining.ipynb  ← Notebook chính
├── train.py                    ← Script train (hỗ trợ --val_lr/hr_data_dir)
├── improved_diffusion/         ← Thư viện core BBDM
│   ├── unet.py                 ← UNet + ConditionEncoder
│   ├── gaussian_diffusion.py   ← BBDM diffusion
│   ├── train_util.py           ← Training loop + checkpoint
│   ├── image_datasets.py       ← Data loader
│   └── ...
├── COLAB_GUIDE.md              ← Hướng dẫn chi tiết từng bước
├── analysis_metrics.py         ← Tính PSNR, SSIM, LPIPS
├── setup.py
└── environment.yml             ← Conda env (cho chạy local)
```

---

## 📖 Hướng Dẫn Chi Tiết

Xem [COLAB_GUIDE.md](COLAB_GUIDE.md) — giải thích từng cell, kết quả mong đợi, cách xử lý lỗi, và cách lưu/resume checkpoint.

---

## 🖥️ Chạy Local

```bash
conda env create -f environment.yml && conda activate bbdm && pip install -e .

python train.py \
  --lr_data_dir ./train/input  --hr_data_dir ./train/target \
  --val_lr_data_dir ./test/input  --val_hr_data_dir ./test/target \
  --batch_size 1 --large_size 256 --small_size 256 \
  --num_channels 128 --num_res_blocks 2 \
  --diffusion_steps 1000 --lr_anneal_steps 5000 \
  --save_interval 1000 --val_interval 500
```
