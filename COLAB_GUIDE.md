# Huong Dan Chay Train Tren Google Colab

## Cach Nhanh Nhat: Upload 1 Notebook Standalone

Neu ban khong muon dung GitHub va cung khong muon upload ca thu muc code len Drive, hay dung file:

```text
colab_standalone_bbdm_png.ipynb
```

File nay da nhung san code project ben trong. Cach dung:

```text
1. Vao https://colab.research.google.com
2. Chon File -> Upload notebook
3. Upload colab_standalone_bbdm_png.ipynb
4. Runtime -> Change runtime type -> GPU
5. Chay tung cell tu tren xuong
6. Den cell Provide Dataset Zip, upload file test-20260515T193153Z-3-001.zip hoac chon path trong Drive
```

Voi cach nay, ban chi can upload notebook standalone vao Colab. Dataset zip van can duoc cung cap vi file du lieu qua lon de nhung vao notebook.

---

## Cach Day Du: Project Nam Tren Google Drive

Huong dan nay gia dinh ban chi dung Google Colab va Google Drive.

## 1. Chuan Bi Tren Google Drive

Tao mot thu muc trong Google Drive:

```text
MyDrive/
  Super-resolved-virtual-staining/
```

Dat toan bo code project da sua vao thu muc nay. Thu muc can co cac file quan trong:

```text
Super-resolved-virtual-staining/
  colab_train_bbdm_png.ipynb
  train.py.py
  improved_diffusion/
    image_datasets.py
    unet.py
    dist_util.py
    ...
  test-20260515T193153Z-3-001.zip
```

File du lieu zip nen dat tai:

```text
MyDrive/Super-resolved-virtual-staining/test-20260515T193153Z-3-001.zip
```

Notebook mac dinh se doc dung duong dan nay.

## 2. Mo Notebook Tren Colab

Vao Google Colab:

```text
https://colab.research.google.com
```

Chon:

```text
File -> Open notebook -> Google Drive
```

Mo file:

```text
MyDrive/Super-resolved-virtual-staining/colab_train_bbdm_png.ipynb
```

## 3. Bat GPU

Trong Colab chon:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

Sau do bam:

```text
Save
```

Cell dau tien `!nvidia-smi` phai hien GPU. Neu bao loi hoac khong co GPU, can kiem tra lai Runtime type.

## 4. Chay Notebook Theo Thu Tu

Chay tung cell tu tren xuong duoi.

Thu tu chinh:

```text
1. Check GPU
2. Mount Google Drive
3. Config path
4. Install dependencies
5. Copy project tu Drive sang /content
6. Extract dataset zip vao /content/data
7. Inspect dataset
8. Smoke test loader va ConditionEncoder
9. Optional diffusion loss check
10. Smoke train
11. Visualize validation output
12. Longer train neu smoke train thanh cong
13. Save models/logs ve Drive
```

Cell install dependencies se cai OpenMPI truoc de `mpi4py` chay duoc tren Colab:

```python
!apt-get -qq update
!apt-get -qq install -y libopenmpi-dev openmpi-bin
%pip -q install blobfile mpi4py pytorch-wavelets lpips scikit-image opencv-python tqdm
```

## 5. Config Mac Dinh

Notebook mac dinh dung:

```python
PROJECT_DRIVE_DIR = Path('/content/drive/MyDrive/Super-resolved-virtual-staining')
DATASET_ZIP = PROJECT_DRIVE_DIR / 'test-20260515T193153Z-3-001.zip'
```

Neu ban dat project hoac dataset o vi tri khac, sua hai dong tren trong cell `Configuration`.

## 6. Kiem Tra Dataset

Cell inspect dataset can in ra dang gan nhu:

```text
input_count: 1200
target_count: 1200
paired_count: 1200
only_input: 0
only_target: 0
00000.png input L (512, 512) | target RGB (512, 512)
```

Neu `paired_count = 0`, thuong la do duong dan sai hoac zip giai nen ra cau truc khac.

## 7. Smoke Test Dung Mong Doi

Cell smoke test loader va encoder can hien:

```text
target: (2, 3, 256, 256)
cond: (2, 1, 256, 256)
encoded: (2, 3, 256, 256)
```

Neu shape khac, khong nen train tiep ma can sua loader/model truoc.

## 8. Smoke Train

Smoke train chi chay 20 step de kiem tra pipeline.

Luu y:

```text
Smoke train khong tao ket qua dep.
Muc tieu chi la kiem tra code train co chay duoc khong.
```

Sau smoke train, notebook se hien thi:

```text
Condition | Prediction | Target
```

Prediction luc nay co the nhieu noise vi moi train rat it step.

## 9. Train Lau Hon

Trong cell `Longer Train`, doi:

```python
RUN_LONG_TRAIN = False
```

thanh:

```python
RUN_LONG_TRAIN = True
```

Co the bat dau voi:

```python
LONG_STEPS = 5000
```

Neu Colab bi het thoi gian, giam xuong:

```python
LONG_STEPS = 1000
```

## 10. Noi Luu Output

Notebook se copy model va log ve:

```text
MyDrive/bbdm_outputs/
```

Ben trong co the co:

```text
bbdm_smoke_models/
bbdm_smoke_log/
bbdm_long_models/
bbdm_long_log/
```

## 11. Loi Thuong Gap

### Khong Tim Thay Project

Loi:

```text
Project path not found
```

Sua `PROJECT_DRIVE_DIR` trong cell Configuration.

### Khong Tim Thay Dataset Zip

Loi:

```text
Dataset zip not found
```

Kiem tra file zip co dung ten:

```text
test-20260515T193153Z-3-001.zip
```

va nam trong thu muc project tren Drive.

### Het VRAM

Neu gap CUDA out of memory:

```text
Giu batch_size = 1
Giam num_channels tu 128 xuong 64
Giam num_res_blocks tu 2 xuong 1
Giam LONG_STEPS de test truoc
```

### Smoke Train Cham

Diffusion train cham hon U-Net/GAN. Nen chay smoke train truoc de dam bao pipeline dung, roi moi train dai.

## 12. Lenh Train Chinh Neu Muon Chay Rieng

Sau khi notebook da copy project va giai nen data, co the chay lenh:

```bash
python train.py.py \
  --lr_data_dir /content/data/test/input \
  --hr_data_dir /content/data/test/target \
  --batch_size 1 \
  --large_size 256 \
  --small_size 256 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --diffusion_steps 1000 \
  --lr_anneal_steps 5000 \
  --log_interval 10 \
  --val_interval 500 \
  --save_interval 1000
```
