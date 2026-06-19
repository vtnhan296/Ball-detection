# SoccerNet

Project nay tap trung vao cac bai toan thi giac may tinh cho bong da tren du lieu SoccerNet:
phat hien cau thu/thu mon/trong tai, suy luan vai tro va doi bong, theo doi bong, va tao ban do
nhin tu tren xuong (BEV minimap) tu annotation GameState.

## Tong Quan Pipeline

Repo duoc chia thanh 4 nhanh cong viec chinh:

1. Tai va chuan bi du lieu SoccerNet tracking.
2. Huan luyen YOLO cho phat hien nguoi choi va/hoac bong.
3. Chay inference YOLO + PRTReID de gan `role`, `team_id`, `side_label` va xuat video MP4 co annotation.
4. Render BEV minimap tu annotation GameState bang homography.

Duong chay inference quan trong nhat la:

```text
thu muc anh -> YOLO detect -> crop nguoi choi -> PRTReID embedding/role
           -> gan role/team -> ve bbox + label -> MP4 + JSON
```

## Cau Truc Thu Muc

```text
SoccerNet/
+-- README.md
+-- requirements.txt
+-- bev/
|   +-- bev_minimap.py
+-- data/
|   +-- raw/
|   |   +-- tracking/
|   +-- SoccerNetGS/
|   +-- yolo/
|       +-- ball/
|       +-- fullframe/
+-- experiments/
|   +-- hota_yolo_bytetrack.ipynb
|   +-- runs/
+-- models/
|   +-- ball-tracking.pt
|   +-- pretrained/
|   +-- reid/
+-- notebooks/
|   +-- 00_download_soccernet_tracking.ipynb
|   +-- 01_train_player_yolo_detector.ipynb
|   +-- 02_infer_player_role_team.ipynb
|   +-- 03_bev_minimap_yolo_tracking_team.ipynb
|   +-- 04_infer_ball_yolo_h264.ipynb
|   +-- 05_infer_ball_detr_h264.ipynb
|   +-- balldetection-yolo.ipynb
|   +-- balldetection-detr.ipynb
+-- outputs/
|   +-- ball_tracking/
|   +-- bev/
|   +-- detect_player/
+-- src/
    +-- detect_player/
```

## Thanh Phan Chinh

### `notebooks/`

Chua cac workflow de chay tung giai doan:

- `00_download_soccernet_tracking.ipynb`: tai/giai nen du lieu SoccerNet tracking vao `data/raw/`.
- `01_train_player_yolo_detector.ipynb`: chuyen annotation MOT sang YOLO, huan luyen/validate detector cau thu, co ho tro SAHI va ByteTrack.
- `02_infer_player_role_team.ipynb`: chay pipeline YOLO + PRTReID tren folder anh va xuat video MP4.
- `03_bev_minimap_yolo_tracking_team.ipynb`: tao BEV/minimap ket hop tracking va team.
- `04_infer_ball_yolo_h264.ipynb`: inference/tracking bong bang YOLO va xuat H.264.
- `05_infer_ball_detr_h264.ipynb`: inference/tracking bong bang DETR va xuat H.264.
- `balldetection-yolo.ipynb`, `balldetection-detr.ipynb`: notebook thu nghiem/phat trien cho bai toan phat hien bong.

### `src/detect_player/`

Package Python noi bo cho pipeline detect player + role/team inference.

- `config.py`: tim project root, chon device mac dinh (`cuda:0` neu co GPU), va khai bao `PlayerTeamClassifierConfig`.
- `player_role_team_classifier.py`: facade end-to-end. File nay load YOLO, goi PRTReID, gan role/team, ve anh va luu JSON.
- `inference_runner.py`: doc anh tu folder, sap xep theo ten file, chay inference tung frame va ghi MP4/JSON.
- `reid_backend.py`: quan ly PRTReID, tai/checkpoint ReID va HRNet, fallback sang `models/reid/prtreid_src` neu can.
- `team_assignment.py`: logic gan role va team. Neu YOLO co class `player/goalkeeper/referee` thi uu tien role cua YOLO; neu khong thi dung role score tu PRTReID. Team duoc gan bang KMeans tren embedding hoac fallback theo vi tri.
- `results.py`: dataclass `DetectionResult` va helper serialize sang JSON.
- `visualization.py`: ve bbox, label role/confidence va mau theo team.
- `cli.py`: entry point dong lenh cho pipeline folder anh sang MP4.

### `bev/`

- `bev_minimap.py`: script render minimap nhin tu tren xuong tu `Labels-GameState.json`. Script uoc luong hoac doc homography, project toa do tu anh sang mat san, ve player/referee/ball, va xuat video original, BEV, side-by-side cung cac CSV debug neu bat.

### `data/`

- `data/raw/tracking/`: du lieu SoccerNet tracking goc theo format MOT.
- `data/yolo/fullframe/`: dataset YOLO full-frame va `data.yaml`.
- `data/yolo/ball/`: dataset YOLO cho bong.
- `data/SoccerNetGS/`: du lieu GameState/annotation dung cho BEV va ball tracking.

Thu muc `data/` co the rat lon, nen thuong chi luu local artifact can thiet cho training/inference.

### `models/`

- `models/pretrained/`: checkpoint YOLO pretrained, vi du `yolov8n.pt`, `yolo26n.pt`.
- `models/ball-tracking.pt`: model YOLO da train cho tracking bong.
- `models/reid/`: checkpoint PRTReID va HRNet:
  - `prtreid-soccernet-baseline.pth.tar`
  - `hrnetv2_w32_imagenet_pretrained.pth`
  - `prtreid_src/` neu can clone source PRTReID tren Windows.

### `outputs/`

Noi luu ket qua sinh ra trong qua trinh train/inference:

- `outputs/detect_player/runs/`: Ultralytics training/validation runs.
- `outputs/detect_player/infer/`: video/JSON inference cua detect-player.
- `outputs/ball_tracking/`: output tracking bong bang YOLO/DETR.
- `outputs/bev/`: output BEV minimap.

### `experiments/`

Chua notebook va artifact danh gia thu nghiem, vi du `hota_yolo_bytetrack.ipynb` va cac ket qua TrackEval/HOTA trong `experiments/runs/`.

## Cai Dat

Tao moi truong Python, sau do cai dependencies:

```bash
pip install -r requirements.txt
```

Mot so dependency quan trong:

- `ultralytics`: YOLO training/inference.
- `torch`: backend deep learning.
- `opencv-python`: doc/ghi anh va video.
- `scikit-learn`: KMeans de gan team.
- `yacs`, `albumentations`, `torchmetrics`, `monai`: phu thuoc cho PRTReID.

Tren Windows, PRTReID co the kho cai truc tiep neu thieu Microsoft C++ Build Tools. Cach fallback cua project la clone source PRTReID vao:

```text
models/reid/prtreid_src/
```

Khi folder nay ton tai, `src/detect_player/reid_backend.py` se tu dong them vao `sys.path`.

## Chay Inference Folder Anh Sang MP4

Vi du dung Python API:

```python
from detect_player import PlayerRoleTeamClassifier

clf = PlayerRoleTeamClassifier.from_project_defaults()
summary = clf.predict_folder(
    "data/yolo/fullframe/images/val",
    output_video_path="outputs/detect_player/infer/team_classifier/val_preview.mp4",
    output_json_path="outputs/detect_player/infer/team_classifier/val_preview.json",
    max_images=20,
    output_fps=25.0,
)
print(summary)
```

Hoac dung CLI:

```bash
python -m detect_player.cli data/yolo/fullframe/images/val \
  --output-video outputs/detect_player/infer/team_classifier/val_preview.mp4 \
  --output-json outputs/detect_player/infer/team_classifier/val_preview.json \
  --max-images 20 \
  --fps 25
```

Mac dinh config se tim model YOLO tai:

```text
outputs/detect_player/runs/E1_yolo_fullframe_img960/weights/best.pt
```

Neu model nam o vi tri khac, truyen override khi khoi tao:

```python
clf = PlayerRoleTeamClassifier.from_project_defaults(
    yolo_weights="path/to/best.pt",
    device="cuda:0",
    yolo_conf=0.30,
)
```

## Schema Ket Qua JSON

Moi detection duoc serialize voi cac truong chinh:

- `image_idx`: index anh/frame trong batch.
- `bbox_xyxy`: bounding box `[x1, y1, x2, y2]`.
- `detection_confidence`: confidence tu YOLO.
- `yolo_class_id`, `yolo_class_name`: class cua YOLO.
- `role`: vai tro cuoi cung, vi du `player`, `goalkeeper`, `referee`.
- `role_source`: nguon gan role, `yolo` hoac `prtreid`.
- `role_confidence`: confidence cua role.
- `reid_role`, `reid_role_confidence`: role du doan boi PRTReID.
- `team_id`: `0`, `1` hoac `null`.
- `side_label`: `left`, `right` hoac `null`.

Embedding ReID duoc giu trong bo nho va khong ghi ra JSON mac dinh de file nhe hon.

## Chay BEV Minimap

Script BEV lam viec voi annotation SoccerNet GameState:

```bash
python bev/bev_minimap.py \
  --labels data/SoccerNetGS/valid/<sequence>/Labels-GameState.json \
  --output outputs/bev/<sequence>_bev.mp4 \
  --debug \
  --show-track-ids
```

Script co the xuat:

- video BEV/homography.
- video source goc co bbox.
- video side-by-side.
- CSV homography/debug/velocity neu bat cac flag tuong ung.

## Ghi Chu Van Hanh

- Cac notebook la noi tot nhat de tai du lieu, train model va thu nghiem.
- Package `src/detect_player` la phan nen nen dung lai khi muon inference tu code hoac CLI.
- Cac artifact lon nhu dataset, checkpoint va output video nam trong `data/`, `models/`, `outputs/`.
- Khi chay tu notebook, neu import `detect_player` loi, hay dam bao `src/` da nam trong `PYTHONPATH` hoac chay notebook tu root project.
