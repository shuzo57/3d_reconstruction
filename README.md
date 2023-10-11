# 3d reconstruction for golf swing

Download Models:
```Bash
chmod +x download_models.sh
./download_models.sh
```

Download Sample Video:
```Bash
chmod +x download_sample_videos.sh
./download_sample_videos.sh
```

Sample Usage:
```Bash
python3 -m swing3dify -f1 /home/ohwada/golf/2023_09_29/yokota_hanako/d1/1/uratani_0929_007_1.mp4 -f2 /home/ohwada/golf/2023_09_29/yokota_hanako/d1/1/view1_0923_4317_1.mp4 -o 
. -p models/pose-l.pt -c models/club-yokota.pt
```
