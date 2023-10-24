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
python3 -m swing3dify -f1 videos/sample_video1.mp4 -f2 videos/sample_video2.mp4 -o . -p models/pose-l.pt -c models/club-v1.pt
```