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

# DoTo List
- 各体の部位の長さは一定である
- 両手首の距離は一定である
現在、クラブのグリップの位置は正しく取得できていると仮定したうえで、各手首とグリップの位置の距離をもとに、外れ値を検出している。
その後、外れ値と検出したフレームの前のフレームの値をもとに、線形補間を行っている。
ただし、補完を行っているのは手首のみであり、肘の位置が修正していないので、変な動きになっている。
また、連続して外れ値と検出された場合に補完の位置の精度が落ちる。
部位の長さの情報、スイング動作の情報などを用いて、より正確に補完を行う必要がある。