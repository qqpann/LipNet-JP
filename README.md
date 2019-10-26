# LipNet-JP

## Directory

```
src/
```

`.py` files here.

```
notebook/
```

`.ipynb` files here.

## Prepare video

```
youtube-dl -F https://www.youtube.com/watch?v={youtube_id}
youtube-dl -f 22 -o 'data/raw_input/%(id)s.%(ext)s' https://www.youtube.com/watch?v={youtube_id}
ffmpeg -i {youtube_id}.mp4 -c:v copy {youtube_id}.avi
```

これでdockerを立ち上げて，中に入る．psでid確認してcpなどする．
```
docker run -it --rm algebr/openface:latest
```


```
docker cp {youtube_id}.avi {docker_id}:/home/openface-build
```

Docker の中
```
build/bin/FeatureExtraction -simsize 200 -f {youtube_id}.avi
build/bin/FeatureExtraction -simsize 200 -fdir processed/{youtube_id}_aligned/ -out_dir processed2
```

Dockerの外
```
docker cp {docker_id}:/home/openface-build/processed data/.
docker cp {docker_id}:/home/openface-build/processed2 data/.

python src/crop_lip.py {youtube_id}
```

## Prepare audio

```
ffmpeg -y -i {youtube_id}.mp4 -c:a flac audio2.flac
```
sample2を適宜ファイル名に読み替える