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
youtube-dl -F https://www.youtube.com/watch?v=cLgEcNPr-ZE
youtube-dl -f 22 -o '%(id)s.%(ext)s' https://www.youtube.com/watch?v=cLgEcNPr-ZE
ffmpeg -i cLgEcNPr-ZE.mp4 -c:v copy cLgEcNPr-ZE.mp4.avi
```

```
docker cp cLgEcNPr-ZE.mp4.avi determined_buck:/home/openface-build
```

```
build/bin/FeatureExtraction -simsize 200 -f cLgEcNPr-ZE.mp4.avi
build/bin/FeatureExtraction -simsize 200 -fdir processed/cLgEcNPr-ZE.mp4_aligned/ -out_dir processed2
```

## Prepare audio

```
ffmpeg -y -i sample2.mp4 -c:a flac audio2.flac
```