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

## Prepare data

```
youtube-dl -F https://www.youtube.com/watch?v=cLgEcNPr-ZE
youtube-dl -f 22 -o '%(id)s.%(ext)s' https://www.youtube.com/watch?v=cLgEcNPr-ZE
ffmpeg -i cLgEcNPr-ZE.mp4 -c:v copy cLgEcNPr-ZE.mp4.avi
```

```
docker cp cLgEcNPr-ZE.mp4.avi determined_buck:/home/openface-build/input 
```

```
ffmpeg -y -i sample2.mp4 -c:a flac audio2.flac
```