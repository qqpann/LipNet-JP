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
