FROM mwader/static-ffmpeg:5.1.2 AS ffmpeg_image

FROM pytorch/torchserve-nightly:cpu-2022.10.18

COPY --from=ffmpeg_image /ffmpeg /usr/local/bin/
COPY --from=ffmpeg_image /ffprobe /usr/local/bin/
