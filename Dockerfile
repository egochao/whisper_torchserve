FROM mwader/static-ffmpeg:5.1.2 AS ffmpeg_image

FROM pytorch/torchserve-nightly:cpu-2022.10.18

ARG MODELTYPE=base.en
ARG MODELNAME=whisper_base

COPY --from=ffmpeg_image /ffmpeg /usr/local/bin/
# COPY --from=ffmpeg_image /ffprobe /usr/local/bin/

COPY --chown=model-server . .

RUN pip install ffmpeg-python==0.2.0 && \
    python custom_mar_build.py --model-type $MODELTYPE --model-name $MODELNAME

EXPOSE 8080 8081 8082

CMD ["torchserve", "--start", "--model-store", "model-store", "--ncs"]