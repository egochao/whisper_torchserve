# Whisper production deyloyment with torchserve


## 1. [Development environment](https://code.visualstudio.com/docs/remote/containers) - Devcontainer

Dependency: Docker

Start the working environment by:

- Open command pallete - Crl Shift P

- Open in Container command

## 2. Manual build step

1. Build mar file for torchserve(download included)
```
python custom_mar_build.py
```

2. Run torchserve 
```
torchserve --start --model-store model_store --models whisper_base.mar --foreground --no-config-snapshots --ts-config config.properties
```


## 3. Serve from container
Or you can start docker and test it right away
```
docker build -t whisper .

docker run --network=host whisper:latest
```

Test out the local endpoint
```
python send_request.py 
```

## 4. Note
Enable profiling - run in same terminal as torchserve
```
export ENABLE_TORCH_PROFILER=true
```
