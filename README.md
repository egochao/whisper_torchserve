# Whisper production deyloyment with torchserve


## 1. [Development environment](https://code.visualstudio.com/docs/remote/containers) - Devcontainer

Dependency: Docker

Start the working environment by:

- Open command pallete - Crl Shift P

- Open in Container command

## 2. Build step

1. Build mar file for torchserve(download included)
```
python custom_mar_build.py
```

2. Run torchserve 
```
torchserve --start --model-store   model_store --models whisper_base.mar --foreground --no-config-snapshots
```
