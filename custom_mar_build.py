import argparse
from datetime import datetime
import whisper
from pathlib import Path
import json
import shutil



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="whisper_base")
    parser.add_argument("--model-type", type=str, default="base.en")
    parser.add_argument("--model-dir", type=str, default="model_pt")
    parser.add_argument("--handler", type=str, default="handler.py")
    args = parser.parse_args()
    return args

def copy_extra_files(extra_files, tmp_dir):
    """Copy extra files to tmp_dir. work with both folder and file"""
    for file in extra_files:
        if Path(file).is_dir():
            shutil.copytree(file, tmp_dir.joinpath(file))
        else:
            shutil.copy(file, tmp_dir.joinpath(file))

def main():
    args = parse()
    model_name = args.model_name
    model_type = args.model_type
    model_dir = args.model_dir
    handler = args.handler

    extra_files = ["whisper", model_dir, handler]
    tmp_dir = Path("./dummy_temp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    _ = whisper.load_model(model_type, download_root=model_dir)

    copy_extra_files(extra_files, tmp_dir)

    manifest = {
        "createdOn": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "runtime": "python",
        "model": {
            "modelName": model_name,
            "serializedFile": "",
            "modelType": model_type,
            "modelDir": model_dir,
            "handler": "handler.py",
            "modelVersion": "1.0"
        },
        "archiverVersion": "egochao_custom"
    }
    manifest_folder = tmp_dir.joinpath("MAR-INF")
    manifest_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_folder.joinpath("MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    zipfile = shutil.make_archive(f"./model_store/{model_name}", 'zip', tmp_dir)
    zippath = Path(zipfile)
    marpath = zippath.rename(zippath.with_suffix(".mar"))
    print(f"Created MAR file: {marpath}")


if __name__ == '__main__':
    main()