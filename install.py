import launch
import os
import sys
#tensorboardX == 2.6.2.2
#onnxruntime == 1.16.3
req = [
    ["parselmouth","praat-parselmouth>=0.4.2"],
    ["torchcrepe","torchcrepe==0.0.15"],
    ["pyworld","pyworld>=0.3.2"],
    ["faiss","faiss-cpu==1.7.3"],
    ["ffmpeg","ffmpeg-python>=0.2.0"],
    ["torchaudio","torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118"],
    ["fairseq","fairseq==0.12.2"],
    ["soundfile","soundfile>=0.12.1"],
    ["gtts","gtts==2.3.2"],
    ["pydub","pydub>=0.25.1"],
    ["numpy","numpy>=1.23.5"],
    ["scipy","scipy>=1.9.3"],
    ["tqdm","tqdm>=4.63.1"],
    ["tensorboardX","tensorboardX"],
    ["onnxruntime","onnxruntime"],
    ["onnxruntime_gpu","onnxruntime_gpu==1.15.1"],
    ["voicefixer","voicefixer==0.1.2"],
]

for lib in req:
    lib_name = lib[0]
    lib_ins = lib[1]
    if not launch.is_installed(lib_name):
        launch.run_pip(
            f"install {lib_ins}",
            f"sd-webui-rvc requirement: {lib_name}")


req_vers = [
    ["librosa", "0.8.1", "librosa==0.8.1"],
    ["hydra-core", "1.2.0.dev2", "hydra-core==1.2.0.dev2"],
    ["antlr4-python3-runtime", "4.8", "antlr4-python3-runtime==4.8"],
    ["omegaconf", "2.2.0", "omegaconf==2.2.0"],
]

import pkg_resources
for lib in req_vers:
    lib_name = lib[0]
    lib_vers = lib[1]
    lib_ins = lib[2]
    if not launch.is_installed(lib_name):
        launch.run_pip(
            f"install {lib_ins}",
            f"sd-webui-rvc requirement: {lib_name}")
    else:
        version = pkg_resources.get_distribution(lib_name).version
        if(version != lib_vers):
            print(f'{lib_name} version: {version} will update to version: {lib_vers}')
            launch.run_pip(f"install {lib_ins}", "update requirements for RVC")


t_name = "triton"
if not launch.is_installed(t_name):
    if sys.platform == "win32":
        launch.run_pip(
            f"install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl",
            f"sd-webui-rvc requirement: {t_name}")
    else:
        launch.run_pip(
            f"install triton==2.0.0",
            f"sd-webui-rvc requirement: {t_name}")
