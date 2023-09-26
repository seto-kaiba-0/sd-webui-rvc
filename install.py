import launch
import os
import sys

req = [
    ["parselmouth","praat-parselmouth>=0.4.2"],
    ["torchcrepe","torchcrepe==0.0.20"],
    ["pyworld","pyworld>=0.3.2"],
    ["faiss","faiss-cpu==1.7.3"],
    ["ffmpeg","ffmpeg-python>=0.2.0"],
    ["fairseq","fairseq==0.12.2"],
    ["soundfile","soundfile>=0.12.1"],
    ["gtts","gtts==2.3.2"],
    ["pydub","pydub>=0.25.1"],
    ["numpy","numpy>=1.23.5"],
    ["scipy","scipy>=1.9.3"],
    ["librosa","librosa>=0.9.1"],
    ["pyworld","pyworld>=0.3.2"],
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

package_name = "omegaconf"
package_version = "2.2.0"
if not launch.is_installed(package_name):
            launch.run_pip(
                f"install {package_name}",
                f"sd-webui-rvc requirement: {package_name}")
import pkg_resources
version = pkg_resources.get_distribution(package_name).version
if(version != package_version):
    print(f'{package_name} version: {version} will update to version: {package_version}')
    launch.run_pip(f"install {package_name}=={package_version}", "update requirements for RVC")
    

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