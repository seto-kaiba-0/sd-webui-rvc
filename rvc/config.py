import os, sys
import torch
from multiprocessing import cpu_count
from pathlib import Path
import rvc.files_d as files_d
import subprocess
current_dir = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = str(Path().absolute())
os.environ['PATH'] += os.pathsep + current_dir + '\\ffmpeg'

if sys.platform == "win32":
    if not os.path.exists(current_dir + '\\ffmpeg'):
        os.mkdir(current_dir + '\\ffmpeg')
    wget_lc = current_dir + '\\ffmpeg\\wget.exe'
    ffmpeg_lc = current_dir + '\\ffmpeg\\ffmpeg.exe'
    ffprobe_lc = current_dir + '\\ffmpeg\\ffprobe.exe'
    if not os.path.exists(wget_lc):
        from torch.hub import download_url_to_file
        download_url_to_file(files_d.wget, wget_lc, progress=True)
    if not os.path.exists(ffmpeg_lc):
        subprocess.run(
            ["wget", files_d._ffmpeg_base, "-O", ffmpeg_lc]
        )
    if not os.path.exists(ffprobe_lc):
        subprocess.run(
            ["wget", files_d._ffprobe_base, "-O", ffprobe_lc]
        )

def config_file_change_fp32():
    for config_file in ["32k.json", "40k.json", "48k.json", "32k_v2.json", "48k_v2.json"]:
        with open(f"{current_dir}/configs/{config_file}", "r") as f:
            strr = f.read().replace("true", "false")
        with open(f"{current_dir}/configs/{config_file}", "w") as f:
            f.write(strr)
    with open(f"{current_dir}/trainset_preprocess_pipeline_print.py", "r") as f:
        strr = f.read().replace("3.7", "3.0")
    with open(f"{current_dir}/trainset_preprocess_pipeline_print.py", "w") as f:
        f.write(strr)


class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        if sys.platform == "win32":
            self.python_cmd = ROOT_DIR + "/venv/Scripts/python.exe"
        else:
            self.python_cmd = ROOT_DIR + "/venv/bin/python"
        self.noparallel = False
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

   

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                #print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                config_file_change_fp32()
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open(f"{current_dir}/trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open(f"{current_dir}/trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            #print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
            self.is_half = False
            config_file_change_fp32()
        else:
            #print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = False
            config_file_change_fp32()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max
