import os, sys, traceback
import datetime, shutil
from pathlib import Path

import torch
import gradio as gr
import ffmpeg
import numpy as np
import json
import subprocess

from infer.modules.uvr5.preprocess import AudioPre, AudioPreDeEcho

from rvc.config import Config
import rvc.files_d as files_d
config = Config()

import logging
logger = logging.getLogger(__name__)

ROOT_DIR = str(Path().absolute())
RVC_Folder = ROOT_DIR + "/RVC"
output_folder = ROOT_DIR + "/outputs/RVC/UVR"
weight_uvr5_root = RVC_Folder + "/uvr5_weights"
weight_uvr5_mdx = RVC_Folder + "/mdx_weights"

from rvc.mdx_processing_script import prepare_mdx, run_mdx


for i in files_d.Model_UVR5:
    file = f'{weight_uvr5_root}/{i}'
    if not os.path.exists(file):
        model = files_d._uvr_base + i
        print('Downloading model...',end=' ')
        subprocess.run(
            ["wget", model, "-O", file]
        )

uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

now_dir = os.path.dirname(os.path.realpath(__file__))

d_checks = f"{now_dir}/uvr5/download_checks.json"
if not os.path.exists(d_checks):
    import urllib.request
    model_ids = json.load(urllib.request.urlopen(files_d._mdx_downcheck))
else:
    with open(d_checks) as fp:
        model_ids = json.load(fp)
model_ids = model_ids["mdx_download_list"].values()

def but_opn_fold():
    if sys.platform == "win32":
        os.system(
            "explorer \"%s\""
            % (output_folder.replace('/', '\\'))
        )
    yield ("\"%s\"" % (output_folder.replace('/', '\\')))

def mdx_get_model_list():
    return model_ids

def uvr5_update_list():
    uvr5_names = []
    for name in os.listdir(weight_uvr5_root):
        if name.endswith(".pth") or "onnx" in name:
            uvr5_names.append(name.replace(".pth", ""))

def id_to_ptm(mkey):
    if mkey in model_ids:
        mpath = f'{weight_uvr5_mdx}/{mkey}'
        if not os.path.exists(mpath):
            print('Downloading model...',end=' ')
            subprocess.run(
                ["wget", files_d._Models+mkey, "-O", mpath]
            )
            print(f'saved to {mpath}')
            # get_ipython().system(f'gdown {model_id} -O /content/tmp_models/{mkey}')
            return mpath
        else:
            return mpath
    else:
        mpath = f'{weight_uvr5_mdx}/{mkey}'
        return mpath

def uvr(architecture, model_name, save_root_vocal, paths, save_root_ins, agg, format0):
    if type(paths) == type(None) or len(paths) < 1:
        gr.Warning("Select audio file(s) first!")
        yield "Select audio file(s) first!"
        return
    elif type(model_name) == type(None) or model_name == None or len(model_name) < 1:
        gr.Warning("Please select model.")
        yield "Please select model."
        return
    infos = []
    save_root_vocal = (
        save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    )
    save_root_ins = (
        save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    )
    if architecture == "VR":
        try:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            if model_name == "onnx_dereverb_By_FoxJoy":
                pass #REMOVED pre_fun = MDXNetDereverb(15, config.device, weight_uvr5_root)
            else:
                pre_fun = func(agg=int(agg), model_path=os.path.join(weight_uvr5_root,model_name+".pth"), device=config.device, is_half=config.is_half)
            paths = [path.name for path in paths]
            for path in paths:
                inp_path= path
                info = ffmpeg.probe(path, cmd="ffprobe")
                if (info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100") == False:
                    tmp_path = "%s/%s.reformatted.wav" % (os.path.join(os.environ["TEMP"]), os.path.basename(inp_path))
                    os.system(
                        "ffmpeg -i \"%s\" -vn -acodec pcm_s16le -ac 2 -ar 44100 \"%s\" -y"
                        % (inp_path, tmp_path)
                    )
                    inp_path = tmp_path
                try:
                    pre_fun._path_audio_(inp_path , save_root_ins,save_root_vocal, format0)
                    infos.append("%s->Success"%(os.path.basename(inp_path)))
                    
                    yield "\n".join(infos)
                except:
                    infos.append("%s->%s" % (os.path.basename(inp_path),traceback.format_exc()))
                    yield "\n".join(infos)
        except:
            infos.append(traceback.format_exc())
            yield "\n".join(infos)
        finally:
            try:
                if model_name == "onnx_dereverb_By_FoxJoy":
                    del pre_fun.pred.model
                    del pre_fun.pred.model_
                else:
                    del pre_fun.model
                    del pre_fun
            except:
                traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Executed torch.cuda.empty_cache()")
        yield "\n".join(infos)
    elif architecture == "MDX":
        try:
            paths = [path.name for path in paths]

            invert = True
            denoise = True
            use_custom_parameter = False
            dim_f = 3072
            dim_t = 256
            n_fft = 7680
            use_custom_compensation = False
            compensation = 1.025
            suffix = "Vocals_custom"  # @param ["Vocals", "Drums", "Bass", "Other"]{allow-input: true}
            suffix_invert = "Instrumental_custom"  # @param ["Instrumental", "Drumless", "Bassless", "Instruments"]{allow-input: true}
            print_settings = True  # @param{type:"boolean"}
            onnx = id_to_ptm(model_name)
            compensation = (
                compensation
                if use_custom_compensation or use_custom_parameter
                else None
            )
            mdx_model = prepare_mdx(
                onnx,
                use_custom_parameter,
                dim_f,
                dim_t,
                n_fft,
                compensation=compensation,
            )

            for path in paths:
                inp_path = path
                info = ffmpeg.probe(path, cmd="ffprobe")
                if (info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100") == False:
                    tmp_path = "%s/%s.reformatted.wav" % (os.path.join(os.environ["TEMP"]), os.path.basename(inp_path))
                    os.system(
                        "ffmpeg -i \"%s\" -vn -acodec pcm_s16le -ac 2 -ar 44100 \"%s\" -y"
                        % (inp_path, tmp_path)
                    )
                    inp_path = tmp_path
                # inp_path = os.path.join(inp_root, path)
                suffix_naming = suffix if use_custom_parameter else None
                diff_suffix_naming = suffix_invert if use_custom_parameter else None
                run_mdx(
                    onnx,
                    mdx_model,
                    inp_path,
                    format0,
                    diff=invert,
                    suffix=suffix_naming,
                    diff_suffix=diff_suffix_naming,
                    denoise=denoise,
                    save_root_vocal=save_root_vocal,
                    save_root_ins=save_root_ins,
                )

            if print_settings:
                print()
                print("[MDX-Net settings used]")
                print(f"Model used: {onnx}")
                print(f"Model parameters:")
                print(f"    -dim_f: {mdx_model.dim_f}")
                print(f"    -dim_t: {mdx_model.dim_t}")
                print(f"    -n_fft: {mdx_model.n_fft}")
                print(f"    -compensation: {mdx_model.compensation}")
                print()
                print("[Input file]")
                print("filename(s): ")
                for filename in paths:
                    print(f"    -{filename}")
                    infos.append(f"{os.path.basename(filename)}->Success")
                    yield "\n".join(infos)
        except:
            infos.append(traceback.format_exc())
            yield "\n".join(infos)
        finally:
            try:
                del mdx_model
            except:
                traceback.print_exc()

            print("clean_empty_cache")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
