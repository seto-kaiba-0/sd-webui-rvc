import os, sys, traceback
import datetime, shutil
from pathlib import Path

import torch
import gradio as gr
from infer_pack.models import (SynthesizerTrnMs256NSFsid,SynthesizerTrnMs256NSFsid_nono,SynthesizerTrnMs768NSFsid,SynthesizerTrnMs768NSFsid_nono)
import ffmpeg
import numpy as np
from scipy.io.wavfile import write as writewav

from rvc.config import Config
import rvc.files_d as files_d
from rvc.load_audio import load_audio
from rvc.vc_infer_pipeline import VC
config = Config()
from fairseq import checkpoint_utils
from gtts import gTTS
from pydub import AudioSegment
import subprocess

def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

ROOT_DIR = str(Path().absolute())
RVC_Folder = ROOT_DIR + "/RVC"
output_folder_general = ROOT_DIR + "/outputs"
output_folder = ROOT_DIR + "/outputs/RVC/"
output_folder_uvr = ROOT_DIR + "/outputs/RVC/UVR"
output_folder_fix = ROOT_DIR + "/outputs/RVC/FIX"
weight_root = RVC_Folder + "/weights"
weight_uvr5_root = RVC_Folder + "/uvr5_weights"
weight_uvr5_mdx = RVC_Folder + "/mdx_weights"
index_root = RVC_Folder + "/logs"
rmvpe = RVC_Folder + "/rmvpe"
hubertfolder = RVC_Folder + "/hubert"
audiofolder = output_folder + "tmp_audio/"
make_dir(RVC_Folder)
make_dir(ROOT_DIR + "/outputs")
make_dir(output_folder_general)
make_dir(output_folder)
make_dir(weight_root)
make_dir(weight_uvr5_root)
make_dir(weight_uvr5_mdx)
make_dir(output_folder_uvr)
make_dir(output_folder_fix)
make_dir(index_root)
make_dir(audiofolder)
make_dir(rmvpe)
make_dir(hubertfolder)

filev1 = f'{rmvpe}/rmvpe.pt'
if not os.path.exists(filev1):
    print('Downloading model...',end=' ')
    subprocess.run(
        ["wget", files_d._rmvpe_base, "-O", filev1]
    )
filev1 = f'{hubertfolder}/hubert_base.pt'
if not os.path.exists(filev1):
    print('Downloading model...',end=' ')
    subprocess.run(
        ["wget", files_d._hubert_base, "-O", filev1]
    )

make_dir(index_root + "/mute")
make_dir(index_root + "/mute/0_gt_wavs")
make_dir(index_root + "/mute/1_16k_wavs")
make_dir(index_root + "/mute/2a_f0")
make_dir(index_root + "/mute/2b-f0nsf")
make_dir(index_root + "/mute/3_feature256")
make_dir(index_root + "/mute/3_feature768")


for i in files_d.mute_files:
    file_mt = f"{index_root}/mute/{i}"
    if not os.path.exists(file_mt):
        mute_dw = files_d.mute_uri + i
        print('Downloading mute...',end=' ')
        subprocess.run(
            ["wget", mute_dw, "-O", file_mt]
        )

def but_opn_fold():
    if sys.platform == "win32":
        os.system(
            "explorer \"%s\""
            % (output_folder.replace('/', '\\'))
        )
    yield ("\"%s\"" % (output_folder.replace('/', '\\')))

hubert_model = None
def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [f"{RVC_Folder}/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
def check_for_name():
    if len(names) > 0:
        return sorted(names)[0]
    else:
        return ''
        
def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def get_index():
    if check_for_name() != '':
        chosen_model=sorted(names)[0].rsplit('.', maxsplit=1)[0]
        logs_path= index_root + "/" + chosen_model
        if os.path.isfile(logs_path + ".index"):
            return logs_path + ".index"
        elif os.path.exists(index_root):
            for file in os.listdir(index_root):
                if file.endswith(".index"):
                    if "added" in file and chosen_model in file and "nprobe" in file:
                        return os.path.join(index_root, file)
            return ''
        else:
            return ''
        
def get_indexes():
    indexes_list=[]
    for dirpath, dirnames, filenames in os.walk(index_root):
        for filename in filenames:
            if filename.endswith(".index"):
                indexes_list.append(os.path.join(dirpath,filename))
    if len(indexes_list) > 0:
        return indexes_list
    else:
        return ''

def match_index(model_list):
    try:
        model=model_list.rsplit('.', maxsplit=1)[0]
    except:
        model=""
    parent_dir= index_root + "/" + model
    if os.path.isfile(index_root + "/" + model + ".index"):
        return index_root + "/" + model + ".index"
    elif os.path.exists(parent_dir):
        for filename in os.listdir(parent_dir):
            if filename.endswith(".index"):
                index_path=os.path.join(parent_dir,filename)
                return index_path
    elif os.path.exists(index_root):
        for file in os.listdir(index_root):
            if file.endswith(".index"):
                if "added" in file and model in file and "nprobe" in file:
                    return os.path.join(index_root, file)
    else:
        return ''

def get_vc(sid):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model != None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###楼下不这么折腾清理不干净
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("RVC Model: %s" % sid)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"maximum": n_spk, "__type__": "update"}

def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    #file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
    root_location=audiofolder
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if type(input_audio_path) is type(None) or len(str(input_audio_path)) < 1 or input_audio_path is None:
        gr.Warning("You need to provide the path to an audio file.")
        return "You need to provide the path to an audio file.", None
    full_audio_path = str(root_location) + str(input_audio_path)
    if not os.path.exists(full_audio_path):
        gr.Warning(f"Could not find that file in audios/{input_audio_path}")
        return f"Could not find that file in audios/{input_audio_path}", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(full_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        if file_index == None:
            file_index = ""
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
        )  # 防止小白写错，自动帮他替换掉
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        gr.Info("Starting process...")
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_file=f0_file,
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
        new_path= output_folder + new_name
        scaled = np.int16(audio_opt / np.max(np.abs(audio_opt)) * 32767)
        writewav(new_path, tgt_sr, scaled)
        
        gr.Info('Success.')
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

def save_to_wav(record_button):
    for f in os.listdir(audiofolder):
        try:
            os.remove(os.path.join(audiofolder, f))
        except:
            pass
    if record_button is None:
        pass
    else:
        path_to_file=record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
        new_path= audiofolder + new_name
        shutil.move(path_to_file,new_path)
        return os.path.basename(new_path)

def tts_speech(txt, ln, speed):
    for f in os.listdir(audiofolder):
        try:
            os.remove(os.path.join(audiofolder, f))
        except:
            pass
            
    if len(txt) < 2 or len(ln) not in [2,3]:
        if len(txt) < 2:
            gr.Warning("TTS minumum length: 2")
        else:
            gr.Warning("Please check lang code.")
        return
    tts = gTTS(text=txt, lang=ln)
    new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.mp3'
    new_path = audiofolder + "/" + new_name
    tts.save(new_path)
    
    sound = AudioSegment.from_file(new_path, format="mp3")
    mod_sound = speed_change(sound, speed)
    mod_sound.export(new_path, format="mp3")
    return new_path, os.path.basename(new_path)
    

def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })

    # convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. They often only
    # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)