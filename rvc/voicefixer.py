import os
import sys
import librosa
import numpy as np
import torch

from pathlib import Path
import ffmpeg

ROOT_DIR = str(Path().absolute())
RVC_Folder = ROOT_DIR + "/RVC"
output_folder_fix = ROOT_DIR + "/outputs/RVC/FIX"
current_dir = os.path.dirname(os.path.realpath(__file__))

def but_opn_fold():
    if sys.platform == "win32":
        os.system(
            "explorer \"%s\""
            % (output_folder_fix.replace('/', '\\'))
        )
    yield ("\"%s\"" % (output_folder_fix.replace('/', '\\')))

def voice_fix(input_audio, select_mode, format_fix, format_clean):
    if type(input_audio) is type(None) or len(str(input_audio)) < 1 or input_audio is None:
        gr.Warning("You need to provide the path to an audio file.")
        return "You need to provide the path to an audio file.", None

    input_audio_path = input_audio
    info = ffmpeg.probe(input_audio, cmd="ffprobe")
    reformat = True
    if Path(input_audio).suffix ==  ".wav" or Path(input_audio).suffix ==  ".flac":
        reformat = False
    if reformat == True or (info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100") == False:
        tmp_path = "%s/%s_rf.flac" % (os.path.join(os.environ["TEMP"]), os.path.splitext(os.path.basename(input_audio))[0])
        os.system(
            "ffmpeg -i \"%s\" -vn -acodec flac -ac 2 -ar 44100 \"%s\" -y"
            % (input_audio, tmp_path)
        )
        input_audio_path = tmp_path

    cuda_enable = True if torch.cuda.is_available() else False
    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    if "Remove" in select_mode:
        from voicefixer import VoiceFixer
        voicefixer = VoiceFixer()
        mod_e = 0 if "(Mode 0)" in format_fix else 1
        o_tput=os.path.join(
                output_folder_fix,  f"{base_name}_{str(mod_e)}_.flac"
            )
        voicefixer.restore(
            input=input_audio_path,  # low quality .wav/.flac file
            output=o_tput,  # save file path
            cuda=cuda_enable,  # GPU acceleration
            mode=mod_e,
        )
        return ("Success.", o_tput)

    else:
        if format_clean == "TFGAN":
            from voicefixer import Vocoder
            vocoder = Vocoder(sample_rate=44100)
            o_tput=os.path.join(
                        output_folder_fix,  f"{base_name}_or_cle_.flac"
                    )
            vocoder.oracle(
                fpath=input_audio_path,
                out_path=o_tput,
                cuda=cuda_enable,
            )  
            return ("Success.", o_tput)
    