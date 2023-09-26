wget = "https://" + "eternallybored.org/misc/wget/1.21.4/64/wget.exe"
#wget = "https://" + "web.archive.org/web/20230716081841if_/https://" + "eternallybored.org/misc/wget/1.21.4/64/wget.exe"

_Models = "https://" + "github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
_uvr_base = "https://" + "huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/"
Model_UVR5 = ["HP2_all_vocals.pth", "HP3_all_vocals.pth", "HP5_only_main_vocal.pth", "VR-DeEchoAggressive.pth", "VR-DeEchoDeReverb.pth", "VR-DeEchoNormal.pth"]

_t_base = "https://" + "huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/"
_t_base_v2 = "https://" + "huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/"
pretrained_md = ["D32k.pth", "D40k.pth", "D48k.pth", "f0D32k.pth", "f0D40k.pth", "f0D48k.pth", "f0G32k.pth", "f0G40k.pth", "f0G48k.pth", "G32k.pth", "G40k.pth", "G48k.pth"]

mute_files = ["0_gt_wavs/mute32k.wav", "0_gt_wavs/mute40k.wav", "0_gt_wavs/mute48k.wav", "1_16k_wavs/mute.wav", "2a_f0/mute.wav.npy", "2b-f0nsf/mute.wav.npy", "3_feature256/mute.npy", "3_feature768/mute.npy"]
mute_uri = "https://" + "github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/raw/main/logs/mute/"

_hubert_base = "https://" + "huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
_rmvpe_base = "https://" + "huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
_ffmpeg_base = "https://" + "huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe"
_ffprobe_base = "https://" + "huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe"


_mdx_downcheck = "https://" + "raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"
