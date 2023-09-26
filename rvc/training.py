import os, sys, shutil
from time import sleep
from random import shuffle
from pathlib import Path
import threading
from subprocess import Popen

import torch
import gradio as gr
import numpy as np
import faiss
import subprocess

from rvc.config import Config
import rvc.files_d as files_d
config = Config()
from rvc.i18n import I18nAuto
i18n = I18nAuto()

current_dir = os.path.dirname(os.path.realpath(__file__))

def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

ROOT_DIR = str(Path().absolute())
RVC_Folder = ROOT_DIR + "/RVC"
training_folder = RVC_Folder + "/training"
dataset_folder = RVC_Folder + "/training/datasets"
models_folder = RVC_Folder + "/training/models"
logs_folder = RVC_Folder + "/training/logs"
pretrained = RVC_Folder + "/pretrained_v1"
pretrainedv2 = RVC_Folder + "/pretrained_v2"
weight_root = RVC_Folder + "/weights"
index_root = RVC_Folder + "/logs"
make_dir(training_folder)
make_dir(dataset_folder)
make_dir(models_folder)
make_dir(logs_folder)
make_dir(pretrained)
make_dir(pretrainedv2)



for i in files_d.pretrained_md:
    filev1 = f'{pretrained}/{i}'
    if not os.path.exists(filev1):
        modelv1 = files_d._t_base + i
        print('Downloading model...',end=' ')
        subprocess.run(
            ["wget", modelv1, "-O", filev1]
        )
    filev2 = f'{pretrainedv2}/{i}'
    if not os.path.exists(filev2):
        modelv2 = files_d._t_base_v2 + i
        print('Downloading model...',end=' ')
        subprocess.run(
            ["wget", modelv2, "-O", filev2]
        )

def but_opn_fold():
    if sys.platform == "win32":
        os.system(
            "explorer \"%s\""
            % (RVC_Folder.replace('/', '\\'))
        )
    yield ("\"%s\"" % (RVC_Folder.replace('/', '\\')))

def clean_training_folder():
    try:
        if os.path.exists(training_folder):
            shutil.rmtree(training_folder)
        make_dir(training_folder)
        make_dir(dataset_folder)
        make_dir(models_folder)
        make_dir(logs_folder)
        yield "Training directory cleaned."
    except Exception as e:
        print(e)
        yield "Training directory in use."


ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if (not torch.cuda.is_available()) or ngpu == 0:
    if_gpu_ok = False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if (
            "10" in gpu_name
            or "16" in gpu_name
            or "20" in gpu_name
            or "30" in gpu_name
            or "40" in gpu_name
            or "A2" in gpu_name.upper()
            or "A3" in gpu_name.upper()
            or "A4" in gpu_name.upper()
            or "P4" in gpu_name.upper()
            or "A50" in gpu_name.upper()
            or "A60" in gpu_name.upper()
            or "70" in gpu_name
            or "80" in gpu_name
            or "90" in gpu_name
            or "M4" in gpu_name.upper()
            or "T4" in gpu_name.upper()
            or "TITAN" in gpu_name.upper()
        ):  # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok == True and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

def if_done(done, p):
    while 1:
        if p.poll() == None:
            sleep(0.5)
        else:
            break
    done[0] = True

def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() == None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def upload_to_dataset(files, trainset_dir):
    if trainset_dir == '':
        fullpth_trainset_folder = dataset_folder + "/dataset/"

    fullpth_trainset_folder = dataset_folder + "/" + trainset_dir
    if not os.path.exists(fullpth_trainset_folder):
        os.makedirs(fullpth_trainset_folder)
    count = 0
    for file in files:
        path=file.name
        shutil.copy2(path,fullpth_trainset_folder)
        count += 1
    gr.Info(f'{count} files ready. > Now click "Process The Dataset."')
    yield f' {count} files moved to {fullpth_trainset_folder}.'


def preprocess_dataset(exp_dir1, sr, n_p, gpus, if_f0_3, method_train, version19, echl):
    gr.Info("Step 1 \r\n Wait to see \"all feature done\" in the status box to know it finished.")
    #preprocessing
    exp_dir = exp_dir1
    if(exp_dir1 == ''):
        exp_dir = "dataset"
        
    fullpth_trainset_folder = dataset_folder + "/" + exp_dir
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (training_folder, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (training_folder, exp_dir), "w")
    f.close()
    cmd = (
        config.python_cmd
        + " %s/trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s "
        % (current_dir, fullpth_trainset_folder, sr, n_p, training_folder, exp_dir)
        + str(config.noparallel)
    )
    print(cmd)
    p = Popen(cmd, shell=True)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (training_folder, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0] == True:
            break
    with open("%s/logs/%s/preprocess.log" % (training_folder, exp_dir), "r") as f:
        log = f.read()
    print(log)
    gr.Info("Step 2 : \r\n Wait to see \"all feature done\" in the status box to know it finished.")
    yield log
    
    ##extract_f0_feature
    
    gpus = gpus.split("-")

    f = open("%s/logs/%s/extract_f0_feature.log" % (training_folder, exp_dir), "w")
    f.close()
    if if_f0_3:
        cmd = config.python_cmd + " extract_f0_print.py %s/logs/%s %s %s %s" % (
            training_folder,
            exp_dir,
            n_p,
            method_train,
            echl,
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=current_dir)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
        ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
        done = [False]
        threading.Thread(
            target=if_done,
            args=(
                done,
                p,
            ),
        ).start()
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (training_folder, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0] == True:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (training_folder, exp_dir), "r") as f:
            log = f.read()
        print(log)
        yield log
    ####对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            config.python_cmd
            + " extract_feature_print.py %s %s %s %s %s/logs/%s %s"
            % (
                config.device,
                leng,
                idx,
                n_g,
                training_folder,
                exp_dir,
                version19,
            )
        )
        print(cmd)
        p = Popen(
            cmd, shell=True, cwd=current_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=training_folder
        ps.append(p)
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (training_folder, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0] == True:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (training_folder, exp_dir), "r") as f:
        log = f.read()
    print(log)
    yield log
    gr.Info("Dataset ready, You can use train button now.")

def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (training_folder, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (RVC_Folder, sr2, RVC_Folder, fea_dim, RVC_Folder, RVC_Folder, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (RVC_Folder, sr2, RVC_Folder, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:", gpus16)
    if pretrained_G14 == "":
        print("no pretrained Generator")
    if pretrained_D15 == "":
        print("no pretrained Discriminator")
    if gpus16:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -modeldir %s -wroot %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                exp_dir,
                weight_root,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "",
                ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -modeldir %s -wroot %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                exp_dir,
                weight_root,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "\b",
                ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "\b",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    print(cmd)
    p = Popen(cmd, shell=True, cwd=current_dir)
    p.wait()
    gr.Warning('Done! Check your console in Colab to see if it trained successfully.')
    return 'Done! Check your console in Colab to see if it trained successfully.'
    
    
# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    exp_dir = "%s/logs/%s" % (training_folder, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if os.path.exists(feature_dir) == False:
        return str(feature_dir) + " folder not found. Run process the dateset first!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return str(feature_dir) + " empty. Run process the dateset first! (*Sometimes, builder does not create index file, its normal*)."
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos = []
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/%s_trained_IVF%s_Flat_nprobe_%s_%s.index"
        % (exp_dir,exp_dir1, n_ivf, index_ivf.nprobe, version19),
    )
    # faiss.write_index(index, '%s/trained_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/%s.index"
        % (index_root, exp_dir1),
    )
    infos.append(
        "Index build sucessfully，%s_added_IVF%s_Flat_nprobe_%s_%s.index renamed to > %s.index | Loc: %s/%s.index"
        % (exp_dir1, n_ivf, index_ivf.nprobe, version19, exp_dir1, index_root, exp_dir1)
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    gr.Info('Successfully trained the index file!')
    yield "\n".join(infos)
    
def change_sr2(sr2, if_f0_3, version19):
    path_str = "_v1" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    if_pretrained_generator_exist = os.access("%s/pretrained%s/%sG%s.pth" % (RVC_Folder, path_str, f0_str, sr2), os.F_OK)
    if_pretrained_discriminator_exist = os.access("%s/pretrained%s/%sD%s.pth" % (RVC_Folder, path_str, f0_str, sr2), os.F_OK)
    if (if_pretrained_generator_exist == False):
        print("%s/pretrained%s/%sG%s.pth" % (RVC_Folder, path_str, f0_str, sr2), "not exist, will not use pretrained model")
    if (if_pretrained_discriminator_exist == False):
        print("%s/pretrained%s/%sD%s.pth" % (RVC_Folder, path_str, f0_str, sr2), "not exist, will not use pretrained model")
    return (
        ("%s/pretrained%s/%sG%s.pth" % (RVC_Folder, path_str, f0_str, sr2)) if if_pretrained_generator_exist else "",
        ("%s/pretrained%s/%sD%s.pth" % (RVC_Folder, path_str, f0_str, sr2)) if if_pretrained_discriminator_exist else "",
        {"visible": True, "__type__": "update"}
    )
        
def change_version19(sr2, if_f0_3, version19):
    path_str = "_v1" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    if_pretrained_generator_exist = os.access("%s/pretrained%s/%sG%s.pth" % (RVC_Folder, path_str, f0_str, sr2), os.F_OK)
    if_pretrained_discriminator_exist = os.access("%s/pretrained%s/%sD%s.pth" % (RVC_Folder, path_str, f0_str, sr2), os.F_OK)
    if (if_pretrained_generator_exist == False):
        print("%s/pretrained%s/%sG%s.pth" % (RVC_Folder, path_str, f0_str, sr2), "not exist, will not use pretrained model")
    if (if_pretrained_discriminator_exist == False):
        print("%s/pretrained%s/%sD%s.pth" % (RVC_Folder, path_str, f0_str, sr2), "not exist, will not use pretrained model")
    return (
        ("%s/pretrained%s/%sG%s.pth" % (RVC_Folder, path_str, f0_str, sr2)) if if_pretrained_generator_exist else "",
        ("%s/pretrained%s/%sD%s.pth" % (RVC_Folder, path_str, f0_str, sr2)) if if_pretrained_discriminator_exist else "",
    )
    
def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "_v1" else "_v2"
    if_pretrained_generator_exist = os.access("%s/pretrained%s/f0G%s.pth" % (RVC_Folder, path_str, sr2), os.F_OK)
    if_pretrained_discriminator_exist = os.access("%s/pretrained%s/f0D%s.pth" % (RVC_Folder, path_str, sr2), os.F_OK)
    if (if_pretrained_generator_exist == False):
        print("%s/pretrained%s/f0G%s.pth" % (RVC_Folder, path_str, sr2), "not exist, will not use pretrained model")
    if (if_pretrained_discriminator_exist == False):
        print("%s/pretrained%s/f0D%s.pth" % (RVC_Folder, path_str, sr2), "not exist, will not use pretrained model")
    if if_f0_3:
        return (
            {"visible": True, "__type__": "update"},
            "%s/pretrained%s/f0G%s.pth" % (RVC_Folder, path_str, sr2) if if_pretrained_generator_exist else "",
            "%s/pretrained%s/f0D%s.pth" % (RVC_Folder, path_str, sr2) if if_pretrained_discriminator_exist else "",
        )
    return (
        {"visible": False, "__type__": "update"},
        ("%s/pretrained%s/G%s.pth" % (RVC_Folder, path_str, sr2)) if if_pretrained_generator_exist else "",
        ("%s/pretrained%s/D%s.pth" % (RVC_Folder, path_str, sr2)) if if_pretrained_discriminator_exist else "",
    )