import modules.scripts as scripts
import gradio as gr
import os, sys, io
import datetime
from modules import script_callbacks
from rvc.i18n import I18nAuto
from rvc import RVC as rvc
from rvc import UVR as uvr
from rvc import training
from rvc import voicefixer
i18n = I18nAuto()

class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "RVC"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Tab("RVC"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                model_list = gr.Dropdown(label="Model", choices=sorted(rvc.names), value=rvc.check_for_name())
                                """if rvc.check_for_name() != '':
                                    rvc.get_vc(sorted(rvc.names)[0])"""
                                spk_item = gr.Slider(
                                    minimum=0,
                                    maximum=2333,
                                    step=1,
                                    label="model_id",
                                    value=0,
                                    visible=False,
                                    interactive=True,
                                )
                                file_index1 = gr.Dropdown(
                                    label="Index (use only if it didn't auto find it / if model not have dont use)",
                                    choices=rvc.get_indexes(),
                                    value=rvc.get_index(),
                                    visible=True,
                                    interactive=True,
                                    )
                                refresh_button = gr.Button("🔄", elem_classes="tool")
                                
                                model_list.change(fn=rvc.match_index, inputs=[model_list],outputs=[file_index1])
                                """model_list.change(
                                    fn=rvc.get_vc,
                                    inputs=[model_list],
                                    outputs=[spk_item],
                                )"""
                                
                                refresh_button.click(fn=rvc.change_choices, inputs=[], outputs=[model_list, file_index1])
                    with gr.Row():
                        pitch_nb = gr.Number(label="Optional: You can change the pitch here or leave it at 0.", value=0)
                    with gr.Row():
                        method = gr.Radio(
                            label="Optional: Change the Pitch Extraction Method.",
                            choices=["pm", "rmvpe", "dio", "mangio-crepe-tiny", "crepe-tiny", "crepe", "mangio-crepe", "harvest"], # Fork Feature. Add Crepe-Tiny
                            value="pm",
                            interactive=True,
                        )
                    with gr.Row():
                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Row():
                                with gr.Column():
                                    index_rate1 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("检索特征占比"),
                                        value=0.66,
                                        interactive=True,
                                        )
                                with gr.Column():
                                    filter_radius0 = gr.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                                        value=3,
                                        step=1,
                                        interactive=True,
                                        )
                            with gr.Row():
                                with gr.Column():
                                    resample_sr0 = gr.Slider(
                                        minimum=0,
                                        maximum=48000,
                                        label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                                        value=0,
                                        step=1,
                                        interactive=True,
                                        visible=False,
                                        )
                                with gr.Column():
                                    rms_mix_rate0 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                                        value=0.21,
                                        interactive=True,
                                        )
                            with gr.Row():
                                with gr.Column():
                                    protect0 = gr.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        label=i18n("保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"),
                                        value=0.33,
                                        step=0.01,
                                        interactive=True,
                                        )
                                with gr.Column():
                                    crepe_hop_length = gr.Slider(
                                        minimum=1,
                                        maximum=512,
                                        step=1,
                                        label="Mangio-Crepe Hop Length. Higher numbers will reduce the chance of extreme pitch changes but lower numbers will increase accuracy.",
                                        value=120,
                                        interactive=True
                                        )
                with gr.Column():
                    with gr.Row():
                        #hidden
                        input_audio0 = gr.Dropdown(label="Audio File", value="", visible=False)
                    with gr.Row():
                        with gr.Tab("From File"):
                            input_audio = gr.Audio(type="filepath")
                            input_audio.change(fn=rvc.save_to_wav, inputs=[input_audio], outputs=[input_audio0])
                        with gr.Tab("From Microphone"):
                            input_audio2 = gr.Audio(source="microphone", type="filepath")
                            input_audio2.change(fn=rvc.save_to_wav, inputs=[input_audio2], outputs=[input_audio0])
                        with gr.Tab("From TTS"):
                            with gr.Row():
                                tts_text = gr.Textbox(label="Text")
                                tts_lang = gr.Textbox(label="Lang", value="en")
                            with gr.Row():
                                with gr.Column():
                                    tts_speed = gr.Slider(
                                        minimum=0.25,
                                        maximum=2.0,
                                        label="Speed",
                                        value=1.25,
                                        step=0.05,
                                        interactive=True,
                                        )
                                with gr.Column():
                                    import_text = gr.Button("Speech", variant="primary")
                            with gr.Row():
                                with gr.Column():
                                    input_audio3 = gr.Audio(label="preview", type="filepath")
                            import_text.click(rvc.tts_speech, inputs=[tts_text,tts_lang,tts_speed], outputs=[input_audio3,input_audio0])
                    with gr.Row():
                        with gr.Column():
                            convert_button = gr.Button(i18n("转换"), variant="primary")
                        with gr.Column():
                            out_fld_btn = gr.Button("Open Output Directory", variant="primary")
                    with gr.Tab("Output"):
                        f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"), visible=False)
                        output_information = gr.Textbox(label=i18n("输出信息"))
                        output_audio = gr.Audio(label="Output Audio (Click on the Three Dots in the Right Corner to Download)", type='filepath')
                convert_button.click(fn=rvc.get_vc, inputs=[model_list], outputs=[spk_item]).then(rvc.vc_single, inputs=[spk_item,input_audio0, pitch_nb, f0_file, method, file_index1, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0, crepe_hop_length], outputs=[output_information, output_audio])
                out_fld_btn.click(rvc.but_opn_fold, outputs=[output_information])
        with gr.Tab("UVR"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_choose = gr.Dropdown(label=i18n("模型"), choices=list(uvr.mdx_get_model_list()), value=list(uvr.mdx_get_model_list())[0])
                    with gr.Row():
                        model_select = gr.Radio(
                            label=i18n("Model Architecture:"),
                            choices=["VR", "MDX"],
                            value="MDX",
                            interactive=True,
                        )
                    with gr.Row():
                        agg = gr.Slider(
                            minimum=0,
                            maximum=20,
                            step=1,
                            label="人声提取激进程度",
                            value=10,
                            interactive=True,
                            visible=False,  # 先不开放调整
                        )
                    with gr.Row():
                        format0 = gr.Radio(
                                    label=i18n("导出文件格式"),
                                    choices=["wav", "flac", "mp3"],
                                    value="mp3",
                                    interactive=True,
                                )
                    opt_vocal_root = gr.Textbox(
                        label=i18n("指定输出主人声文件夹"), value=uvr.output_folder, visible=False
                    )
                    opt_ins_root = gr.Textbox(
                        label=i18n("指定输出非主人声文件夹"), value=uvr.output_folder, visible=False
                    )
                with gr.Column():
                    with gr.Row():
                        with gr.Tab("Select Files"):
                            wav_inputs = gr.File(
                                file_types=["audio"], file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
                            )
                    with gr.Row():
                        vc_output4 = gr.Textbox(label=i18n("输出信息"))
                    with gr.Row():
                        with gr.Column():
                            butn2 = gr.Button(i18n("转换"), variant="primary")
                        with gr.Column():
                            out_fld_btn2 = gr.Button("Open Output Directory", variant="primary")

                model_select.change(
                            fn=update_model_choices,
                            inputs=model_select,
                            outputs=model_choose,
                        )
                out_fld_btn2.click(uvr.but_opn_fold, outputs=[vc_output4])
                butn2.click(
                            uvr.uvr,
                            [
                                model_select,
                                model_choose,
                                opt_vocal_root,
                                wav_inputs,
                                opt_ins_root,
                                agg,
                                format0,
                            ],
                            [vc_output4],
                            api_name="uvr_convert",
                )
        with gr.Tab("Voice Fixer"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        select_mode = gr.Radio(
                            label="Mode",
                            choices=["Remove Noises", "Clean Voice"],
                            value="Remove Noises",
                            interactive=True,
                        )
                    with gr.Row():
                        format_fix = gr.Radio(
                            label="Noise Removal Method",
                            choices=["(Mode 0) Remove Noise", "Mode 0 + Remove higher frequency"],
                            value="(Mode 0) Remove Noise",
                            interactive=True,
                            visible=True,
                        )
                        format_clean = gr.Radio(
                            label="Clean Voice Method",
                            choices=["TFGAN"],
                            value="TFGAN",
                            interactive=True,
                            visible=False,
                        )
                    with gr.Row():
                        input_audio_fix = gr.Audio(type="filepath")
                with gr.Column():
                    with gr.Row():
                        fix_output = gr.Textbox(label=i18n("输出信息"))
                    with gr.Row():
                        output_audio_fix = gr.Audio(label="Output Audio (Click on the Three Dots in the Right Corner to Download)", type='filepath')
                    with gr.Row():
                        with gr.Column():
                            butn_fix = gr.Button(i18n("转换"), variant="primary")
                        with gr.Column():
                            out_fld_btn3 = gr.Button("Open Output Directory", variant="primary")
                out_fld_btn3.click(voicefixer.but_opn_fold, outputs=[output_information])
                butn_fix.click(voicefixer.voice_fix, inputs=[input_audio_fix,select_mode, format_fix, format_clean], outputs=[fix_output, output_audio_fix])
                select_mode.change(change_mode, inputs=[select_mode], outputs=[format_fix,format_clean])
        with gr.Tab("RVC Training"):
            with gr.Row():
                with gr.Column():
                    with gr.Tab("Train Settings"):
                        with gr.Row():
                            cpu_cores = gr.Slider(
                                minimum=0,
                                maximum=rvc.config.n_cpu,
                                step=1,
                                label="Use # CPU(~Cores) for processing",
                                value=rvc.config.n_cpu,
                                interactive=True,
                                visible=True
                            )
                        with gr.Row():
                            method_train = gr.Radio(
                                label="Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'dio': improved speech but slower extraction; 'harvest': better quality but slower extraction):",
                                choices=["harvest","crepe", "mangio-crepe"], # Fork feature: Crepe on f0 extraction for training.
                                value="mangio-crepe",
                                interactive=True,
                            )
                        with gr.Row():
                            extraction_crepe_hop_length = gr.Slider(
                                minimum=1,
                                maximum=512,
                                step=1,
                                label=i18n("crepe_hop_length"),
                                value=128,
                                interactive=True
                            )
                        with gr.Accordion("Training Preferences (You can leave these as they are)", open=False):
                            #gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                            with gr.Column():
                                save_epoch10 = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    step=5,
                                    label="Backup every # of epochs:",
                                    value=25,
                                    interactive=True,
                                )
                                batch_size12 = gr.Slider(
                                    minimum=1,
                                    maximum=40,
                                    step=1,
                                    label="Batch Size (LEAVE IT unless you know what you're doing!):",
                                    value=training.default_batch_size,
                                    interactive=True,
                                )
                                if_save_latest13 = gr.Radio(
                                    label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),
                                    choices=[i18n("是"), i18n("否")],
                                    value=i18n("是"),
                                    interactive=True,
                                )
                                if_cache_gpu17 = gr.Radio(
                                    label=i18n(
                                        "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                                    ),
                                    choices=[i18n("是"), i18n("否")],
                                    value=i18n("否"),
                                    interactive=True,
                                )
                                if_save_every_weights18 = gr.Radio(
                                    label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                                    choices=[i18n("是"), i18n("否")],
                                    value=i18n("是"),
                                    interactive=True,
                                )
                                with gr.Accordion("Base Model Locations:", open=False, visible=True):
                                    pretrained_G14 = gr.Textbox(
                                        label=i18n("加载预训练底模G路径"),
                                        value= rvc.RVC_Folder + "/pretrained_v2/f0G40k.pth",
                                        interactive=True,
                                    )
                                    pretrained_D15 = gr.Textbox(
                                        label=i18n("加载预训练底模D路径"),
                                        value= rvc.RVC_Folder + "/pretrained_v2/f0D40k.pth",
                                        interactive=True,
                                    )
                                    gpus16 = gr.Textbox(
                                        label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                                        value=training.gpus,
                                        interactive=True,
                                    )
                                    sr2 = gr.Radio(
                                        label=i18n("目标采样率"),
                                        choices=["40k", "48k"],
                                        value="40k",
                                        interactive=True
                                    )
                                    if_f0_3 = gr.Radio(
                                        label=i18n("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
                                        choices=[True, False],
                                        value=True,
                                        interactive=True
                                    )
                                    spk_id5 = gr.Slider(
                                        minimum=0,
                                        maximum=4,
                                        step=1,
                                        label=i18n("请指定说话人id"),
                                        value=0,
                                        interactive=True
                                    )
                                    version19 = gr.Radio(
                                        label="RVC version",
                                        choices=["v1", "v2"],
                                        value="v2",
                                        interactive=True
                                    )
                        with gr.Row():
                            total_epoch11 = gr.Slider(
                                minimum=0,
                                maximum=10000,
                                step=10,
                                label="Total # of training epochs (IF you choose a value too high, your model will sound horribly overtrained.):",
                                value=250,
                                interactive=True,
                            )
                with gr.Column():
                    with gr.Tab("Inputs"):
                        with gr.Row():
                            voice_name = gr.Textbox(label="Voice Name:", value="Voice-Name")
                        with gr.Row():
                            upload_files = gr.Files(label='Drop your audios here.',file_types=['audio'])
                        with gr.Row():
                            status = gr.Textbox(label="Status (wait until it says 'end preprocess'):", value="")
                        with gr.Row():
                            with gr.Column():
                                but2 = gr.Button("1 - Process The Dataset", variant="primary")
                        with gr.Row():
                            with gr.Column():
                                but3 = gr.Button("2 - Train Model", variant="primary")
                            with gr.Column():
                                but4 = gr.Button("3 - Train Index", variant="primary")
                        with gr.Row():
                            with gr.Column():
                                but_opn = gr.Button("Open RVC Directory", variant="primary")
                            with gr.Column():
                                but_cln = gr.Button("(Optional-LAST) Clean the Training Directory", variant="primary")
                        #gpus16 = gpu6 cpu_cores = np7, exp_dir1 = voice_name, if_f0_3 = if_f0, f0method8 = method_train
                        upload_files.upload(fn=training.upload_to_dataset, inputs=[upload_files, voice_name], outputs=[status])
                        sr2.change(
                            training.change_sr2,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15, version19],
                        )
                        version19.change(
                            training.change_version19,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15],
                        )
                        if_f0_3.change(
                            training.change_f0,
                            [if_f0_3, sr2, version19],
                            [method_train, pretrained_G14, pretrained_D15],
                        )
                        but_cln.click(training.clean_training_folder, outputs=[status])
                        but_opn.click(training.but_opn_fold, outputs=[status])
                        but2.click(
                                training.preprocess_dataset,
                                [voice_name, sr2, cpu_cores, gpus16, if_f0_3, method_train, version19, extraction_crepe_hop_length],
                                [status],
                        )
                        but3.click(
                            training.click_train,
                            [
                                voice_name,
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
                            ],
                            status,
                        )
                        but4.click(training.train_index, [voice_name, version19], status)
        return [(ui_component, "RVC", "rvc_ui_tab")]

def change_mode(select_mode):
    if "Remove" in select_mode:
        return {"visible": True, "__type__": "update"}, {"visible": False, "__type__": "update"}
    else:
        return {"visible": False, "__type__": "update"}, {"visible": True, "__type__": "update"}
    
def update_model_choices(select_value):
    uvr.uvr5_update_list()
    model_ids = uvr.mdx_get_model_list()
    model_ids_list = list(model_ids)
    if select_value == "VR":
        return {"choices": uvr.uvr5_names, "value": uvr.uvr5_names[0], "__type__": "update"}
    elif select_value == "MDX":
        return {"choices": model_ids_list, "value": model_ids_list[0], "__type__": "update"}
        
script_callbacks.on_ui_tabs(on_ui_tabs)
