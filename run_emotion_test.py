import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import random
import shutil
import tempfile
import time
from pathlib import Path
from xml.etree.ElementTree import Element, tostring

import numpy as np
import soundfile as sf
import torch
from dots_tts.edit_runtime import DotsTtsEditRuntime


def read_path(prompt):
    value = input(prompt).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return Path(value).expanduser().resolve()


# 只读取本地缓存，不重新下载模型。
try:
    from modelscope.hub.snapshot_download import snapshot_download
    model_dir = Path(snapshot_download(
        "dots-studio/dots.tts.edit",
        local_files_only=True,
    )).resolve()
except Exception as error:
    print(f"未能自动定位缓存：{error}")
    model_dir = read_path("请输入已下载的模型目录：")

required = [
    "config.json", "llm_config.json", "tokenizer.json",
    "latent_stats.pt", "model.safetensors",
    "speaker_encoder.safetensors", "vocoder.safetensors",
]
missing = [
    name for name in required
    if not (model_dir / name).is_file()
    or (model_dir / name).stat().st_size == 0
]
if missing:
    raise SystemExit(f"模型目录缺少文件：{missing}\n目录：{model_dir}")

source = Path('/home/yu/Downloads/桑梓旁白男.wav').expanduser().resolve()
if not source.is_file():
    raise SystemExit(f"音频不存在：{source}")

text = '白嘉轩就更进一步数落起来前头六个女人已经花光了父亲几十年节俭积攒的银钱而且连着卖掉两匹骡子槽头现有的红马和黄牛即使全拉到集上卖了也不够定一个媳妇的聘礼'
if not text:
    raise SystemExit("原文不能为空。")

info = sf.info(str(source))
if info.frames <= 0:
    raise SystemExit("输入音频为空。")
if not torch.cuda.is_available():
    raise SystemExit("当前环境未检测到 GPU，请使用 dots-edit 环境。")

output_root = Path("/home/yu/Workspace/dots.tts/test_outputs")
output_root.mkdir(parents=True, exist_ok=True)
output_dir = Path(tempfile.mkdtemp(prefix="emotion-", dir=output_root))
shutil.copy2(source, output_dir / ("original" + source.suffix))

print(f"\n模型目录：{model_dir}")
print(f"输入时长：{info.duration:.2f} 秒")
print(f"GPU：{torch.cuda.get_device_name(0)}")
print(f"输出目录：{output_dir}")
print("加载模型……", flush=True)

runtime = DotsTtsEditRuntime.from_pretrained(
    str(model_dir),
    precision="bfloat16",
    optimize=False,
)

settings = {
    "seed": 42,
    "num_steps": 10,
    "guidance_scale": 1.2,
}
report = {
    "model_dir": str(model_dir),
    "source_audio": str(source),
    "text": text,
    **settings,
    "outputs": [],
}

emotions = [
    ("angry", "a very angry and strongly displeased tone"),
    ("sad", "a deeply sad and sorrowful tone"),
]

for name, description in emotions:
    # 将情绪标签包住全部原文，源文本和目标文本保持一致。
    tag = Element("emo", {"desc": description})
    tag.text = text
    instruction = tostring(tag, encoding="unicode")

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])

    print(f"\n生成 {name}……", flush=True)
    start = time.perf_counter()
    with torch.inference_mode():
        result = runtime.generate_edit(
            source_audio_path=str(source),
            instruction=instruction,
            source_text=text,
            target_text=text,
            use_xvector="auto",
            num_steps=settings["num_steps"],
            guidance_scale=settings["guidance_scale"],
        )

    wave = result["audio"].detach().float().cpu().squeeze().numpy()
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise RuntimeError(f"{name} 输出音频无效。")

    sample_rate = int(result["sample_rate"])
    destination = output_dir / f"{name}.wav"
    sf.write(str(destination), wave, sample_rate, subtype="FLOAT")

    report["outputs"].append({
        "emotion": name,
        "instruction": instruction,
        "file": str(destination),
        "duration": len(wave) / sample_rate,
        "elapsed_seconds": time.perf_counter() - start,
    })
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已保存：{destination}", flush=True)

print(f"\n完成。对比试听目录：{output_dir}")
