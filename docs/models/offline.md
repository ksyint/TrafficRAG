# Local pretrained artifacts

Prepare local pretrained directories on a connected machine and retain each model configuration and processor files.

```bash
hf download OpenGVLab/VideoMAE2 distill/vit_s_k710_dl_from_giant.pth --local-dir weights/videomaev2
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir weights/qwen3-vl
hf download answerdotai/ModernBERT-base --local-dir weights/modernbert
```

Training accepts `--weights weights/videomaev2/distill/vit_s_k710_dl_from_giant.pth --offline`. Caption and batch commands accept `--vlm weights/qwen3-vl --text-encoder weights/modernbert --offline`. The pretrained model pages and download commands are also listed in the root README. The separately trained traffic checkpoint is supplied through `--motion-checkpoint` for batch inference.
