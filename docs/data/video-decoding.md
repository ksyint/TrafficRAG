# Video timestamps and frame preparation

Video decoding reads presentation timestamps with PyAV. Motion windows use the same timestamps as the source annotations. Uniform sampling supplies the configured VideoMAE frame count and spatial size.

Qwen receives its own crop samples with crop-relative timestamp metadata. Native processors apply the model-specific image normalization. A frame directory is not required.

```bash
ffmpeg -i recording.mov -map 0:v:0 -an -c:v libx264 -pix_fmt yuv420p data/videos/recording.mp4
```

If a recording is re-encoded, keep the annotation time origin aligned with the converted video. Batch caches include the source file size and modification time and therefore refresh when the input file changes.
