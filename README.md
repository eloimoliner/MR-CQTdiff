# MR-CQTdiff: An Octave-based Multi-Resolution CQT Architecture for Diffusion-based Audio Generation

This repository contains the demo page and audio samples for our paper "An Octave-based Multi-Resolution CQT Architecture for Diffusion-based Audio Generation".

## 🎵 Listen to Audio Samples

Visit our [demo page](https://eloimoliner.github.io/MR-CQTdiff/) to listen to unconditional audio generation samples comparing different diffusion models:

- **UNet-1D**: A 1-dimensional U-Net composed of temporal convolutions, similar to architectures used in waveform-domain diffusion
- **NCSN++**: A 2-dimensional U-Net operating on STFT representations, using 2D convolutions over time and frequency axes
- **CQTDiff+**: A baseline model that uses a differentiable and invertible CQT representation combined with a U-Net architecture
- **MR-CQTDiff**: Our proposed model, which extends CQTdiff+ by introducing a multi-resolution CQT filter bank
- **LDM**: Latent Diffusion Model

All architectures are configured to share a similar parameter count of around 40 million parameters.

## Citation

If you use this work, please cite:

```bibtex
@article{costa2025mrcqtdiff,
  title={An Octave-based Multi-Resolution CQT Architecture for Diffusion-based Audio Generation},
  author={Costa, Maurício do V. M. da and Moliner, Eloi},
  year={2025}
}
```

## 👥 Authors

- **Maurício do V. M. da Costa** - MTDML, IMM, University of Osnabrück, Germany
- **Eloi Moliner** - Acoustics Lab, Aalto University, Finland

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.