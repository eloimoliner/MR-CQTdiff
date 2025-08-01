# MR-CQTdiff: An Octave-based Multi-Resolution CQT Architecture for Diffusion-based Audio Generation

This repository contains the demo page and audio samples for our paper "An Octave-based Multi-Resolution CQT Architecture for Diffusion-based Audio Generation".

---

## Listen to Audio Samples

Visit our [demo page](https://eloimoliner.github.io/MR-CQTdiff/) to listen to unconditional audio generation samples comparing different diffusion models:

- **UNet-1D**: A 1-dimensional U-Net composed of temporal convolutions, similar to architectures used in waveform-domain diffusion
- **NCSN++**: A 2-dimensional U-Net operating on STFT representations, using 2D convolutions over time and frequency axes
- **CQTDiff+**: A baseline model that uses a differentiable and invertible CQT representation combined with a U-Net architecture
- **MR-CQTDiff**: Our proposed model, which extends CQTdiff+ by introducing a multi-resolution CQT filter bank
- **LDM**: Latent Diffusion Model

All architectures are configured to share a similar parameter count of around 40 million parameters.

---

## Code Organization

```

src/
├── conf/                  # YAML configs for models, datasets, samplers
├── datasets/             # Toy dataset loaders
├── diff\_params/          # Diffusion schedules (e.g. EDM)
├── networks/             # Implementations of MR-CQTdiff, CQTdiff+, NCSN++
├── outputs/              # Logs and sample outputs
├── testing/              # Sampler (Euler-Heun) and testing scripts
├── utils/                # Logging, CQT utilities, training tools
samples/                  # Generated WAV files from each model and dataset

````

---

## Training

Edit and run the training script:

```bash
bash train.sh
````

Or manually:

```bash
python train.py --config-name=conf_FMA.yaml \
  network=mr_cqtdiff_44k \
  dset=toy_dataset_FMA \
  diff_params=edm \
  exp=44kHz_6s \
  logging=base_logging
```

This will save logs and checkpoints in:

```bash
src/experiments/<experiment_name>/
```

---

##  Inference (Unconditional Generation)

To generate samples using a trained model:

```bash
bash test.sh
```

Or run manually:

```bash
python test_unconditional.py --config-name=conf_OpenSinger.yaml \
  model_dir=experiments/test \
  tester=unconditional \
  tester.checkpoint=../checkpoints/OpenSinger_MRcqtdiff_500k.pt \
  tester.overriden_name="2D_OpenSinger_HD_CQT_waveform_IS2_100k_v2" \
  tester.unconditional.num_samples=4
```

Generated `.wav` files will appear in:

```bash
samples/OpenSinger/MR-CQTdiff/
```

---

##  Model Architectures

All models are defined in `src/networks/`:

* `mr_cqtdiff.py`: Multi-resolution CQT U-Net (proposed)
* `cqtdiff+.py`: CQT-based baseline
* `ncsnpp.py`: STFT-based U-Net (NCSN++)

Sampler implementation:

* `SamplerEulerHeun.py`

---

##  Configuration

Config files (Hydra-compatible) are stored in `src/conf/`:

* `conf/network/`: Model architectures
* `conf/dset/`: Dataset settings
* `conf/exp/`: Sampling rate, audio length
* `conf/tester/`: Test modes (e.g., unconditional)
* `conf/diff_params/`: Diffusion schedule (EDM)

Use `--config-name=<name>.yaml` and override other values from the command line.

---

## Citation

If you use this work, please cite:

```bibtex
@article{costa2025mrcqtdiff,
  title={An Octave-based Multi-Resolution CQT Architecture for Diffusion-based Audio Generation},
  author={Costa, Maurício do V. M. da and Moliner, Eloi},
  year={2025}
}
```

## Authors

- **Maurício do V. M. da Costa** - MTDML, IMM, University of Osnabrück, Germany
- **Eloi Moliner** - Acoustics Lab, Aalto University, Finland

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

Got it. Here's the updated `README.md` version with the **environment setup section removed**, and everything else kept clean and ready for your repo:

