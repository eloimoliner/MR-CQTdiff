# MR-CQTdiff: An Octave-based Multi-Resolution CQT Architecture for Diffusion-based Audio Generation

This repository contains the demo page and audio samples for our paper "An Octave-based Multi-Resolution CQT Architecture for Diffusion-based Audio Generation".

## 🎵 Listen to Audio Samples

Visit our [demo page](https://eloimoliner.github.io/MR-CQTdiff/) to listen to unconditional audio generation samples comparing different diffusion models:

- **UNet-1D**: Baseline 1D U-Net approach
- **NCSN++**: Score-based generative model
- **CQTDiff+**: CQT-based diffusion model  
- **MR-CQTDiff**: Our proposed multi-resolution CQT approach
- **LDM**: Latent Diffusion Model

## 📁 Repository Structure

```
├── index.html          # Main demo page
├── styles.css          # Styling for the demo page
├── samples/            # Audio samples organized by dataset and model
│   ├── FMA/            # Free Music Archive samples
│   │   ├── 1D/         # UNet-1D generated samples
│   │   ├── NCSN++/     # NCSN++ generated samples
│   │   ├── CQTdiff+/   # CQTdiff+ generated samples
│   │   ├── MR-CQTdiff/ # Our method's generated samples
│   │   └── LDM/        # LDM generated samples
│   └── OpenSinger/     # OpenSinger dataset samples
│       ├── 1D/         
│       ├── NCSN++/     
│       ├── CQTdiff+/   
│       ├── MR-CQTdiff/ 
│       └── LDM/        
└── README.md           # This file
```

## 🚀 Setting up GitHub Pages

To set up this demo page on GitHub Pages:

1. **Enable GitHub Pages**:
   - Go to your repository settings
   - Scroll down to "Pages" section
   - Select "Deploy from a branch" as source
   - Choose "main" branch and "/ (root)" folder
   - Click "Save"

2. **Access your demo page**:
   - Your page will be available at: `https://[username].github.io/[repository-name]/`
   - For this repository: `https://eloimoliner.github.io/MR-CQTdiff/`

3. **Customize the page**:
   - Edit `index.html` to modify content
   - Edit `styles.css` to change styling
   - Add more audio samples to the `samples/` folder

## 🎧 Audio Playback Features

The demo page includes:
- ✅ HTML5 audio controls for all samples
- ✅ Responsive design for mobile and desktop
- ✅ Automatic pause of other audio when one starts playing
- ✅ Keyboard navigation support (Space to play/pause)
- ✅ Visual feedback for loading states
- ✅ Highlighted proposed method column
- ✅ Organized comparison tables by dataset

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{costa2024mrcqtdiff,
  title={An Octave-based Multi-Resolution CQT Architecture for Diffusion-based Audio Generation},
  author={Costa, Maurício do V. M. da and Moliner, Eloi},
  year={2024}
}
```

## 👥 Authors

- **Maurício do V. M. da Costa** - MTDML, IMM, University of Osnabrück, Germany
- **Eloi Moliner** - Acoustics Lab, Aalto University, Finland

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.