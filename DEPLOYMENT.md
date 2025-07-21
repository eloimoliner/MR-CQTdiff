# Deployment Guide for MR-CQTdiff Demo Page

## Quick Setup for GitHub Pages

### Step 1: Push to GitHub
Make sure all your files are committed and pushed to your GitHub repository:

```bash
git add .
git commit -m "Add demo page with audio samples"
git push origin main
```

### Step 2: Enable GitHub Pages
1. Go to your GitHub repository: `https://github.com/eloimoliner/MR-CQTdiff`
2. Click on "Settings" tab
3. Scroll down to "Pages" in the left sidebar
4. Under "Source", select "Deploy from a branch"
5. Choose "main" branch and "/ (root)" folder
6. Click "Save"

### Step 3: Access Your Demo Page
- Your page will be live at: `https://eloimoliner.github.io/MR-CQTdiff/`
- It may take a few minutes for the page to become available
- GitHub will show a green checkmark when deployment is successful

## Local Development

### Testing Locally
To test the page locally before deployment:

```bash
# Navigate to your project directory
cd /path/to/MR-CQTdiff

# Start a simple HTTP server (Python 3)
python3 -m http.server 8000

# Or using Python 2
python -m SimpleHTTPServer 8000

# Or using Node.js (if you have it installed)
npx http-server

# Then open your browser to:
# http://localhost:8000
```

### File Structure Verification
Make sure your repository has this structure:
```
MR-CQTdiff/
├── index.html          # Main page (required)
├── styles.css          # Styling
├── samples/            # Audio files
│   ├── FMA/
│   └── OpenSinger/
├── README.md
└── LICENSE
```

## Troubleshooting

### Audio Files Not Playing
- **File paths**: Ensure audio file paths in HTML match actual folder structure
- **File formats**: GitHub Pages supports WAV, MP3, OGG formats
- **File size**: Large files (>100MB) might not work well; consider compression
- **HTTPS**: GitHub Pages uses HTTPS; make sure audio URLs are relative paths

### Page Not Loading
- Check that `index.html` exists in the root directory
- Verify GitHub Pages is enabled in repository settings
- Check the "Actions" tab for deployment status
- Clear browser cache and try again

### Custom Domain (Optional)
If you want to use a custom domain:
1. Add a `CNAME` file to the root with your domain name
2. Configure DNS settings with your domain provider
3. Update repository settings to use custom domain

## Performance Optimization

### Audio File Optimization
- Compress audio files to reduce loading times
- Consider using MP3 format for better compression
- Use `preload="metadata"` (already implemented) for faster initial loading

### Page Loading
- Images and audio files are loaded on-demand
- Mobile-responsive design ensures good performance on all devices
- Bootstrap CSS is loaded from CDN for better caching

## Security Notes
- All audio files are served directly from GitHub
- No server-side processing required
- HTTPS is enforced by GitHub Pages
- No sensitive data should be included in public repositories

## Updates and Maintenance
To update the demo page:
1. Make changes to HTML, CSS, or add new audio files
2. Commit and push changes to the main branch
3. GitHub Pages will automatically rebuild and deploy
4. Changes typically appear within 1-10 minutes
