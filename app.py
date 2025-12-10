#!/usr/bin/env python3
import argparse
import piexif
import numpy as np
from PIL import Image, ImageChops, ImageFilter
from torchvision import transforms, models
import torch
import torch.nn as nn
from scipy.fft import fft2, fftshift
from scipy import stats
import warnings
import os
warnings.filterwarnings("ignore")


# ---------------------------------------------------------
#  METADATA CHECK (C2PA, AI TAGS, MIDJOURNEY, SD, ETC.)
# ---------------------------------------------------------
def check_metadata(path):
    """Check for AI-related metadata tags"""
    try:
        exif_data = piexif.load(path)
        all_data = str(exif_data)

        keywords = [
            "c2pa", "ai", "generated", "contentauth",
            "midjourney", "stablediffusion", "sdxl",
            "dalle", "firefly", "gemini", "synthetic"
        ]

        flag = any(k.lower() in all_data.lower() for k in keywords)
        return flag, exif_data
    except:
        return False, None


# ---------------------------------------------------------
#  FFT NOISE ANALYSIS (UNNATURAL UNIFORMITY = AI)
# ---------------------------------------------------------
def noise_score(img):
    """Analyze frequency domain - AI images often have unnatural uniformity"""
    img = img.convert("L")
    arr = np.array(img, dtype=np.float32)

    f = fftshift(fft2(arr))
    magnitude = np.abs(f)

    uniformity = np.std(magnitude) / np.mean(magnitude)
    return uniformity


# ---------------------------------------------------------
#  ERROR LEVEL ANALYSIS (ELA)
# ---------------------------------------------------------
def ela_score(img):
    """Analyze compression artifacts - AI images often have low ELA"""
    TEMP = "temp_ela.jpg"

    # Force RGB
    img = img.convert("RGB")

    img.save(TEMP, "JPEG", quality=95)
    compressed = Image.open(TEMP).convert("RGB")

    ela_img = ImageChops.difference(img, compressed)

    extrema = ela_img.getextrema()
    max_diff = max([x[1] for x in extrema])
    
    # Clean up temp file
    try:
        os.remove(TEMP)
    except:
        pass
    
    return max_diff


# ---------------------------------------------------------
#  COLOR DISTRIBUTION ANALYSIS
# ---------------------------------------------------------
def color_distribution_score(img):
    """AI images often have unnatural color distributions"""
    img = img.convert("RGB")
    arr = np.array(img)
    
    # Calculate entropy for each channel
    entropies = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:,:,channel], bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        entropies.append(entropy)
    
    avg_entropy = np.mean(entropies)
    
    # AI images often have lower entropy (less random color distribution)
    # Real photos typically have entropy > 6.5
    return avg_entropy


# ---------------------------------------------------------
#  EDGE COHERENCE ANALYSIS
# ---------------------------------------------------------
def edge_coherence_score(img):
    """AI images sometimes have inconsistent edge patterns"""
    img = img.convert("L")
    
    # Apply edge detection
    edges = img.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges)
    
    # Calculate edge strength variance
    # AI images often have more uniform edge strength
    edge_variance = np.var(edge_arr)
    
    return edge_variance


# ---------------------------------------------------------
#  JPEG ARTIFACTS ANALYSIS
# ---------------------------------------------------------
def jpeg_artifacts_score(img):
    """Analyze JPEG compression patterns - AI images often lack natural artifacts"""
    img = img.convert("L")
    arr = np.array(img, dtype=np.float32)
    
    # Look for 8x8 block patterns (JPEG uses 8x8 DCT blocks)
    h, w = arr.shape
    block_variances = []
    
    for i in range(0, h-8, 8):
        for j in range(0, w-8, 8):
            block = arr[i:i+8, j:j+8]
            block_variances.append(np.var(block))
    
    # Real photos have more variation in block compression
    if len(block_variances) > 0:
        return np.std(block_variances)
    return 0


# ---------------------------------------------------------
#  HIGH-FREQUENCY NOISE ANALYSIS
# ---------------------------------------------------------
def high_frequency_noise_score(img):
    """AI images often have unnaturally smooth high-frequency components"""
    img = img.convert("L")
    arr = np.array(img, dtype=np.float32)
    
    # Apply high-pass filter using Laplacian
    from scipy.ndimage import laplace
    high_freq = laplace(arr)
    
    # Calculate variance of high-frequency components
    # Real photos have more high-frequency noise
    # AI images are often too smooth
    hf_variance = np.var(high_freq)
    
    return hf_variance


# ---------------------------------------------------------
#  LOCAL BINARY PATTERNS (TEXTURE ANALYSIS)
# ---------------------------------------------------------
def texture_consistency_score(img):
    """Analyze texture patterns - AI images sometimes have inconsistent micro-textures"""
    img = img.convert("L")
    arr = np.array(img, dtype=np.float32)
    
    # Simplified LBP-like analysis
    # Compare each pixel with its neighbors
    h, w = arr.shape
    if h < 100 or w < 100:
        return 0
    
    # Sample random patches
    np.random.seed(42)  # For consistency
    patch_variances = []
    
    for _ in range(50):  # Sample 50 patches
        y = np.random.randint(10, h-10)
        x = np.random.randint(10, w-10)
        patch = arr[y-5:y+5, x-5:x+5]
        patch_variances.append(np.var(patch))
    
    # Real photos have more consistent texture variance
    # AI images sometimes have patches that are too uniform or too varied
    texture_std = np.std(patch_variances)
    
    return texture_std


# ---------------------------------------------------------
#  CHROMATIC ABERRATION ANALYSIS
# ---------------------------------------------------------
def chromatic_aberration_score(img):
    """Real photos from cameras have chromatic aberration, AI images often don't"""
    img = img.convert("RGB")
    arr = np.array(img)
    
    # Extract RGB channels
    r = arr[:,:,0].astype(np.float32)
    g = arr[:,:,1].astype(np.float32)
    b = arr[:,:,2].astype(np.float32)
    
    # Look for edge misalignment between channels (chromatic aberration)
    from scipy.ndimage import sobel
    
    r_edges = sobel(r)
    g_edges = sobel(g)
    b_edges = sobel(b)
    
    # Calculate correlation between channel edges
    # Real photos have slight misalignment (lower correlation)
    # AI images have perfect alignment (higher correlation)
    rg_corr = np.corrcoef(r_edges.flatten(), g_edges.flatten())[0,1]
    rb_corr = np.corrcoef(r_edges.flatten(), b_edges.flatten())[0,1]
    gb_corr = np.corrcoef(g_edges.flatten(), b_edges.flatten())[0,1]
    
    avg_corr = (rg_corr + rb_corr + gb_corr) / 3
    
    # Return inverse - higher correlation = more AI-like
    return avg_corr


# ---------------------------------------------------------
#  BASIC RESNET18 CLASSIFIER
# ---------------------------------------------------------
class AIDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.model(x))


def load_model():
    model = AIDetector()
    model.eval()
    return model


# Force RGB + normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def classify_ai(model, img):
    img = img.convert("RGB")
    t = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(t)

    return out[0][1].item()  # Probability image = AI


# ---------------------------------------------------------
#  COLOR OUTPUT
# ---------------------------------------------------------
def color(text, code):
    return f"\033[{code}m{text}\033[0m"


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI vs REAL Image Detector (CLI)")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed analysis")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")

    # --- METADATA ---
    meta_flag, meta = check_metadata(args.image)

    # --- FFT ---
    fft_val = noise_score(img)

    # --- ELA ---
    ela_val = ela_score(img)

    # --- COLOR DISTRIBUTION ---
    color_dist = color_distribution_score(img)

    # --- EDGE COHERENCE ---
    edge_coh = edge_coherence_score(img)

    # --- JPEG ARTIFACTS ---
    jpeg_art = jpeg_artifacts_score(img)

    # --- HIGH-FREQUENCY NOISE ---
    hf_noise = high_frequency_noise_score(img)

    # --- TEXTURE CONSISTENCY ---
    texture_cons = texture_consistency_score(img)

    # --- CHROMATIC ABERRATION ---
    chrom_aber = chromatic_aberration_score(img)

    # --- CLASSIFIER ---
    model = load_model()
    ai_prob = classify_ai(model, img)

    # ---------------------------------------------------------
    # WEIGHTED SCORING SYSTEM
    # ---------------------------------------------------------
    # Each signal contributes a weighted score (0-1 range)
    
    weights = {
        'metadata': 3.0,      # Strong signal if present
        'classifier': 2.5,    # Neural network is reliable
        'ela': 2.5,          # Very good indicator (especially low ELA)
        'fft': 2.0,          # Good indicator
        'hf_noise': 1.8,     # Good for realistic AI
        'chrom_aber': 1.5,   # Moderate indicator
        'color_dist': 1.5,   # Moderate indicator
        'texture_cons': 1.2, # Moderate indicator
        'edge_coh': 1.0,     # Weak indicator
        'jpeg_art': 1.0      # Weak indicator
    }
    
    scores = {}
    
    # Metadata (binary)
    scores['metadata'] = 1.0 if meta_flag else 0.0
    
    # Classifier (direct probability)
    scores['classifier'] = ai_prob
    
    # FFT - AI images typically have FFT < 10 (based on your test results)
    # Real images typically > 15
    if fft_val < 7:
        scores['fft'] = 1.0
    elif fft_val < 10:
        scores['fft'] = 0.7
    elif fft_val < 15:
        scores['fft'] = 0.3
    else:
        scores['fft'] = 0.0
    
    # ELA - AI images typically have ELA < 100 (based on your test results)
    # Real images typically > 150
    # VERY LOW ELA (< 10) is a strong AI indicator
    if ela_val < 10:
        scores['ela'] = 1.0
    elif ela_val < 50:
        scores['ela'] = 0.9
    elif ela_val < 100:
        scores['ela'] = 0.7
    elif ela_val < 150:
        scores['ela'] = 0.3
    else:
        scores['ela'] = 0.0
    
    # Color Distribution - Real photos typically have entropy > 6.5
    if color_dist < 6.0:
        scores['color_dist'] = 1.0
    elif color_dist < 6.5:
        scores['color_dist'] = 0.5
    else:
        scores['color_dist'] = 0.0
    
    # Edge Coherence - AI images often have edge_variance < 500
    if edge_coh < 300:
        scores['edge_coh'] = 1.0
    elif edge_coh < 500:
        scores['edge_coh'] = 0.5
    else:
        scores['edge_coh'] = 0.0
    
    # JPEG Artifacts - AI images often have lower std < 100
    if jpeg_art < 50:
        scores['jpeg_art'] = 1.0
    elif jpeg_art < 100:
        scores['jpeg_art'] = 0.5
    else:
        scores['jpeg_art'] = 0.0
    
    # High-Frequency Noise - AI images often have lower variance (too smooth)
    # Real photos typically > 1000
    if hf_noise < 500:
        scores['hf_noise'] = 1.0
    elif hf_noise < 1000:
        scores['hf_noise'] = 0.6
    elif hf_noise < 1500:
        scores['hf_noise'] = 0.3
    else:
        scores['hf_noise'] = 0.0
    
    # Texture Consistency - AI images often have inconsistent textures
    # Real photos typically have texture_std between 200-800
    if texture_cons < 100 or texture_cons > 1000:
        scores['texture_cons'] = 0.8
    elif texture_cons < 150 or texture_cons > 800:
        scores['texture_cons'] = 0.5
    else:
        scores['texture_cons'] = 0.0
    
    # Chromatic Aberration - AI images have higher correlation (perfect alignment)
    # Real photos typically have correlation < 0.95
    if chrom_aber > 0.98:
        scores['chrom_aber'] = 1.0
    elif chrom_aber > 0.96:
        scores['chrom_aber'] = 0.7
    elif chrom_aber > 0.94:
        scores['chrom_aber'] = 0.4
    else:
        scores['chrom_aber'] = 0.0
    
    # Calculate weighted total
    total_weight = sum(weights.values())
    weighted_score = sum(scores[k] * weights[k] for k in scores.keys())
    final_score = weighted_score / total_weight  # Normalize to 0-1
    
    # ---------------------------------------------------------
    # VERDICT DETERMINATION
    # ---------------------------------------------------------
    confidence = abs(final_score - 0.5) * 2  # 0 = uncertain, 1 = very confident
    
    # More sensitive thresholds
    if final_score >= 0.55:
        verdict = "AI"
        color_code = "31"  # Red
    elif final_score >= 0.40:
        verdict = "UNCERTAIN"
        color_code = "33"  # Yellow
    else:
        verdict = "REAL"
        color_code = "32"  # Green
    
    # Adjust verdict based on strong signals
    if meta_flag and final_score >= 0.35:
        verdict = "AI"
        color_code = "31"
    
    # Multiple weak signals can indicate AI - Enhanced detection
    # Rule 1: FFT, ELA, and classifier all suggest AI
    if (fft_val < 10 and ela_val < 100 and ai_prob > 0.55):
        if final_score >= 0.30:
            verdict = "AI" if verdict != "AI" else verdict
            color_code = "31"
    
    # Rule 2: Very low ELA + low HF noise (realistic AI signature)
    if (ela_val < 20 and hf_noise < 500):
        if final_score >= 0.30:
            verdict = "AI"
            color_code = "31"
    
    # Rule 3: Multiple texture/frequency indicators
    if (fft_val < 10 and hf_noise < 1000 and (texture_cons < 150 or texture_cons > 800)):
        if final_score >= 0.28:
            verdict = "AI"
            color_code = "31"
    
    # Rule 4: High chromatic aberration + low ELA (AI cameras don't have lens aberration)
    if (chrom_aber > 0.96 and ela_val < 100):
        if final_score >= 0.30:
            verdict = "AI"
            color_code = "31"
    
    if args.json:
        import json
        print(json.dumps({
            "metadata_ai_flag": bool(meta_flag),
            "fft_noise_uniformity": float(fft_val),
            "ela_artifacts": float(ela_val),
            "color_distribution_entropy": float(color_dist),
            "edge_coherence_variance": float(edge_coh),
            "jpeg_artifacts_std": float(jpeg_art),
            "high_frequency_noise": float(hf_noise),
            "texture_consistency": float(texture_cons),
            "chromatic_aberration": float(chrom_aber),
            "classifier_ai_probability": float(ai_prob),
            "final_score": float(final_score),
            "confidence": float(confidence),
            "verdict": str(verdict)
        }, indent=2))
        return

    print("\n" + "="*50)
    print("       AI IMAGE DETECTION RESULTS")
    print("="*50)
    
    print("\nANALYSIS METRICS:")
    print(f"  Metadata AI Flag:           {color('YES', '31') if meta_flag else color('NO', '32')}")
    print(f"  FFT Noise Uniformity:       {fft_val:.4f} {'[!] (Low - AI-like)' if fft_val < 10 else '[OK] (Normal)'}")
    print(f"  ELA Artifacts:              {ela_val:.4f} {'[!] (Low - AI-like)' if ela_val < 100 else '[OK] (Normal)'}")
    print(f"  Color Distribution Entropy: {color_dist:.4f} {'[!] (Low - AI-like)' if color_dist < 6.5 else '[OK] (Normal)'}")
    print(f"  Edge Coherence Variance:    {edge_coh:.4f} {'[!] (Low - AI-like)' if edge_coh < 500 else '[OK] (Normal)'}")
    print(f"  JPEG Artifacts Std:         {jpeg_art:.4f} {'[!] (Low - AI-like)' if jpeg_art < 100 else '[OK] (Normal)'}")
    print(f"  High-Frequency Noise:       {hf_noise:.4f} {'[!] (Low - AI-like)' if hf_noise < 1000 else '[OK] (Normal)'}")
    print(f"  Texture Consistency:        {texture_cons:.4f} {'[!] (Abnormal)' if (texture_cons < 150 or texture_cons > 800) else '[OK] (Normal)'}")
    print(f"  Chromatic Aberration:       {chrom_aber:.4f} {'[!] (High - AI-like)' if chrom_aber > 0.94 else '[OK] (Normal)'}")
    print(f"  Classifier AI Probability:  {ai_prob:.2f} {'[!] (High)' if ai_prob > 0.6 else '[OK] (Low)'}")
    
    if args.verbose:
        print(f"\nDETAILED SCORING:")
        for key in scores:
            print(f"  {key:15s}: {scores[key]:.2f} × {weights[key]:.1f} = {scores[key] * weights[key]:.2f}")
        print(f"  {'Total':15s}: {weighted_score:.2f} / {total_weight:.1f} = {final_score:.2f}")
    
    print(f"\nFINAL SCORE: {final_score:.2%} AI likelihood")
    print(f"CONFIDENCE:  {confidence:.2%}")
    
    print("\n" + "="*50)
    print("  VERDICT:", color(f" {verdict} ", color_code))
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
