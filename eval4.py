import os
import argparse
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- New Imports for LPIPS ---
try:
    import torch
    import lpips
except ImportError:
    print("Warning: 'torch' or 'lpips' libraries not found.")
    print("To use the LPIPS metric, please install them: pip install torch lpips")
    torch = None
    lpips = None
# -------------------------------


def np_to_torch(img_np):
    """
    Converts a NumPy image (H, W, C) in range [0, 255]
    to a PyTorch tensor (N, C, H, W) in range [-1, 1].
    """
    # From HWC to NCHW
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    # Normalize from [0, 255] to [-1, 1]
    img_tensor = (img_tensor / 127.5) - 1.0
    return img_tensor

def compute_metric(x, y, metric='ssim', lpips_model=None, device=None):
    """
    Computes the similarity/error between image pair x, y.
    metric: 'ssim', 'mse', 'psnr', or 'lpips'
    """
    if metric == 'ssim':
        # Assumes RGB images. Specify data_range for stability.
        # Ensure data_range is not zero if images are solid color
        data_range = float(x.max() - x.min())
        if data_range == 0:
            return 1.0 # Images are identical and flat
        return ssim(x, y, channel_axis=-1, data_range=data_range)

    # For MSE and PSNR, convert to float and calculate MSE
    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    mse = float(np.mean((xd - yd) ** 2))

    if metric == 'mse':
        return mse
    
    elif metric == 'psnr':
        if mse == 0:
            # Images are identical, PSNR is infinite
            return np.inf
        max_pixel = 255.0  # Assuming 8-bit images (0-255)
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return float(psnr)
        
    elif metric == 'lpips':
        if lpips_model is None or device is None:
            raise ValueError("lpips_model and device must be provided for LPIPS metric.")
        
        # Convert numpy arrays (HWC, 0-255) to torch tensors (NCHW, -1 to 1)
        tensor_x = np_to_torch(x).to(device)
        tensor_y = np_to_torch(y).to(device)
        
        # Calculate LPIPS
        with torch.no_grad(): # No need to track gradients
            dist = lpips_model(tensor_x, tensor_y)
        return float(dist.item())

    else:
        raise ValueError("Metric must be 'ssim', 'mse', 'psnr', or 'lpips'.")

def calculate_snr_vs_metric(sent_path, received_path, metric='ssim', resize=(256,256), lpips_model=None, device=None):
    """
    Compares images in the sent and received directories,
    returning the average metric value per SNR.
    metric: 'ssim', 'mse', 'psnr', or 'lpips'
    resize: Size to resize images to during comparison (None for no resize)
    """
    dic_sum = {}
    dic_num = {}

    if not os.path.isdir(sent_path):
        print(f"Error: Directory not found: {sent_path}")
        return [], []
    if not os.path.isdir(received_path):
        print(f"Error: Directory not found: {received_path}")
        return [], []

    print(f"Processing comparison between '{sent_path}' and '{received_path}'... (metric={metric})")

    for sp in os.listdir(sent_path):
        if not sp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        img_id = "".join(filter(str.isdigit, sp))
        if not img_id:
            continue

        sent_image_path = os.path.join(sent_path, sp)

        for rp in os.listdir(received_path):
            if not rp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            if img_id in rp:
                try:
                    parts = os.path.splitext(rp)[0].split('_')
                    if len(parts) < 2:
                        continue

                    rimg_id_part = "".join(filter(str.isdigit, parts[-1]))
                    snr_str = parts[-2]

                    if rimg_id_part == img_id:
                        sentimg = Image.open(sent_image_path).convert('RGB')
                        recimg = Image.open(os.path.join(received_path, rp)).convert('RGB')

                        if resize is not None:
                            sentimg = sentimg.resize(resize)
                            recimg = recimg.resize(resize)

                        sentarr = np.array(sentimg)
                        recarr = np.array(recimg)

                        # Robust metric calculation with exception handling
                        try:
                            # --- Pass lpips_model and device down ---
                            val = compute_metric(sentarr, recarr, metric=metric, lpips_model=lpips_model, device=device)
                        except Exception as e:
                            print(f"Warning: Error during metric calculation ({rp}): {e}")
                            continue

                        dic_sum[snr_str] = dic_sum.get(snr_str, 0.0) + val
                        dic_num[snr_str] = dic_num.get(snr_str, 0) + 1
                except Exception:
                    # Catch-all for file parsing errors, etc.
                    continue

    if not dic_sum:
        print(f"Warning: No matching images found in '{received_path}'. Check filename format.")
        return [], []

    xy = []
    for snr_key, total in dic_sum.items():
        try:
            snr_float = float("".join(filter(lambda c: c.isdigit() or c in '.-', snr_key)))
            count = dic_num[snr_key]
            avg = total / count
            xy.append((snr_float, avg))
            print(f"SNR: {snr_float} dB, Average {metric.upper()}: {avg:.6f} (count={count})")
        except (ValueError, ZeroDivisionError):
            print(f"Warning: Could not process SNR key '{snr_key}'. Skipping.")
            continue

    xy.sort()  # Sort by SNR
    x_vals = [item[0] for item in xy]
    y_vals = [item[1] for item in xy]
    return x_vals, y_vals

def plot_results(results, title_suffix="", output_filename="snr_vs_metric.png"):
    """
    Plots the results.
    results: list of tuples (x_vals, y_vals, label)
    """
    plt.figure(figsize=(10,6))
    for x_vals, y_vals, label in results:
        if not x_vals:
            continue
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=label)
    
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Metric value", fontsize=12)
    plt.title(f"SNR vs. Metric Comparison {title_suffix}", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"\nPlot saved as '{output_filename}'.")

def main():
    parser = argparse.ArgumentParser(description="SNR vs SSIM/MSE/PSNR/LPIPS comparison script")
    parser.add_argument("--sent", "-s", default="./sentimg", help="Directory for 'sent' (original) images")
    parser.add_argument("--recv", "-r", default="./outputs/nonoise/k=0.0/starttimestep=500", help="Directory for 'received' images (comparison target 1)")
    parser.add_argument("--recv2", "-r2", default="./outputs/nonoise/k=0.0/starttimestep=600", help="Directory for 'received' images (comparison target 2)")
    parser.add_argument("--recv3", "-r3", default="./outputs/nonoise/k=0.0/starttimestep=300", help="Directory for 'received' images (comparison target 3, optional)")
    # --- ⭐ 変更点: recv4 の引数を追加 ---
    parser.add_argument("--recv4", "-r4", default="./outputs/nonoise/k=0.0/starttimestep=400", help="Directory for 'received' images (comparison target 4, optional)")
    
    # --- Updated metric choices ---
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips","all"], default="ssim", help="Metric to use (ssim, mse, psnr, lpips, or all)")
    parser.add_argument("--resize", type=int, nargs=2, metavar=('W','H'), default=(256,256), help="Resize dimensions for comparison (W H)")
    args = parser.parse_args()

    # --- Get base directory names for labels ---
    label1 = os.path.basename(os.path.normpath(args.recv))
    label2 = os.path.basename(os.path.normpath(args.recv2))
    
    label3 = None
    if args.recv3:
        label3 = os.path.basename(os.path.normpath(args.recv3))
    
    # --- ⭐ 変更点: label4 の処理を追加 ---
    label4 = None
    if args.recv4:
        label4 = os.path.basename(os.path.normpath(args.recv4))
        
    # --- New logic to handle metric selection ---
    metrics_to_run = []
    if args.metric == "all":
        metrics_to_run = ["ssim", "mse", "psnr", "lpips"]
    else:
        metrics_to_run = [args.metric]

    # --- Initialize LPIPS model if needed ---
    lpips_model = None
    device = None
    if "lpips" in metrics_to_run:
        if lpips is None or torch is None:
            print("Error: LPIPS metric requested, but 'torch' or 'lpips' libraries are not installed.")
            print("Please run: pip install torch lpips")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing LPIPS model (AlexNet) on device: {device}")
        # Initialize the LPIPS model once. Using .eval() for inference mode.
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()


    results = []
    
    for metric in metrics_to_run:
        # --- Target 1 ---
        print(f"\n--- Calculating {metric.upper()} for {label1} ---")
        x1, y1 = calculate_snr_vs_metric(
            args.sent, args.recv, metric=metric, resize=tuple(args.resize), 
            lpips_model=lpips_model, device=device
        )
        results.append((x1, y1, f"{label1} - {metric.upper()}"))
        
        # --- Target 2 ---
        print(f"\n--- Calculating {metric.upper()} for {label2} ---")
        x2, y2 = calculate_snr_vs_metric(
            args.sent, args.recv2, metric=metric, resize=tuple(args.resize), 
            lpips_model=lpips_model, device=device
        )
        results.append((x2, y2, f"{label2} - {metric.upper()}"))

        # --- Target 3 (Only if provided) ---
        if args.recv3 and label3:
            print(f"\n--- Calculating {metric.upper()} for {label3} ---")
            x3, y3 = calculate_snr_vs_metric(
                args.sent, args.recv3, metric=metric, resize=tuple(args.resize), 
                lpips_model=lpips_model, device=device
            )
            results.append((x3, y3, f"{label3} - {metric.upper()}"))

        # --- ⭐ 変更点: Target 4 の処理を追加 (Only if provided) ---
        if args.recv4 and label4:
            print(f"\n--- Calculating {metric.upper()} for {label4} ---")
            x4, y4 = calculate_snr_vs_metric(
                args.sent, args.recv4, metric=metric, resize=tuple(args.resize), 
                lpips_model=lpips_model, device=device
            )
            results.append((x4, y4, f"{label4} - {metric.upper()}"))

    if not results:
        print("No data to plot.")
        return

    outname = f"snr_vs_{args.metric}_comparison.png"
    plot_results(results, title_suffix=f"({args.metric.upper()})", output_filename=outname)

if __name__ == "__main__":
    main()