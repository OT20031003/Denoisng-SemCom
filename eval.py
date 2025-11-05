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
            # print(f"SNR: {snr_float} dB, Average {metric.upper()}: {avg:.6f} (count={count})") # 冗長なのでコメントアウト
        except (ValueError, ZeroDivisionError):
            print(f"Warning: Could not process SNR key '{snr_key}'. Skipping.")
            continue
    
    # 処理の最後にサマリーを表示
    for snr_float, avg in sorted(xy):
         print(f"  SNR: {snr_float: <5} dB, Average {metric.upper()}: {avg:.6f} (count={dic_num.get(str(snr_float)+'dB', dic_num.get(str(snr_float)))})")


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
            print(f"Warning: No data to plot for label '{label}'. Skipping.")
            continue
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=label)
    
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel(f"Metric value {title_suffix}", fontsize=12)
    plt.title(f"SNR vs. Metric Comparison {title_suffix}", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"\nPlot saved as '{output_filename}'.")

def main():
    parser = argparse.ArgumentParser(description="SNR vs SSIM/MSE/PSNR/LPIPS comparison script")
    parser.add_argument("--sent", "-s", default="./sentimg", help="Directory for 'sent' (original) images")
    
    # --- ⭐ 変更点: 固定の --recv* を削除 ---
    # parser.add_argument("--recv", "-r", default="./outputs/k2=0.0", help="Directory for 'received' images (comparison target 1)")
    # parser.add_argument("--recv2", "-r2", default="./outputs/k=0.0/starttimestep=200", help="Directory for 'received' images (comparison target 2)")
    # parser.add_argument("--recv3", "-r3", default="./outputs/k=0.0/starttimestep=300", help="Directory for 'received' images (comparison target 3, optional)")
    # parser.add_argument("--recv4", "-r4", default="./outputs/k=0.0/starttimestep=350", help="Directory for 'received' images (comparison target 4, optional)")
    
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips","all"], default="ssim", help="Metric to use (ssim, mse, psnr, lpips, or all)")
    parser.add_argument("--resize", type=int, nargs=2, metavar=('W','H'), default=(256,256), help="Resize dimensions for comparison (W H)")

    # --- ⭐ 変更点: 任意の数の「受信ディレクトリ」を位置引数として受け取る ---
    # nargs='+' は「1つ以上」を意味します
    parser.add_argument("recv_dirs", 
                        nargs='+', 
                        metavar='RECV_DIR', 
                        help="One or more directories for 'received' images to compare against 'sent'")

    args = parser.parse_args()

    # --- ⭐ 変更点: recv_dirs リストからラベルを動的に生成 ---
    recv_paths = args.recv_dirs
    labels = [os.path.basename(os.path.normpath(p)) for p in recv_paths]
    recv_targets = list(zip(recv_paths, labels)) # (path, label) のタプルのリストを作成
    
    if not recv_targets:
        print("Error: No received directories specified.")
        return

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


    # --- ⭐ 変更点: ループ構造を変更 ---
    # メトリクスごとにループし、各メトリクス内で全ディレクトリを処理
    # これにより、'all' の場合にメトリクスごとにグラフが分かれる
    
    for metric in metrics_to_run:
        print(f"\n==========================================")
        print(f" PROCESSING METRIC: {metric.upper()} ")
        print(f"==========================================")
        
        metric_results = [] # このメトリクスの結果を格納 (x, y, label)

        # --- 指定された各受信ディレクトリに対して計算を実行 ---
        for recv_path, label in recv_targets:
            print(f"\n--- Calculating {metric.upper()} for {label} ---")
            
            x_vals, y_vals = calculate_snr_vs_metric(
                args.sent, recv_path, metric=metric, resize=tuple(args.resize), 
                lpips_model=lpips_model, device=device
            )
            
            if x_vals: # データが正常に取得できた場合のみ追加
                metric_results.append((x_vals, y_vals, label))
            else:
                print(f"Warning: No valid data found for {label} with metric {metric}.")

        # --- このメトリクスの全ディレクトリの処理が完了したら、プロット ---
        if not metric_results:
            print(f"\nNo data to plot for metric '{metric}'.")
            continue

        # メトリクスごとに別々のファイル名で保存
        outname = f"snr_vs_{metric}_comparison.png"
        plot_results(metric_results, title_suffix=f"({metric.upper()})", output_filename=outname)

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()

"""
python eval.py --sent ./sentimg --metric lpips ./outputs/k2=0.0 ./outputs/k=0.0/starttimestep=200 ./outputs/k=0.0/starttimestep=300 ./outputs/k=0.0/starttimestep=350 ./outputs/k=0.0/starttimestep=400 ./outputs/k=0.0/starttimestep=450



"""