import os
import argparse
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re # ファイル名からステップ数を抽出するために使用

# --- LPIPS Imports (変更なし) ---
try:
    import torch
    import lpips
except ImportError:
    print("Warning: 'torch' or 'lpips' libraries not found.")
    print("To use the LPIPS metric, please install them: pip install torch lpips")
    torch = None
    lpips = None
# -------------------------------

# --- np_to_torch, compute_metric (変更なし) ---
def np_to_torch(img_np):
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor / 127.5) - 1.0
    return img_tensor

def compute_metric(x, y, metric='ssim', lpips_model=None, device=None):
    if metric == 'ssim':
        data_range = float(x.max() - x.min())
        if data_range == 0:
            return 1.0
        return ssim(x, y, channel_axis=-1, data_range=data_range)

    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    mse = float(np.mean((xd - yd) ** 2))

    if metric == 'mse':
        return mse
    
    elif metric == 'psnr':
        if mse == 0:
            return np.inf
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return float(psnr)
        
    elif metric == 'lpips':
        if lpips_model is None or device is None:
            raise ValueError("lpips_model and device must be provided for LPIPS metric.")
        tensor_x = np_to_torch(x).to(device)
        tensor_y = np_to_torch(y).to(device)
        with torch.no_grad():
            dist = lpips_model(tensor_x, tensor_y)
        return float(dist.item())

    else:
        raise ValueError("Metric must be 'ssim', 'mse', 'psnr', or 'lpips'.")

def main():
    parser = argparse.ArgumentParser(description="SNR vs Metric (Multiple pairs)")
    parser.add_argument("--sent", "-s", default="./sentimg", help="Directory for 'sent' (original) images")
    
    parser.add_argument("--base_recv", "-r", default="./intermediate/k2=0.0", 
                        help="Base directory for 'received' images (e.g., ./intermediate/k=0.0)")
    
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips"], default="ssim", 
                        help="Metric to use (ssim, mse, psnr, lpips)") # 'all' は元のコードでも未対応のため除外
    
    parser.add_argument("--resize", type=int, nargs=2, metavar=('W','H'), default=(256,256), 
                        help="Resize dimensions for comparison (W H)")
    
    # --- 変更点: 複数の値を受け取れるように nargs='+' を追加 ---
    parser.add_argument("--snr",type=float, nargs='+', required=True, help="List of channel SNRs (e.g., -5 0 5)")
    parser.add_argument("--timestep", "-t",type=int, nargs='+', required=True, help="List of all timesteps (e.g., 300 500 1000)")
    # --- 変更点: 出力ファイル名用の引数を追加 ---
    parser.add_argument("--output", "-o", type=str, default=None, help="Output plot filename. If None, defaults to comparison_[metric].png")

    
    args = parser.parse_args()

    # --- 変更点: SNRとTimestepのペアの数をチェック ---
    if len(args.snr) != len(args.timestep):
        print(f"Error: The number of --snr values ({len(args.snr)}) must match the number of --timestep values ({len(args.timestep)}).")
        return
        
    snr_ts_pairs = list(zip(args.snr, args.timestep))

    lpips_model = None
    device = None
    # --- 変更点: LPIPSの初期化をループの外に移動 ---
    if "lpips" == args.metric:
        if lpips is None or torch is None:
            print("Error: LPIPS metric requested, but 'torch' or 'lpips' libraries are not installed.")
            print("Please run: pip install torch lpips")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing LPIPS model (AlexNet) on device: {device}")
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    
    # --- 変更点: sent_dir の読み込みをループの外に移動 ---
    sent_dir = {}
    print(f"\nScanning 'sent' directory: {args.sent}")
    for sd in os.listdir(args.sent):
        ss = sd.split(".")
        ss = ss[0].split("_")
        sent_dir[ss[1]] = os.path.join(args.sent, sd)
    
    # --- 変更点: グラフの初期化をループの外に移動 ---
    plt.figure(figsize=(10, 7)) # グラフサイズを少し大きめに設定

    # --- 変更点: (SNR, Timestep) のペアでループ ---
    for snr, timestep in snr_ts_pairs:
        print(f"\nProcessing: SNR = {snr}, TotalTimestep = {timestep}")
        
        # obj_dir の初期化は各ループで行う
        obj_dir = {}
        for k, v in sent_dir.items():
            obj_dir[k] = []
        
        print(f"Scanning {args.base_recv} for matching directories...")
        
        # --- 変更点: ループ変数 (snr, timestep) を使用 ---
        for snr_dir in os.listdir(args.base_recv):
            snr_path = os.path.join(args.base_recv, snr_dir)
            if not os.path.isdir(snr_path):
                continue
            
            try:
                # ディレクトリ名が数値（SNR）かチェック
                if snr != float(snr_dir):
                    continue
            except ValueError:
                print(f"Skipping non-numeric directory: {snr_dir}")
                continue

            for img_id in os.listdir(snr_path):
                img_id_path = os.path.join(snr_path, img_id)
                if not os.path.isdir(img_id_path) or img_id not in obj_dir:
                    continue # sent_dir にない画像IDはスキップ
                    
                for timestep_dir in os.listdir(img_id_path):
                    if not os.path.isdir(os.path.join(img_id_path, timestep_dir)):
                        continue
                        
                    try:
                        # ディレクトリ名が数値（Timestep）かチェック
                        if int(timestep_dir) != timestep:
                            continue
                    except ValueError:
                        print(f"Skipping non-numeric directory: {timestep_dir}")
                        continue
                        
                    obj_dir[img_id].append(os.path.join(img_id_path, timestep_dir))

        # ステップごとに計算
        ans = {} # {step:metric_sum}
        ans_num = {} 
        for k, v in obj_dir.items():
            for d in v:
                if not os.path.isdir(d):
                    continue
                for file_name in os.listdir(d):
                    step_match = re.match(r".*_(\d+)\.(png|jpg|jpeg|bmp|webp)$", file_name, re.IGNORECASE)
                    if not step_match:
                        continue # ファイル名がパターンに合致しない
                        
                    step = int(step_match.group(1))
                    
                    file_path = os.path.join(d, file_name)
                    
                    try:
                        # 画像のリサイズも考慮（元のコードにはresize引数があったが使われていなかったため追加）
                        sentimg = Image.open(sent_dir[k]).convert('RGB').resize(args.resize)
                        recimg = Image.open(file_path).convert('RGB').resize(args.resize)

                        sentarr = np.array(sentimg)
                        recarr = np.array(recimg)

                        val = compute_metric(sentarr, recarr, metric=args.metric,lpips_model=lpips_model, device=device)
                        
                        ans[step] = ans.get(step, 0.0) + val
                        ans_num[step] = ans_num.get(step, 0) + 1
                    except Exception as e:
                        print(f"Warning: Error processing {file_path}: {e}")
                        continue

        if not ans:
            print(f"Warning: No valid data found for SNR={snr}, Timestep={timestep}. Skipping this pair.")
            continue
            
        ave = {}
        for k, v in ans.items():
            ave[k] = v / ans_num[k]
        
        # 降順 (reverse=True) にソート
        ave = dict(sorted(ave.items(), key=lambda item: item[0], reverse=True))
        
        X = []
        Y = []
        for k, v in ave.items():
            X.append(int(k))
            Y.append(float(v))
            
        print(f"Data points for this pair (Step, Metric): {list(zip(X, Y))}")
        
        # --- 変更点: グラフに線を追加 ---
        plt.plot(X, Y, label=f"TotalTimestep = {timestep}, SNR = {snr}", marker='o', markersize=3)

    # --- 変更点: グラフの最終調整と保存をループの外で実行 ---
    
    plt.xlabel("Time Step")
    plt.grid(True)
    plt.legend() # 凡例を表示
    plt.ylabel(f"{args.metric}")
    plt.title(f"Time Step vs {args.metric}")
    plt.gca().invert_xaxis() # X軸を反転
    
    # 出力ファイル名を決定
    if args.output:
        output_filename = args.output
    else:
        output_filename = f"comparison_{args.metric}_samplingeval.png"
        
    plt.savefig(output_filename)
    print(f"\nCombined graph saved to: {output_filename}")

if __name__ == "__main__":
    main()
#  python samplingeval.py --snr 0 0 0 -t 200 400 600 -m psnr