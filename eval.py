import os
from skimage.metrics import structural_similarity as ssim

def MSE(x, y):
    return ((x - y) ** 2).mean()

def ssim_crite(x, y):
    return ssim(x, y, channel_axis=-1)
def main():
    cwd = os.getcwd()
    print("Current working directory:", cwd)
    path1 = "./sentimg" # 送信画像dirのpath
    path2 = "./outputs/LDPC" # 受信画像dirのpath
    dic_sum = {} #{snr : crite sum}
    dic_num = {} #{snr : num}
    for sp in os.listdir(path1):
        img_id = ""
        for i in range(len(sp)):
            if sp[i].isdigit():
                img_id += sp[i]
        for rp in os.listdir(path2):
            c = 0
            rimg_id = ""
            snr = ""
            for j in range(len(rp)):
                if rp[j] == '_':
                    c += 1
                if c == 1:
                    # snr check
                    if (rp[j].isdigit() or rp[j] == '-' or rp[j] == '.'):
                        snr += rp[j]
                if c == 2:
                        # img_id check
                        if rp[j].isdigit():
                            rimg_id += rp[j]
                if rp[j] == '.':
                    break

            if rimg_id == img_id:
                #print("match! rimg_id:", rimg_id, "snr : ", snr)
                # MSE計算
                from PIL import Image
                import numpy as np
                
                sentimg = np.array(Image.open(os.path.join(path1, sp)).convert('RGB').resize((256, 256)))
                recimg = np.array(Image.open(os.path.join(path2, rp)).convert('RGB').resize((256, 256)))
                mse = ssim_crite(sentimg, recimg) # 評価指標
                dic_sum[snr] = dic_sum.get(snr, 0) + mse
                dic_num[snr] = dic_num.get(snr, 0) + 1
                #print("MSE:", mse)
    print("dic_sum:", dic_sum)
    print("dic_num:", dic_num)
    xy = []
    x = []
    y = []
    for k, v in dic_sum.items():
        kk = k
        for i in range(len(k)):
            if k[i] == '-':
                kk = kk[:i] + '.' + kk[i+1:]
                break
        snr = float(kk)
        xy.append((snr, v/dic_num[k]))
        
        print("SNR:", kk, "Average MSE:", v / dic_num[k])
    xy = sorted(xy)
    print(xy)
    for xx, yy in xy:
        x.append(xx)
        y.append(yy)
    # グラフ描画

    import matplotlib.pyplot as plt
    plt.plot(x, y, marker='o')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average SSIM")
    plt.title("SNR vs Average SSIM (LDPC)")
    plt.savefig("snr_vs_ssim_LDPC.png")
    plt.show()
    
    
if __name__ == "__main__":
    main()