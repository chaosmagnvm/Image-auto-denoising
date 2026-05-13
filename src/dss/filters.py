import numpy as np
import matplotlib.pyplot as plt
import cv2
import dss.detector as d


def denoiser_gaussian(img, std):

    def denoise(img_, std_, k_, sigma_): 
        
        denoised = cv2.fastNlMeansDenoising(img_, 0, std_, k_, 21)
        denoised = cv2.GaussianBlur(denoised, (0,0), sigma_)
        deonised = deblur(denoised, sigma_, 0)

        return denoised
    
    sigma = 0.5
    k = 3

    if std >= 28:
         k = 5
         sigma = 1

    denoised = denoise(img, std, k, sigma)
    noise = d.gaussian_noise_detector(denoised)
    return denoised


def denoiser_snp(img, std):
     
     k = 3
     # code
     if std > 30:
          k = 5

     denoised = cv2.medianBlur(img, k)
     
     return denoised


def deblur(img, b, auto):

    def gaussian_psf(shape, sigma, k=0):
        h, w = shape

        if k is None:
            k = int(2 * np.ceil(3 * sigma) + 1)
        if k % 2 == 0:
            k += 1

        g1 = cv2.getGaussianKernel(k, sigma)
        g2 = g1 @ g1.T
        g2 = g2.astype(np.float32)
        g2 /= g2.sum()

        psf = np.zeros((h, w), dtype=np.float32)

        cy, cx = h // 2, w // 2
        sy, sx = cy - k // 2, cx - k // 2

        psf[sy:sy + k, sx:sx + k] = g2

        psf = np.fft.ifftshift(psf)

        return psf


    def wiener_deblur_channel(channel, sigma, nsr=0.001, k=None):
        channel = channel.astype(np.float32)

        h, w = channel.shape
        psf = gaussian_psf((h, w), sigma, k)

        H = np.fft.fft2(psf)
        G = np.fft.fft2(channel)

        H_conj = np.conj(H)
        F_hat = (H_conj / (np.abs(H) ** 2 + nsr)) * G

        restored = np.fft.ifft2(F_hat)
        restored = np.real(restored)

        restored = np.clip(restored, 0, 255).astype(np.uint8)
        return restored

    nsr = 0.001

    if auto:
        sigma = 0.246 * b - 0.339
    else: 
        sigma = float(b)

    if img.ndim == 2:
        return wiener_deblur_channel(img, sigma, nsr)

    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = wiener_deblur_channel(img[:, :, c], sigma, nsr)
    return out


## noise generators

def gaussian_noise(img, std, mean=0.0):

        img_f = img.astype(np.float32)
        noise = np.random.normal(loc=mean, scale=std, size=img.shape).astype(np.float32)
        out = img_f + noise

        out = np.clip(out, 0, 255)

        return out.astype(np.uint8)


def snp_noise(img, p = 0.05):

        h, w = img.shape[:2]
        tmp = img.copy().reshape(-1, 3)
        n = int((h*w)*p)
        
        white_idx = np.random.randint(0, h*w, size = n)
        black_idx = np.random.randint(0, h*w, size = n)

        tmp[white_idx] = 255
        tmp[black_idx] = 0
        tmp = tmp.reshape(h, w, 3)
        return tmp