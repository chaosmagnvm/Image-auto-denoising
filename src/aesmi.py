import numpy as np
import matplotlib.pyplot as plt
import cv2


def snp_noise_detector(img):

    def road_alghorithm(img):
            
        # ROAD algorithm
        gray = img.copy()
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        h, w = gray.shape[:2]

        Iup = gray.copy()
        Iup[:-1, :] = gray[1:, :]
        Iup[-1]=0
        Dup = np.absolute(gray-Iup)

        Idown = gray.copy()
        Idown[1:, :] = gray[:-1, :]
        Idown[0] = 0
        Ddown = np.absolute(gray-Idown)

        Il = gray.copy()
        Il[:,:-1] = gray[:, 1:]
        Il[:, -1] = 0
        Dl = np.absolute(gray-Il)

        Ir = gray.copy()
        Ir[:,1:] = gray[:,:-1]
        Ir[:, 0] = 0
        Dr = np.absolute(gray-Ir)

        Iul = gray.copy()
        Iul[:-1, :-1] = gray[1:, 1:]
        Iul[-1] = 0
        Iul[:, -1] = 0
        Dul = np.absolute(gray-Iul)

        Iur = gray.copy()
        Iur[:-1, 1:] = gray[1:, :-1]
        Iur[-1] = 0
        Iur[:, 0] = 0
        Dur = np.absolute(gray-Iur)

        Idl = gray.copy()
        Idl[1:, :-1] = gray[:-1, 1:]
        Idl[0] = 0
        Idl[:, -1] = 0
        Ddl = np.absolute(gray-Idl)

        Idr = gray.copy()
        Idr[1:, 1:] = gray[:-1, :-1]
        Idr[0] = 0
        Idr[:, 0] = 0
        Ddr = np.absolute(gray-Idr)

        tenz = np.array([Dup, Ddown, Dl, Dr, Dul, Dur, Ddl, Ddr]).reshape(8,h,w)
        tenz_sort = np.partition(tenz, 3, axis = 0) 
        road = np.sum(tenz_sort[:4], axis = 0)
        road = road[1:-1, 1:-1] 

        Sx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]], dtype = np.float32)
        
        Sy = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]], dtype = np.float32)
        
        Ix = cv2.filter2D(gray, -1, Sx, borderType=cv2.BORDER_DEFAULT)
        Iy = cv2.filter2D(gray, -1, Sy, borderType=cv2.BORDER_DEFAULT)

        M = np.sqrt(Ix*Ix+Iy*Iy) 
        M_sorted = M.copy().reshape(h*w)
        M_sorted.sort()
        i = int(np.absolute(0.5*(h-2)*(w-2))) 
        Tg = M_sorted[i]
        flat = M[1:-1, 1:-1] <= Tg 

        road_sorted = road.copy().reshape((h-2)*(w-2))
        road_sorted.sort()
        index = int(0.95*(h-2)*(w-2))
        T = road_sorted[index] 

        noise = road[(road>=T) & flat] 
        return noise
     
    len_src = len(road_alghorithm(img))
    clean3 = cv2.medianBlur(img, 3)
    len_clean3 = len(road_alghorithm(clean3))
    clean5 = cv2.medianBlur(img, 5)
    len_clean5 = len(road_alghorithm(clean5))

    h,w = img.shape[:2]

    p0 = 100*len_src/((h-2)*(w-2))
    p3 = 100*len_clean3/((h-2)*(w-2))
    p5 = 100*len_clean5/((h-2)*(w-2))

    return p0 * 10


def gaussian_noise_detector(img):
     
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    I = I.astype(np.float32)
    h,w = I.shape[:2]

    if h%2==1:
         h -= 1
    if w%2==1:
         w -= 1

    I = I[:h, :w] 

    Sx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]], dtype = np.float32)
        
    Sy = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]], dtype = np.float32)

    Ix = cv2.filter2D(I, -1, Sx, borderType=cv2.BORDER_DEFAULT)
    Iy = cv2.filter2D(I, -1, Sy, borderType=cv2.BORDER_DEFAULT)

    M = np.sqrt(Ix*Ix+Iy*Iy)
    M_sorted = M.copy().reshape(h*w)
    M_sorted.sort()
    index = int(np.absolute(h * w * 0.3))
    Mt = M_sorted[index]
    flat = M <= Mt

    if h%2==1:
         h -= 1
    if w%2==1:
         w -= 1

    tmp = I[:h, :w].copy()
    tmp_even = tmp[:, 0::2].copy()
    tmp_odd = tmp[:, 1::2].copy()
    tmp[:, :w//2] = (tmp_even + tmp_odd) * 0.5
    tmp[:, w//2:] = (tmp_even - tmp_odd) * 0.5

    h_temp = h
    w_temp = w

    if h%2==1:
         h -= 1
    if w%2==1:
         w -= 1

    L = tmp[:, :w//2].copy()
    R = tmp[:, w//2:].copy()

    tmp_even2 = L[0::2, :].copy()
    tmp_odd2 = L[1::2, :].copy()
    L[:h//2, :] = (tmp_even2 + tmp_odd2) * 0.5
    L[h//2:, :] = (tmp_even2 - tmp_odd2) * 0.5
    Ll = L[:h//2, :].copy()
    Hl = L[h//2:, :].copy()

    h = h_temp
    w = w_temp

    if h%2==1:
         h -= 1
    if w%2==1:
         w -= 1

    tmp_even2 = R[0::2, :].copy()
    tmp_odd2 = R[1::2, :].copy()
    R[:h//2, :] = (tmp_even2 + tmp_odd2) * 0.5
    R[h//2:, :] = (tmp_even2 - tmp_odd2) * 0.5
    Lh = R[:h//2, :].copy()
    Hh = R[h//2:, :].copy() 
    h, w = Hh.shape[:2]

    flat00 = flat[0::2, 0::2]
    flat01 = flat[0::2, 1::2]
    flat10 = flat[1::2, 0::2]
    flat11 = flat[1::2, 1::2]

    flat_mask = flat00 & flat01 & flat10 & flat11

    Hh = np.abs(Hh)
    med = np.median(Hh)
    med /= 0.6745
    med *= 2
    return med


def blur_detector(img):
    
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    I = I.astype(np.float32)
    h,w = I.shape[:2]

    Sx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]], dtype = np.float32)
        
    Sy = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]], dtype = np.float32)

    Ix = cv2.filter2D(I, -1, Sx, borderType=cv2.BORDER_DEFAULT)
    Iy = cv2.filter2D(I, -1, Sy, borderType=cv2.BORDER_DEFAULT)

    M = np.sqrt(Ix*Ix+Iy*Iy)
    M_sorted = M.copy().reshape(h*w)
    M_sorted.sort()
    index = int(np.absolute(h * w * 0.95))
    Mt = M_sorted[index]
    strong = M >= Mt
    vertical = np.abs(Ix) >= np.abs(Iy)
    mask = strong & vertical

    Ixl = Ix.copy()
    Ixl[:,:-1] = Ix[:, 1:]
    
    Ixr = Ix.copy()
    Ixr[:, 1:] = Ix[:, :-1]

    max_mask = (Ix>Ixr) & (Ix>Ixl)
    min_mask = (Ix<Ixr) & (Ix<Ixl)

    pos_mask = Ix > 0
    neg_mask = Ix < 0

    lmin = np.zeros((h,w))
    rmin = np.zeros((h,w))
    lmax = np.zeros((h,w)) 
    rmax = np.zeros((h,w)) 

    for i in range(h):
        lmin_tmp = -1
        lmax_tmp = -1
        for j in range(1, w-1):
                
                if min_mask[i, j] == 1:
                    lmin_tmp = j

                lmin[i, j] = lmin_tmp

                if max_mask[i, j] == 1:
                    lmax_tmp = j

                lmax[i, j] = lmax_tmp

    for i in range(h-1, -1, -1): 
        rmax_tmp = -1
        rmin_tmp = -1
        for j in range (w-2, 0, -1): 

            if max_mask[i, j] == 1:
                rmax_tmp = j 
            
            rmax[i, j] = rmax_tmp

            if min_mask[i, j] == 1:
                rmin_tmp = j 
            
            rmin[i, j] = rmin_tmp

    D = np.zeros((h,w))
    D[pos_mask] = rmax[pos_mask] - lmin[pos_mask]
    D[neg_mask] = rmin[neg_mask] - lmax[neg_mask]

    mask = strong & vertical & (D>0) & (D<20) 
    D = D[mask] 
    return np.median(D)


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
    noise = gaussian_noise_detector(denoised)
    return denoised


def denoiser_snp(img, std):
     
     k = 3
     # code
     if std > 30:
          k = 5

     denoised = cv2.medianBlur(img, k)
     
     return denoised


def deblur(img, b, auto=1):

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


def denoise(img):

    ret = img.copy()

    def decide_gaussian(g0, s0):
         
        if g0 <= 10:
            return -1
          
        tmp_g = denoiser_gaussian(img, g0)
        g_g = gaussian_noise_detector(tmp_g)
        g_s = snp_noise_detector(tmp_g)
        
        if g_g > g0 * (2/3) or g_s > s0 * (2/3):
             return -1
        
        return 1


    def decide_snp(g0, s0):
         
        if s0 <= 1:
            return -1
         
        tmp_s = denoiser_snp(img, s0)
        s_s = snp_noise_detector(tmp_s)
        s_g = gaussian_noise_detector(tmp_s)

        if s_s > s0/10 or s_s > s0/10: ## in progress
             return -1
        
        return 1