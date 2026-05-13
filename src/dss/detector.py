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