import numpy as np
import matplotlib.pyplot as plt
import cv2
import dss.detector as d
import dss.filters as f

def denoise(img):

    ret = img.copy()

    def decide_gaussian(g0, s0):
         
        if g0 <= 10:
            return -1
          
        tmp_g = f.denoiser_gaussian(img, g0)
        g_g = d.gaussian_noise_detector(tmp_g)
        g_s = d.snp_noise_detector(tmp_g)
        
        if g_g > g0 * (2/3) or g_s > s0 * (2/3):
             return -1
        
        return 1


    def decide_snp(g0, s0):
         
        if s0 <= 1:
            return -1
         
        tmp_s = f.denoiser_snp(img, s0)
        s_s = d.snp_noise_detector(tmp_s)
        s_g = d.gaussian_noise_detector(tmp_s)

        if s_s > s0/10 or s_s > s0/10: ## in progress
             return -1
        
        return 1