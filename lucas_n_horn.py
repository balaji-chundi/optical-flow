import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import cv2
from __future__ import annotations
from scipy.signal import convolve2d


def runLK(prev_img, curr_img, prev_pts, **lk_params):

    prev_pts = np.array(prev_pts, dtype = np.float32)

    curr_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None, **lk_params)

    est_flow = np.zeros(shape = (prev_img.shape[0], prev_img.shape[1], 2), dtype = np.float32)
    n = 0
    flow_pts = curr_pts - prev_pts

    for r in range(prev_img.shape[0]):
        for c in range(prev_img.shape[1]):
            if(n==flow_pts.shape[0]):
                break
            est_flow[r, c, :] = flow_pts[n, :]
            n = n + 1
    curr_pts = curr_pts[st == 1]
    new_pts = curr_pts.reshape(-1, 1, 2)
  
    return est_flow, new_pts

def warp_image_gray(im, flow):
    
    from scipy import interpolate
    image_height = im.shape[0]
    image_width = im.shape[1]
    flow_height = flow.shape[0]
    flow_width = flow.shape[1]
    n = image_height * image_width
    (iy, ix) = np.mgrid[0:image_height, 0:image_width]
    (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
    fx = fx.astype(np.float64)
    fy = fy.astype(np.float64)
    fx += flow[:,:,0]
    fy += flow[:,:,1]
    mask = np.logical_or(fx <0 , fx > flow_width)
    mask = np.logical_or(mask, fy < 0)
    mask = np.logical_or(mask, fy > flow_height)
    fx = np.minimum(np.maximum(fx, 0), flow_width)
    fy = np.minimum(np.maximum(fy, 0), flow_height)
    points = np.concatenate((ix.reshape(n,1), iy.reshape(n,1)), axis=1)
    xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n,1)), axis=1)
    warp = np.zeros((image_height, image_width))
    for i in range(1):
        channel = im[:, :]
        values = channel.reshape(n, 1)
        new_channel = interpolate.griddata(points, values, xi, method='cubic')
        new_channel = np.reshape(new_channel, [flow_height, flow_width])
        new_channel[mask] = 1
        warp[:, :] = new_channel.astype(np.uint8)

    return warp.astype(np.uint8)


def loss_function(im1, im2, id):
    loss = 0
    if id == 0:
        #MSE Loss
        for i in range(im1.shape[0]):
            for j in range(im1.shape[1]):
                val = np.abs(im1[i,j]-im2[i,j])
                loss = loss + val*val

    return loss

kernelX = np.array([[-1, 1], [-1, 1]]) * 0.25  # kernel for computing d/dx

kernelY = np.array([[-1, -1], [1, 1]]) * 0.25  # kernel for computing d/dy

kernelT = np.ones((2, 2)) * 0.25

def HornSchunck( im1, im2, alpha, Niter):

    kernel_hs = np.array([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], float)

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)
    vInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Iteration to reduce error
    for _ in range(Niter):
        # Compute local averages of the flow vectors
        uAvg = convolve2d(U, kernel_hs, "same")
        vAvg = convolve2d(V, kernel_hs, "same")        
        # common part of update step
        der = (fx * uAvg + fy * vAvg + ft) / (alpha ** 2 + fx ** 2 + fy ** 2) 
        #  iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der
    flow = [U, V]
    return flow

def computeDerivatives(im1, im2):

    fx = convolve2d(im1, kernelX, "same") + convolve2d(im2, kernelX, "same")
    fy = convolve2d(im1, kernelY, "same") + convolve2d(im2, kernelY, "same")
    ft = convolve2d(im1, kernelT, "same") + convolve2d(im2, -kernelT, "same")

    return fx, fy, ft

if __name__ == '__main__':

    #Lucas-Kanade on the corridor frames
    #images might be read in unsorted way
    corridor_images = glob.glob("../corridor/*.pgm") 

    #sorting images by the number in them while comaparing non-digit chars with "\D" of re library
    corridor_images.sort(key=lambda f: int(re.sub("\D", "", f))) 

    c_images = []
    for i in range(len(corridor_images)):
        c_images.append(cv2.imread(corridor_images[i], flags=cv2.IMREAD_GRAYSCALE))

    #parameters for lucas-kanade
    # maxLevel would be 0 always as we are not using multi-level lucas-kanade
    lk_params = dict( winSize  = (13, 13),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # To track the points, first, we need to find the points to be tracked
    p0 = cv2.goodFeaturesToTrack(c_images[0], mask = None, **feature_params)
    i = 0
    lk_errs = []
    while(i<9):
        
        curr_flow_estimate, new_pts = runLK(c_images[i], c_images[i+1], p0, **lk_params)

        warped_img = warp_image_gray(c_images[i+1], 2*curr_flow_estimate)
        plt.imsave(f'predicted_frame_{i+2}_lk_corridor.png', warped_img, cmap = 'gray')

        # err = loss_function(warped_img, images[i+2], id = 0)
        err = np.square(np.subtract(warped_img, c_images[i+2])).mean() #mse
        lk_errs.append(err)

        p0 = new_pts
        i = i + 1

    # print(lk_errs)


     #parameters for horn-schunck
    # performing grid-search over for the best parameters
    # val_alpha = [0.001, 0.01, 0.1, 0.2, 0.5, 1, 1.5]
    # val_iter = [8, 10, 20, 25, 50, 100]
    bestError = float('inf')
    # i = 0
    var1 = 1.5
    var2 = 100
    hs_errs = []
    i = 0
    while(i<9):

        curr_flow_estimate = HornSchunck(c_images[i], c_images[i+1], alpha = var1, Niter = var2)
        curr_flow_estimate = np.array(curr_flow_estimate)
        # print(curr_flow_estimate.shape)
        curr_flow_estimate = curr_flow_estimate.transpose(1, 2, 0)
        warped_img = warp_image_gray(c_images[i+1], curr_flow_estimate)
        plt.imsave(f'predicted_frame_{i+2}_hs_corridor.png', warped_img, cmap = 'gray')

        # err = loss_function(warped_img, images[i+2], id = 0)
        err = np.square(np.subtract(warped_img, c_images[i+2])).mean() #mse
        hs_errs.append(err)
        i = i + 1

    maxError = np.amax(np.array(hs_errs))

    # print(maxError)


    #lucas-kanade on sphere frames
    sphere_images = glob.glob("../sphere/*.ppm") 

    #sorting images by the number in them while comaparing non-digit chars with "\D" of re library
    sphere_images.sort(key=lambda f: int(re.sub("\D", "", f))) 

    s_images = []
    for i in range(len(sphere_images)):
        # img = cv2.imread(sphere_images[i], flags=cv2.IMREAD_GRAYSCALE)
        s_images.append(cv2.imread(sphere_images[i], flags=cv2.IMREAD_GRAYSCALE))

    lk_params = dict( winSize  = (12, 12),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # To track the points, first, we need to find the points to be tracked
    p0 = cv2.goodFeaturesToTrack(s_images[0], mask = None, **feature_params)
    i = 0
    sphere_lk_errs = []
    while(i<18):
        
        curr_flow_estimate, new_pts = runLK(s_images[i], s_images[i+1], p0, **lk_params)

        warped_img = warp_image_gray(s_images[i+1], 4*curr_flow_estimate)
        plt.imsave(f'predicted_frame_{i+2}_lk_sphere.png', warped_img, cmap = 'gray')

        # err = loss_function(warped_img, images[i+2], id = 0)
        err = np.square(np.subtract(warped_img, s_images[i+2])).mean() #mse
        sphere_lk_errs.append(err)

        p0 = new_pts
        i = i + 1

    # print(sphere_lk_errs)

     #parameters for horn-schunck
    # performing grid-search over for the best parameters
    # val_alpha = [0.001, 0.01, 0.1, 0.2, 0.5, 1, 1.5]
    # val_iter = [8, 10, 20, 25, 50, 100]
    bestError = float('inf')
    # i = 0
    var1 = 1.5
    var2 = 100
    sphere_hs_errs = []
    i = 0
    while(i<18):

        curr_flow_estimate = HornSchunck(s_images[i], s_images[i+1], alpha = var1, Niter = var2)
        curr_flow_estimate = np.array(curr_flow_estimate)
        # print(curr_flow_estimate.shape)
        curr_flow_estimate = curr_flow_estimate.transpose(1, 2, 0)
        warped_img = warp_image_gray(s_images[i+1], curr_flow_estimate)
        plt.imsave(f'predicted_frame_{i+2}_hs_sphere.png', warped_img, cmap = 'gray')

        # err = loss_function(warped_img, images[i+2], id = 0)
        err = np.square(np.subtract(warped_img, s_images[i+2])).mean() #mse
        sphere_hs_errs.append(err)
        i = i + 1

    maxError = np.amax(np.array(sphere_hs_errs))

    # print(maxError)
    

                
