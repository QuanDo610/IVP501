import numpy as np
import cv2
from skimage import exposure

def histeq_modified(a, cm=None, hgram=None):
    """
    Histogram Equalization (modified version of MATLAB histeq).
    
    Parameters
    ----------
    a : np.ndarray
        Input image (grayscale or indexed).
    cm : int or np.ndarray, optional
        - If int: number of bins (N).
        - If colormap: numpy array shape (n,3).
    hgram : np.ndarray, optional
        Desired histogram.
    
    Returns
    -------
    out : np.ndarray
        Output image or colormap.
    T : np.ndarray
        Transformation function.
    """

    is_intensity = True if (hgram is not None or cm is None or np.isscalar(cm)) else False

    # Determine number of points (levels)
    if a.dtype == np.uint8:
        NPTS = 256
    elif a.dtype == np.uint16:
        NPTS = 65536
    else:
        NPTS = 256

    # Case 1: only image
    if cm is None and hgram is None:
        n = 64
        hgram = np.ones(n) * (a.size / n)
        is_intensity = True

    # Case 2: I, N
    elif np.isscalar(cm):
        m = int(cm)
        hgram = np.ones(m) * (a.size / m)
        is_intensity = True

    # Case 3: I, hgram
    elif cm is not None and hgram is None and cm.ndim == 1:
        hgram = cm
        is_intensity = True

    # Normalize hgram
    hgram = hgram * (a.size / np.sum(hgram))
    m = len(hgram)

    if is_intensity:
        # Compute histogram of input
        nn, bins = np.histogram(a.flatten(), NPTS, [0, NPTS])
        cum = np.cumsum(nn)

        # Desired cumulative histogram
        cumd = np.cumsum(hgram)

        # Error matrix
        err = np.abs(cumd[:, None] - cum[None, :])

        # Mapping: find index with min error
        T = np.argmin(err, axis=0)
        T = (T / (m - 1)).astype(np.float64)

        # Apply transform
        out = np.interp(a.flatten(), np.arange(NPTS), T * (NPTS - 1))
        out = out.reshape(a.shape).astype(a.dtype)

        return out, T

    else:
        # Indexed image + colormap (ít dùng trong Python)
        raise NotImplementedError("Colormap mode chưa được port đầy đủ sang Python.")
