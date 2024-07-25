import cv2
import numpy as np

# Load the noisy image
noisy_img = cv2.imread('noisy_image.png', 0)

# Define the window sizes
ASWM_window_size = 5
AMF_window_size = 3
# Maximum window size for AMF
Smax = 60

# Pad the noisy image for border pixels for ASWM
ASWM_pad_size = ASWM_window_size // 2
padded_img_ASWM = cv2.copyMakeBorder(noisy_img, ASWM_pad_size, ASWM_pad_size, ASWM_pad_size, ASWM_pad_size, cv2.BORDER_CONSTANT)

# Create a copy of the padded image for ASWM restoration
ASWM_restored_img = padded_img_ASWM.copy()

# Create an empty list to store the noisy pixels
noisy_pixels = []

# Iterate over each pixel in the image for ASWM
for i in range(ASWM_pad_size, padded_img_ASWM.shape[0] - ASWM_pad_size):
    for j in range(ASWM_pad_size, padded_img_ASWM.shape[1] - ASWM_pad_size):
        # Get the neighborhood
        window = padded_img_ASWM[i - ASWM_pad_size:i + ASWM_pad_size + 1, j - ASWM_pad_size:j + ASWM_pad_size + 1].flatten()
        
        # Remove the pixel of interest from the window
        window = np.delete(window, len(window) // 2)
        
        # Calculate the minimum and maximum value
        min_val, max_val = np.min(window), np.max(window)
        
        # If the pixel value is less than or equal to the minimum value or greater than or equal to the maximum value, classify it as noisy
        if padded_img_ASWM[i, j] <= min_val or padded_img_ASWM[i, j] >= max_val:
            noisy_pixels.append((i, j))

# Define the weights for the ASWM
w1, w2, w3 = 1, 1, 10  # Adjust these weights as needed

# Convert noisy_pixels to a set for faster lookup
noisy_pixels_set = set(noisy_pixels)

# Iterate over each noisy pixel for ASWM
for x, y in noisy_pixels:
    # Get the neighborhood
    window = padded_img_ASWM[x - ASWM_pad_size:x + ASWM_pad_size + 1, y - ASWM_pad_size:y + ASWM_pad_size + 1]
    
    # Separate the pixels in the diagonals and the rest of the window
    D1 = [(i, i) for i in range(ASWM_window_size) if (x - ASWM_pad_size + i, y - ASWM_pad_size + i) not in noisy_pixels_set]
    D2 = [(i, ASWM_window_size - i - 1) for i in range(ASWM_window_size) if (x - ASWM_pad_size + i, y - ASWM_pad_size + ASWM_window_size - i - 1) not in noisy_pixels_set]
    D3 = [(i, j) for i in range(ASWM_window_size) for j in range(ASWM_window_size) if (i, j) not in D1 and (i, j) not in D2 and (x - ASWM_pad_size + i, y - ASWM_pad_size + j) not in noisy_pixels_set]
    
    # Calculate the ASWM only if D1, D2, and D3 are not all empty
    if D1 or D2 or D3:
        sum_D1 = np.sum([window[i, j] for i, j in D1], dtype=np.int32)
        sum_D2 = np.sum([window[i, j] for i, j in D2], dtype=np.int32)
        sum_D3 = np.sum([window[i, j] for i, j in D3], dtype=np.int32)
        weights_sum = (len(D1) * w1 + len(D2) * w2 + len(D3) * w3)
        if weights_sum != 0:
            ASWM = (w1 * sum_D1 + w2 * sum_D2 + w3 * sum_D3) / weights_sum
        else:
            ASWM = 0
        
        # Replace the noisy pixel with the ASWM in the restored image only if ASWM is not NaN
        if not np.isnan(ASWM):
            ASWM_restored_img[x, y] = np.clip(ASWM, 0, 255).astype(np.uint8)

ASWM_restored_img = ASWM_restored_img[ASWM_pad_size:-ASWM_pad_size, ASWM_pad_size:-ASWM_pad_size]

# Pad the ASWM restored image for AMF
AMF_pad_size = Smax // 2
padded_img_AMF = cv2.copyMakeBorder(ASWM_restored_img, AMF_pad_size, AMF_pad_size, AMF_pad_size, AMF_pad_size, cv2.BORDER_REFLECT)

# Create a copy of the padded image for AMF restoration
restored_img = padded_img_AMF.copy()

# Iterate over each pixel in the image for AMF
for i in range(AMF_pad_size, padded_img_AMF.shape[0] - AMF_pad_size):
    for j in range(AMF_pad_size, padded_img_AMF.shape[1] - AMF_pad_size):
        local_AMF_window_size = AMF_window_size
        local_AMF_pad_size = local_AMF_window_size // 2
        while local_AMF_window_size <= Smax:
            # Get the neighborhood for AMF
            window = padded_img_AMF[i - local_AMF_pad_size:i + local_AMF_pad_size + 1, j - local_AMF_pad_size:j + local_AMF_pad_size + 1].flatten()
            
            # Convert window to int32 for safe arithmetic operations
            window = window.astype(np.int32)
            
            # Calculate the minimum, median, and maximum value for AMF
            Zmin, Zmed, Zmax = np.min(window), np.median(window), np.max(window)
            Zxy = int(padded_img_AMF[i, j])
            
            # Level A for AMF
            A1 = Zmed - Zmin
            A2 = Zmed - Zmax
            if A1 > 0 and A2 < 0:
                # Level B for AMF
                B1 = Zxy - Zmin
                B2 = Zxy - Zmax
                if B1 > 0 and B2 < 0:
                    restored_img[i, j] = Zxy
                else:
                    restored_img[i, j] = Zmed
                break
            else:
                local_AMF_window_size += 2
                local_AMF_pad_size = local_AMF_window_size // 2

# Save the final restored image (removing the padding)
cv2.imwrite('frestoredimage.png', restored_img[AMF_pad_size:-AMF_pad_size, AMF_pad_size:-AMF_pad_size])

# Print the number of noisy pixels
print("Noisy pixels: ", len(noisy_pixels))
