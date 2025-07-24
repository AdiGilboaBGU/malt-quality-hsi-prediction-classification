############################################################################################################
#### AVERAGE REFLECTANCES WITH EROSION ####
############################################################################################################

from sklearn.model_selection import train_test_split
from skimage.morphology import erosion, disk
import sys
import shutil
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import spectral as spy
spy.settings.envi_support_nonlowercase_params = True
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import pandas as pd
import spectral as spy
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion
from skimage.filters import threshold_otsu
import scipy.signal
from scipy.ndimage import gaussian_filter1d
import os
import re

############################################################################################################
# Finding the band with the highest reflectance
# and otsu threshold calculation is in Main_V2 code

# Define the cropping function
def crop_image(mask,spec_img):
    labeled_seeds = label(mask)
    props = regionprops(labeled_seeds)

    # Filter out segments with an area greater than 300
    filtered_props = [prop for prop in props if prop.area > 300]

    # Get the centroids of these filtered segments
    centroids = np.array([prop.centroid for prop in filtered_props])

    # Sort the centroids by their y-coordinate (the vertical position)
    sorted_indices = np.argsort(centroids[:, 0])

    # Exclude the first two and last two segments to remove the top-most and bottom-most segments
    # This step adjusts for only focusing on segments with significant area
    remaining_indices = sorted_indices[2:-2]

    # Calculate the bounding box for the remaining segments
    min_row, min_col, max_row, max_col = np.inf, np.inf, 0, 0
    for idx in remaining_indices:
        prop = filtered_props[idx]
        minr, minc, maxr, maxc = prop.bbox
        min_row, min_col = min(min_row, minr), min(min_col, minc)
        max_row, max_col = max(max_row, maxr), max(max_col, maxc)

    n_pixels = 15

    # Crop the image to this bounding box
    # Ensure cropping is within the bounds of the original image dimensions
    min_row, max_row = max(0, min_row - n_pixels), min(spec_img.shape[0], max_row + n_pixels)
    min_col, max_col = max(0, min_col - n_pixels), min(spec_img.shape[1], max_col + n_pixels)

    cropped_img = spec_img[min_row:max_row, min_col:max_col]

    return cropped_img

############################################################################################################

def snv(input_data):
    # Apply SNV to each spectrum
    snv_output = (input_data - np.mean(input_data, axis=2, keepdims=True)) / np.std(input_data, axis=2, keepdims=True)
    return snv_output

############################################################################################################

def msc(input_data):
    # Compute the mean spectrum across all pixels (mean along axis 0 and axis 1)
    mean_spectrum = np.mean(input_data, axis=(0, 1))
    msc_output = np.zeros_like(input_data)
    
    # Iterate over each pixel
    for i in range(input_data.shape[0]):
        for j in range(input_data.shape[1]):
            # Perform linear regression of each pixel's spectrum on the mean spectrum
            fit = np.polyfit(mean_spectrum, input_data[i, j, :], 1, full=True)
            slope, intercept = fit[0][0], fit[0][1]
            
            # Apply correction
            msc_output[i, j, :] = (input_data[i, j, :] - intercept) / slope
    
    return msc_output

############################################################################################################
"""
def apply_filter(spec_img, filter_type):
    
    spec_img_filtered = spec_img.copy()
    
    if filter_type == 'SG':
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_filtered, window_length=11, polyorder=2, axis=2)

    elif filter_type == 'FD_SG':
        # Increase polyorder to 3 to keep effective polynomial order at 2 after first derivative
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_filtered, window_length=11, polyorder=3, deriv=1, axis=2)
    
    elif filter_type == 'SD_SG':
        # Increase polyorder to 4 to keep effective polynomial order at 2 after second derivative
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_filtered, window_length=11, polyorder=4, deriv=2, axis=2)

    elif filter_type == 'SNV':
        spec_img_filtered = snv(spec_img_filtered)

    elif filter_type == 'MSC':
        spec_img_filtered = msc(spec_img_filtered)

    elif filter_type == 'FD':
        spec_img_filtered = np.gradient(spec_img_filtered, axis=2)

    elif filter_type == 'SD':
        spec_img_filtered = gaussian_filter1d(spec_img_filtered, sigma=1, axis=2)
        spec_img_filtered = np.gradient(np.gradient(spec_img_filtered, axis=2), axis=2)
#   This log is applied element-wise to each individual pixel value across the entire array

    elif filter_type == 'log(1_R)':
        spec_img_filtered = np.log(1 / (spec_img_filtered + 1e-10))

    elif filter_type == 'log(1_R)_SG':
        spec_img_filtered = np.log(1 / (spec_img_filtered + 1e-10))
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_filtered, window_length=11, polyorder=2, axis=2)

    elif filter_type == 'SNV_SG':
        # First apply SNV to normalize, then apply SG to smooth the normalized data
        spec_img_normalized = snv(spec_img_filtered)
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_normalized, window_length=11, polyorder=2, axis=2)
        
    elif filter_type == 'SG_SNV':
        # First apply SNV to normalize, then apply SG to smooth the normalized data
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_filtered, window_length=11, polyorder=2, axis=2)
        spec_img_filtered = snv(spec_img_filtered)


    elif filter_type == 'MSC_SG':
        # First apply SNV to normalize, then apply SG to smooth the normalized data
        spec_img_normalized = msc(spec_img_filtered)
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_normalized, window_length=11, polyorder=2, axis=2)
    
    elif filter_type == 'NONE':
        pass
    
    return spec_img_filtered
"""


def apply_filter(spec_img, filter_type): 
    
    spec_img_filtered = spec_img.copy()
    
    if filter_type == 'SG_FD':
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_filtered, window_length=11, polyorder=2, deriv=1, axis=2)

    elif filter_type == 'SG_SD':
        # Increase polyorder to 4 to keep effective polynomial order at 2 after second derivative
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_filtered, window_length=11, polyorder=2, deriv=2, axis=2)

    elif filter_type == 'SG_SNV':
        # First apply SNV to normalize, then apply SG to smooth the normalized data
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_filtered, window_length=11, polyorder=2, axis=2)
        spec_img_filtered = snv(spec_img_filtered)
        
    elif filter_type == 'SG_MSC':
        # First apply SNV to normalize, then apply SG to smooth the normalized data
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_filtered, window_length=11, polyorder=2, axis=2)
        spec_img_filtered = msc(spec_img_filtered)
        
    elif filter_type == 'SG_FD_SNV':
        # First apply SNV to normalize, then apply SG to smooth the normalized data
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_filtered, window_length=11, polyorder=2, deriv=1, axis=2)
        spec_img_filtered = snv(spec_img_filtered)

    elif filter_type == 'MSC_SNV':
        # First apply SNV to normalize, then apply SG to smooth the normalized data
        spec_img_filtered = msc(spec_img_filtered)
        spec_img_filtered = snv(spec_img_filtered) 
        
    elif filter_type == 'DE_SG_SD':
        # First detrend the data
        spec_img_detrended = scipy.signal.detrend(spec_img_filtered, axis=2, type='linear')
        # Then apply Savitzky-Golay filter to the detrended data
        spec_img_filtered = scipy.signal.savgol_filter(spec_img_detrended, window_length=11, polyorder=2, deriv=2, axis=2)
        

    elif filter_type == 'NONE':
        pass
    
    return spec_img_filtered


############################################################################################################

def calc_avg_ref (erosion_radius, filter_type):

    # Go to Parent folder of the images
    dataset_path = 'C:\\Users\gilboa\Desktop\TargetFolder'
    os.chdir(dataset_path)
    
    # List the files
    photo_folder = os.listdir()
    # photo_folder = [file for file in os.listdir() if file.endswith('.dat')][:10]
    # photo_folder = ['REFLECTANCE_33.dat', 'REFLECTANCE_27.dat', 'REFLECTANCE_65.dat']
    
    # Create a data frame to save the image
    data = pd.DataFrame(columns=['Name of image', 'hyper cube', 'number of bands'])
    
    # Dictionary to hold dataframes for each image
    dataframes = {} # keep the 204 AVG reflectances per image (spec_img_eroded)
    masks_dict = {} # keep each mask per eroded image (for later use)
    eroded_images_dict = {} # keep each image pixels (for later use)
    avg_reflectances_list = [] # Define a list to store average reflectances
                               # for each band in the current image

    
    # Loading the images
    for file in photo_folder:
        if file.endswith('dat'):
            dat_file = os.path.join(dataset_path, file)
            hdr_file = dat_file.replace('.dat', '.hdr')
    
            spec_img_obj = spy.io.envi.open(hdr_file, dat_file)  # Keep it as an object first
            wavelength_range = np.array(spec_img_obj.metadata['wavelength'])  # Extract metadata first
    
            spec_img = spec_img_obj.load()  # Convert to numpy array
            avg_reflectances_list.clear()
            # Create a binary mask according to threshold otsu
            mask = np.all(spec_img[:, :, 158] > 0.33610326051712036, axis=2)
            # Crop the image
            spec_img_cropped = crop_image(mask, spec_img)
            
            mask = np.all(spec_img_cropped[:, :, 158] > 0.33610326051712036, axis=2)
            # masks_dict[file] = mask
    
            # Apply the mask to the image - this sets the value of all background pixels to zero
            spec_img_masked = spec_img_cropped.copy()
            spec_img_masked[~mask, :] = 0
            
            # spec_img_masked is a masked array where all zero values are masked. This means that operations applied to this array will ignore the masked (zero) values.
            spec_img_masked = np.ma.masked_equal(spec_img_masked, 0)
       
            print(spec_img_masked.shape)
            
            # Apply erosion to remove the external layer of the seeds
            selem = disk(erosion_radius)
                        
            # Apply erosion to the 158th band
            eroded_band = erosion(spec_img_masked[:, :, 158], selem)
            
            spec_img_eroded = spec_img_masked.copy()
            
            # Assuming 'eroded_band' is the result of the erosion on band 158
            mask_eroded_band = eroded_band > 0
            
            for i in range(spec_img_eroded.shape[2]):
                # Apply the erosion mask: Set values to zero where the erosion mask is False
                spec_img_eroded[:, :, i][~mask_eroded_band] = 0
            
            # Mask the zeros in the eroded image so that the filters ignore them
            spec_img_eroded = np.ma.masked_equal(spec_img_eroded, 0)

            # Now apply the filter to the eroded image
            spec_img_filtered = apply_filter(spec_img_eroded, filter_type)
                            
            for i in range(spec_img_filtered.shape[2]):
               # Directly use the erosion mask when calculating the average for the eroded image
                avg_ref = np.ma.mean(np.ma.masked_array(spec_img_filtered[:, :, i], mask=~mask_eroded_band), axis=(0, 1))
    
                avg_reflectances_list.append(avg_ref)
                
                # Calculate the count of contributing pixels by using the erosion mask
                eroded_pixel_count = np.sum(mask_eroded_band)
                
                # Calculate the original average and pixel count as before
                original_avg = np.ma.mean(spec_img_masked[:, :, i], axis=(0, 1))
                original_pixel_count = np.ma.count(spec_img_masked[:, :, i])
            
                print(f"Processing band {i+1}/{spec_img_filtered.shape[2]}")
                print(f"Band {i+1}: Original average: {original_avg}, Eroded average: {avg_ref}")
                print(f"Band {i+1}: Original pixel count: {original_pixel_count}, Eroded pixel count: {eroded_pixel_count}")
                   
            print(f"processing image: {file}, with filter: {filter_type}")
            new_mask = eroded_band > 0  # Apply threshold to generate new binary mask
            # Use the new_mask for your calculations instead of the original mask
            masks_dict[file] = new_mask  # Store the new mask for geometric feature calculations
            eroded_images_dict[file] = spec_img_eroded
            
            # Example for one band
            # Calculate averages for the original image
            # avg_reflectances = np.ma.mean(spec_img_eroded, axis=(0, 1))
    
            # Save each image's average reflectances in a separate DataFrame
            dataframes[file] = pd.DataFrame(avg_reflectances_list, columns=['Average Reflectance'])
            # Apply the mask to the image - this sets the value of all background pixels to zero
#             rgb_img_masked = spec_img_cropped[:, :, [70, 53, 19]]
#             # Show the image
#             plt.imshow(rgb_img_masked)
#             plt.show()
    
    # Initialize an empty DataFrame for combining all data
    combined_data = pd.DataFrame()
    
    # Sort the keys of the dictionary based on the numerical value after the underscore in the filename
    sorted_keys = sorted(dataframes.keys(), key=lambda x: int(x.split('_')[-1].replace('.dat', '')))
    
    # Iterate through the sorted keys to add each DataFrame as a column in the combined DataFrame
    for key in sorted_keys:
        df = dataframes[key]
        # Ensure that the index is consistent across all DataFrames for proper alignment
        df.reset_index(drop=True, inplace=True)
        # Rename the column to the image's name for clarity, assuming the keys include the '.dat' extension
        df.columns = [key.replace('.dat', '')]
        # Concatenate horizontally (axis=1) to add as a new column
        combined_data = pd.concat([combined_data, df], axis=1)
    
    combined_data = combined_data.transpose()
    
    combined_data.reset_index(inplace=True)
    combined_data.rename(columns={'index': 'Labels'}, inplace=True)
   
    # Create a Pandas Excel writer using XlsxWriter as the engine
    excel_file = excel_file = r'G:\My Drive\Thesis\Temp_Work\excel_files\testing_datasets\average_reflectances_with_erosion_'+str(erosion_radius) +'_'+ filter_type +'.xlsx'

    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        combined_data.to_excel(writer, sheet_name='All_Images', index=False)
        print("All images' data written to one sheet in the Excel file")
    
    return masks_dict, eroded_images_dict


############################################################################################################

# calc_avg_ref(1, 'NONE')

# Define the erosion levels and filter types
erosion_levels = [1, 3, 5, 7]

filter_types = ['SG_FD', 'SG_SD', 'SG_SNV',
                'SG_MSC', 'SG_FD_SNV', 'MSC_SNV',
                'DE_SG_SD' ,'NONE']

# Nested loop to call calc_avg_ref for each combination
for erosion_radius in erosion_levels:
    for filter_type in filter_types:
        calc_avg_ref(erosion_radius, filter_type)
    
      
##############################################################################################################
#### From now, to run all 2 parts next, make sure that until now the running was made with selem = 1

############################################################################################################
#### GEOMETRIC PARAMETERS ####
############################################################################################################
masks_dict, eroded_images_dict = calc_avg_ref(1, 'NONE')

from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import pandas as pd

def calculate_geometric_parameters(mask):
    labeled_seeds = label(mask)
    props = regionprops(labeled_seeds)

    # Initialize sums for calculating averages
    total_height = 0
    total_width = 0
    total_area = 0
    total_perimeter = 0
    total_eccentricity = 0
    total_orientation = 0
    total_major_axis_length = 0
    total_minor_axis_length = 0
    total_extent = 0
    total_equivalent_diameter = 0
    total_convex_area = 0
    total_feret_diameter_max = 0
    total_solidity = 0
    count = 0

    # Extract properties of each seed and filter by width
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        width = maxc - minc
        if width > 20:
            if width > 100: # for image_33
                # Split the seed into two
                mid_point = minc + width // 2
                widths = [mid_point - minc, maxc - mid_point]
                mincs = [minc, mid_point]
                maxcs = [mid_point, maxc]
                
                for w, minc, maxc in zip(widths, mincs, maxcs):
                    height = maxr - minr
                    area = (maxc - minc) * height
                    perimeter = 2 * (height + w)
                    major_axis_length = max(width, height)
                    minor_axis_length = min(width, height)
                    orientation = 0
                    eccentricity = (1 - (minor_axis_length ** 2 / major_axis_length ** 2)) ** 0.5
                    
                    # Calculating properties for each split part
                    total_height += height
                    total_width += w
                    total_area += area
                    total_perimeter += perimeter
                    total_eccentricity += eccentricity
                    total_orientation += orientation
                    total_major_axis_length += major_axis_length
                    total_minor_axis_length += minor_axis_length
                    total_extent += area / ((maxc - minc) * (maxr - minr))
                    total_equivalent_diameter += np.sqrt(4 * area / np.pi)
                    total_convex_area += area  # Simplified as area for split parts
                    total_feret_diameter_max += maxc - minc
                    total_solidity += area / area  # Simplified as 1 for split parts
                    count += 1

            else:
                height = maxr - minr
                total_height += height
                total_width += width
                total_area += prop.area
                total_perimeter += prop.perimeter
                total_eccentricity += prop.eccentricity
                total_orientation += prop.orientation
                total_major_axis_length += prop.major_axis_length
                total_minor_axis_length += prop.minor_axis_length
                total_extent += prop.extent
                total_equivalent_diameter += prop.equivalent_diameter
                total_convex_area += prop.convex_area
                total_feret_diameter_max += prop.feret_diameter_max
                total_solidity += prop.solidity
                count += 1

    # Calculate averages if there are seeds found
    averages = {
        "average_height": total_height / count if count else 0,
        "average_width": total_width / count if count else 0,
        "average_area": total_area / count if count else 0,
        "average_perimeter": total_perimeter / count if count else 0,
        "average_eccentricity": total_eccentricity / count if count else 0,
        "average_orientation": total_orientation / count if count else 0,
        "average_major_axis_length": total_major_axis_length / count if count else 0,
        "average_minor_axis_length": total_minor_axis_length / count if count else 0,
        "average_extent": total_extent / count if count else 0,
        "average_equivalent_diameter": total_equivalent_diameter / count if count else 0,
        "average_convex_area": total_convex_area / count if count else 0,
        "average_feret_diameter_max": total_feret_diameter_max / count if count else 0,
        "average_solidity": total_solidity / count if count else 0,
    }

    return count, averages


# List to hold image data
image_data = []

# Masks_dict is defined elsewhere and contains image masks
for file, new_mask in masks_dict.items():
    print(f"Processing image: {file}")
    num_seeds, averages = calculate_geometric_parameters(new_mask)
    image_data.append({
        "Name of the Image": file,
        "Number of Seeds Found": num_seeds,
        **averages  # Unpack averages directly into the dictionary
    })

# Create a DataFrame and print the table
df = pd.DataFrame(image_data)
print(df)

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Convert 'Name of the Image' to string in case it's not
df['Name of the Image'] = df['Name of the Image'].astype(str)

# Extract the numerical part from the 'Name of the Image' column
df['Image_Number'] = df['Name of the Image'].str.extract('(\d+)').astype(int)

# Sort the DataFrame based on the 'Image_Number' column
df_sorted = df.sort_values(by='Image_Number')

# Select only the columns with geometric parameters for scaling
geometric_columns = df_sorted.columns[2:]  # Adjust this as per your actual DataFrame
geometric_data = df_sorted[geometric_columns]


df_sorted[geometric_columns] = geometric_data

# Drop the 'Image_Number' column as it's no longer needed
df_sorted = df_sorted.drop('Image_Number', axis=1)

# Specify the path to your Excel file
excel_path = r'G:\My Drive\Thesis\Temp_Work\excel_files\testing_datasets\geometric_parameters.xlsx'

# Use the to_excel method to write the sorted and scaled DataFrame to an Excel file
df_sorted.to_excel(excel_path, index=False)

print(f"DataFrame with geometric parameters has been exported to {excel_path}")



############################################################################################################
#### CLUSTERING USING KMEANS ####
############################################################################################################
###########################################################################################################

masks_dict, eroded_images_dict = calc_avg_ref(1, 'NONE')
    
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
import numpy as np
import pandas as pd

def process_and_plot_seeds_with_correct_reflectance(image, mask, image_key, n_clusters=2, filter_type='NONE'):
    print(f"Processing image: {image_key}")
    labels = label(mask)
    props = regionprops(labels)
    
    embryo_reflectances = []
    endosperm_reflectances = []
    
    # Initialize sums and pixel counts for reflectance calculations
    sum_reflectances_cluster_1 = np.zeros(image.shape[2])
    pixel_counts_cluster_1 = 0
    sum_reflectances_cluster_2 = np.zeros(image.shape[2])
    pixel_counts_cluster_2 = 0
    seed_count = 0  # Initialize the seed counter
   
    plt.figure(figsize=(20, 20))  # Create a larger figure to hold 64 subplots
    # Add the main title with the current image name, increase font size, and adjust position
    plt.suptitle(f'{image_key} - All Seeds', fontsize=25, y=0.9)   
    # Process each seed
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        width = maxc - minc
        if width > 20:
            seed_count += 1  # Increment the seed counter
#            if seed_count > 64:  # Prevent exceeding the number of subplots available
#                print("Reached the maximum number of seeds to display.")
#                break

            seed_image = image[minr:maxr, minc:maxc]
            seed_mask = mask[minr:maxr, minc:maxc]
            
            # Apply the mask to the seed image
            masked_seed_image = seed_image[seed_mask]
            
            # Flatten the masked seed image for PCA
            flattened_seed = masked_seed_image.reshape(-1, seed_image.shape[2])
            pca = PCA(n_components=2)  # Reduce to two dimensions
            reduced_data = pca.fit_transform(flattened_seed)
            
            # Apply k-means clustering only to the seed pixels
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
            cluster_labels = kmeans.fit_predict(reduced_data) + 1  # Adjust cluster labels to start from 1 instead of 0
            
            # Create an empty array to hold the clustering results for the full seed image
            cluster_image = np.zeros(seed_mask.shape, dtype=int)        
            
            # Place the clustering results back into the full seed image, at the locations of the seed
            seed_mask_indices = np.where(seed_mask)
            cluster_image[seed_mask_indices] = cluster_labels
                    
            # Apply filter to the original seed image before calculating reflectance
            filtered_seed_image = apply_filter(seed_image, filter_type) 
            
            # Calculate the centroid of each cluster
            cluster_1_centroid = np.mean(np.column_stack(np.where(cluster_image == 1)), axis=0)
            cluster_2_centroid = np.mean(np.column_stack(np.where(cluster_image == 2)), axis=0)
    
            # Determine which cluster is closer to the left (pointed end)
            if cluster_1_centroid[1] < cluster_2_centroid[1]:
                embryo_cluster = 1
                endosperm_cluster = 2
            else:
                embryo_cluster = 2
                endosperm_cluster = 1
    
            
            # Adjust cluster_image labels accordingly to preserve the background as 0
            cluster_image = np.where(cluster_image == embryo_cluster, 1, 
                                     np.where(cluster_image == endosperm_cluster, 2, 0))
            
            
            from matplotlib.colors import ListedColormap
            # Define custom colors
            custom_cmap = ListedColormap(['white', 'blue', 'red'])
    
            # Color the clusters for clear visualization
            colors = np.zeros_like(cluster_image, dtype=int)
            colors[cluster_image == 1] = 1
            colors[cluster_image == 2] = 2
            
            # Plot the original seed image with colored clusters
            plt.subplot(8, 8, seed_count)
            plt.imshow(np.max(seed_image, axis=2), cmap='gray')  # Max projection for visualization
            plt.imshow(colors, alpha=0.4, cmap=custom_cmap)  # Apply color overlay
            
            # Add contour for clarity
            plt.contour(cluster_image, levels=np.arange(n_clusters+1)-0.5, colors='red', linewidths=2)
    
            plt.title(f'Seed {seed_count}')
            plt.axis('off')  # Turn off axes for clarity
            
                                     
            cluster1_mask = (cluster_image == 1)
            filtered_cluster1_image = filtered_seed_image[cluster1_mask]
            sum_reflectances_cluster_1 += np.sum(filtered_cluster1_image, axis=0)
            pixel_counts_cluster_1 += np.sum(cluster1_mask)
            
    
            cluster2_mask = (cluster_image == 2)
            filtered_cluster2_image = filtered_seed_image[cluster2_mask]
            sum_reflectances_cluster_2 += np.sum(filtered_cluster2_image, axis=0)
            pixel_counts_cluster_2 += np.sum(cluster2_mask)
            
            # Calculate average reflectance for each cluster in the current seed
            reflectance_embryo = filtered_cluster1_image.mean(axis=0)
            reflectance_endosperm = filtered_cluster2_image.mean(axis=0)
            
            # Saving the averages for the current seed for boxplot purposes
            if reflectance_embryo.size > 0:
                embryo_reflectances.append(reflectance_embryo.mean())
            if reflectance_endosperm.size > 0:
                endosperm_reflectances.append(reflectance_endosperm.mean())
    
    # End of loop
    print(f'Total seeds processed: {seed_count}')  # Print the total number of seeds processed    

    # Calculation of overall averages for each cluster in the entire image
    avg_reflectance_cluster_1 = sum_reflectances_cluster_1 / pixel_counts_cluster_1
    avg_reflectance_cluster_2 = sum_reflectances_cluster_2 / pixel_counts_cluster_2
    
    # Plotting the boxplot
    plt.figure(figsize=(10, 6))
    if embryo_reflectances and endosperm_reflectances:  # Ensure data is not empty
        plt.boxplot([embryo_reflectances, endosperm_reflectances], labels=['Embryo', 'Endosperm'])
    else:
        plt.text(0.5, 0.5, 'No data available for box plot', horizontalalignment='center', verticalalignment='center', fontsize=14)
    plt.title(f'Reflectance Comparison for Embryo and Endosperm in {image_key}')
    plt.ylabel('Reflectance')
    plt.show()
    
    return avg_reflectance_cluster_1, avg_reflectance_cluster_2

########
# running the function and export to excel


def run_and_export_data():

    # Path where the Excel files will be saved
    base_path = r'G:\My Drive\Thesis\Temp_Work\excel_files\testing_datasets'
    
    # List of filters
    filter_types = ['SG_FD', 'SG_SD', 'SG_SNV',
                    'SG_MSC', 'SG_FD_SNV', 'MSC_SNV',
                    'DE_SG_SD' ,'NONE']

    
    for filter_type in filter_types:
        
        # Initialize DataFrames to hold overall results for Embryo and Endosperm separately
        embryo_reflectance_data = pd.DataFrame()
        endosperm_reflectance_data = pd.DataFrame()
    
        for key, image in eroded_images_dict.items():
            mask = masks_dict[key]  # Get the corresponding mask
            # Assuming process_and_plot_seeds_with_correct_reflectance is defined elsewhere
            avg_reflectance_cluster_1, avg_reflectance_cluster_2 = process_and_plot_seeds_with_correct_reflectance(image, mask, key, filter_type=filter_type)
            
            # Generate column names based on the key
            base_name = key.replace('.dat', '')
            embryo_reflectance_data[f"{base_name}_embryo"] = avg_reflectance_cluster_1
            endosperm_reflectance_data[f"{base_name}_endosperm"] = avg_reflectance_cluster_2
        
        # Transpose and format the data
        embryo_reflectance_data = embryo_reflectance_data.transpose().reset_index().rename(columns={'index': 'Labels'})
        endosperm_reflectance_data = endosperm_reflectance_data.transpose().reset_index().rename(columns={'index': 'Labels'})
    
        # Extract numerical part from 'Labels' for sorting
        embryo_reflectance_data['Image_Number'] = embryo_reflectance_data['Labels'].str.extract('(\d+)').astype(int)
        endosperm_reflectance_data['Image_Number'] = endosperm_reflectance_data['Labels'].str.extract('(\d+)').astype(int)
    
        # Sort the DataFrames by 'Image_Number' and drop the 'Image_Number' column
        embryo_reflectance_data.sort_values('Image_Number', inplace=True)
        endosperm_reflectance_data.sort_values('Image_Number', inplace=True)
    
        # Export to Excel
        embryo_file_path = f'{base_path}\\embryo_reflectances_{filter_type}.xlsx'
        endosperm_file_path = f'{base_path}\\endosperm_reflectances_{filter_type}.xlsx'
    
        with pd.ExcelWriter(embryo_file_path, engine='xlsxwriter') as writer:
            embryo_reflectance_data.to_excel(writer, sheet_name='Reflectances_Embryo', index=False)
    
        with pd.ExcelWriter(endosperm_file_path, engine='xlsxwriter') as writer:
            endosperm_reflectance_data.to_excel(writer, sheet_name='Reflectances_Endosperm', index=False)

# Call the function to process and export data
run_and_export_data()

####----------------------------------------------------------------------------------------------------####
####----------------------------------------------------------------------------------------------------####
####------------------------------------------ PER SEED ------------------------------------------------####
####----------------------------------------------------------------------------------------------------####
####----------------------------------------------------------------------------------------------------####

############################################################################################################
#### AVERAGE REFLECTANCES PER SEED WITH FILTERS AND EROSION = 1 ####
############################################################################################################
############################################################################################################

from scipy.signal import savgol_filter, detrend


# Define MSC function
def apply_msc(X):
    """
    Apply Multiplicative Scatter Correction (MSC) to hyperspectral data.
    
    Parameters:
    - X: 2D array of hyperspectral data (n_samples, n_features)
    
    Returns:
    - msc_X: MSC-corrected data
    """
    mean_spectrum = np.mean(X, axis=0)
    msc_X = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        slope, intercept = np.polyfit(mean_spectrum, X[i, :], 1)
        msc_X[i, :] = (X[i, :] - intercept) / slope
    
    return msc_X

def process_images_to_table():
    dataset_path = r'C:\Users\gilboa\Desktop\TargetFolder'
    labels_file = r'G:\My Drive\Thesis\Temp_Work\excel_files\labels_categorized.xlsx'
    labels = pd.read_excel(labels_file, index_col=0)
    
    # Store processed features and labels
    all_features = []
    feature_indices = []  # Indices for features and labels
    y_labels_categorized = []  # To store labels
    
    
    os.chdir(dataset_path)
    photo_folder = os.listdir()
    photo_folder = sorted(os.listdir(), key=lambda x: int(x.split('_')[-1].split('.')[0])) 
    
    for file in photo_folder:
        print(file)
        if file.endswith('dat'):
            dat_file = os.path.join(dataset_path, file)
            hdr_file = dat_file.replace('.dat', '.hdr')
            
            # Load hyperspectral image
            spec_img_obj = spy.io.envi.open(hdr_file, dat_file)
            spec_img = spec_img_obj.load()
            
            # Extract the base filename and retrieve the corresponding label
            file_base = os.path.splitext(file)[0]
            label_value = labels.loc[file_base, 'Label']  # Generic label column name
            print(f"File: {file}, Label: {label_value}")
            
            # Create a binary mask exactly as in your code
            mask = np.all(spec_img[:, :, 158] > 0.33610326051712036, axis=2)
            
            # Crop the image
            spec_img_cropped = crop_image(mask, spec_img)
            
            # Apply a new mask to the cropped image
            mask = np.all(spec_img_cropped[:, :, 158] > 0.33610326051712036, axis=2)
            # Apply the mask to the image - this sets the value of all background pixels to zero
            spec_img_masked = spec_img_cropped.copy()
            spec_img_masked[~mask, :] = 0
            
            # spec_img_masked is a masked array where all zero values are masked. This means that operations applied to this array will ignore the masked (zero) values.
            spec_img_masked = np.ma.masked_equal(spec_img_masked, 0)
            
            # Apply erosion to the mask to reduce noise and small artifacts
            selem = disk(1)  # Define erosion element with radius 1
            # Apply erosion to the 158th band
            eroded_band = erosion(spec_img_masked[:, :, 158], selem)
            
            spec_img_eroded = spec_img_masked.copy()
            
            # Assuming 'eroded_band' is the result of the erosion on band 158
            mask_eroded_band = eroded_band > 0
#            print(f"mask_eroded_band {mask_eroded_band.shape}")
            
            for i in range(spec_img_eroded.shape[2]):
                # Apply the erosion mask: Set values to zero where the erosion mask is False
                spec_img_eroded[:, :, i][~mask_eroded_band] = 0
            
            # Mask the zeros in the eroded image so that the filters ignore them
            spec_img_eroded = np.ma.masked_equal(spec_img_eroded, 0)
#            print(f"spec_img_eroded {spec_img_eroded.shape}")
            
            new_mask = eroded_band > 0
#            print(f"new_mask {new_mask.shape}")
#            print("Number of pixels in mask:", np.sum(new_mask))
            
            # Label and extract regions from the eroded mask
            labeled_seeds = label(new_mask)
            
            props = regionprops(labeled_seeds)
            
            # Visualize the original image, mask, and labeled seeds
#            visualize_processing(spec_img, mask, new_mask)
            
            count = 0                  
            for i, prop in enumerate(props):
                minr, minc, maxr, maxc = prop.bbox
                width = maxc - minc
                
                # Ensure the seed meets size criteria
                if 100 > width > 20:  # Add width filter here as well
                    count += 1  # Increment count for each valid seed

#                    seed_region = spec_img_eroded[minr:maxr, minc:maxc, :]
#                    seed_pixels = seed_region.reshape(-1, seed_region.shape[-1])
                    
                    seed_region = spec_img_eroded[minr:maxr, minc:maxc]
                    seed_mask = new_mask[minr:maxr, minc:maxc]
                    seed_pixels = seed_region[seed_mask, :]

                    # Detrending each pixel in seed region
                    detrended_pixels = detrend(seed_pixels, axis=1)
                    
                    # Applying Savitzky-Golay filter
                    smoothed_pixels = savgol_filter(detrended_pixels, window_length=11, polyorder=2, deriv=2, axis=1)

                    
                    """
                    smoothed_pixels = savgol_filter(seed_pixels, window_length=11, polyorder=2, deriv=2, axis=1)
                    """
#                    snv_pixels = (smoothed_pixels - smoothed_pixels.mean(axis=1, keepdims=True)) / smoothed_pixels.std(axis=1, keepdims=True)

#                    mean_reflectance = snv_pixels.mean(axis=0)

# Apply MSC instead of SG

#                    msc_pixels = apply_msc(seed_pixels)                   
#                   mean_reflectance = msc_pixels.mean(axis=0)
#                    snv_pixels = (msc_pixels - msc_pixels.mean(axis=1, keepdims=True)) / msc_pixels.std(axis=1, keepdims=True)
                    
                    mean_reflectance = smoothed_pixels.mean(axis=0)


#                    seed_std = seed_pixels.std(axis=0).mean()
                    
                    # Add data
                    seed_index = f"{file_base}_seed_{count}"
                    feature_indices.append(seed_index)
                    all_features.append(mean_reflectance)
                    
                    y_labels_categorized.append(label_value)
            
            
            # Debug statement to check how many seeds are detected
            print(f"Number of seeds detected in {file_base}: {count}")
    
    return feature_indices, all_features, y_labels_categorized

# Process images and save resized seeds
feature_indices, all_features, y_labels_categorized = process_images_to_table()
all_features_df = pd.DataFrame(all_features, index=feature_indices)
all_features_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files\X_by_seed_DE_SG_SD.xlsx')

# Save labels to a file
y_labels_categorized_df = pd.DataFrame(y_labels_categorized, columns=["Label"])
y_labels_categorized_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files\y_labels_categorized_2_moisture_2%.xlsx', index=False)


############################################################################################################
#### CALCULATING GEOMETRIC PARAMETERS PER SEED ####
############################################################################################################
###########################################################################################################

from skimage.measure import label, regionprops
import pandas as pd
import numpy as np

def calculate_geometric_parameters(mask, image_name):
    labeled_seeds = label(mask)
    props = regionprops(labeled_seeds)
    seeds_data = []
    seed_index = 1
         
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        width = maxc - minc
        if width > 20:  # Filter based on width
            if width > 100:  # for image_33
                # Split the seed into two
                mid_point = minc + width // 2
                segments = [
                    (minc, mid_point),
                    (mid_point, maxc)
                ]

                for minc, maxc in segments:
                    height = maxr - minr
                    width = maxc - minc
                    area = width * height
                    perimeter = 2 * (height + width)
                    major_axis_length = max(width, height)
                    minor_axis_length = min(width, height)
                    orientation = 0
                    eccentricity = (1 - (minor_axis_length ** 2 / major_axis_length ** 2)) ** 0.5
                    extent = area / (width * height)
                    equivalent_diameter = np.sqrt(4 * area / np.pi)
                    convex_area = area
                    feret_diameter_max = width
                    solidity = 1

                    seed_data = {
                        "Name of the Seed": f"{image_name}_seed_{seed_index}",
                        "Height": height,
                        "Width": width,
                        "Area": area,
                        "Perimeter": perimeter,
                        "Eccentricity": eccentricity,
                        "Orientation": orientation,
                        "Major Axis Length": major_axis_length,
                        "Minor Axis Length": minor_axis_length,
                        "Extent": extent,
                        "Equivalent Diameter": equivalent_diameter,
                        "Convex Area": convex_area,
                        "Feret Diameter Max": feret_diameter_max,
                        "Solidity": solidity
                    }
                    seeds_data.append(seed_data)
                    seed_index += 1
            else:
                height = maxr - minr
                area = prop.area
                perimeter = prop.perimeter
                eccentricity = prop.eccentricity
                orientation = prop.orientation
                major_axis_length = prop.major_axis_length
                minor_axis_length = prop.minor_axis_length
                extent = prop.extent
                equivalent_diameter = prop.equivalent_diameter
                convex_area = prop.convex_area
                feret_diameter_max = prop.feret_diameter_max
                solidity = prop.solidity

                seed_data = {
                    "Name of the Seed": f"{image_name}_seed_{seed_index}",
                    "Height": height,
                    "Width": width,
                    "Area": area,
                    "Perimeter": perimeter,
                    "Eccentricity": eccentricity,
                    "Orientation": orientation,
                    "Major Axis Length": major_axis_length,
                    "Minor Axis Length": minor_axis_length,
                    "Extent": extent,
                    "Equivalent Diameter": equivalent_diameter,
                    "Convex Area": convex_area,
                    "Feret Diameter Max": feret_diameter_max,
                    "Solidity": solidity
                }
                seeds_data.append(seed_data)
                seed_index += 1
    
    return seeds_data

# Assuming 'masks_dict' is already defined and filled with data:
all_seeds_data = []

# Sort the keys to process images in numerical order
sorted_keys = sorted(masks_dict.keys(), key=lambda x: int(x.split('_')[1].split('.')[0]))
for file in sorted_keys:
    print(f"Processing image: {file}")
    new_mask = masks_dict[file]
    seeds_data = calculate_geometric_parameters(new_mask, file)
    all_seeds_data.extend(seeds_data)

# Create a DataFrame and print the table
df = pd.DataFrame(all_seeds_data)

# Specify the path to your Excel file
excel_path = r'G:\My Drive\Thesis\Temp_Work\excel_files\testing_datasets\geometric_parameters_PER_SEED.xlsx'

# Use the to_excel method to write the DataFrame to an Excel file
df.to_excel(excel_path, index=False)

print(f"DataFrame with geometric parameters has been exported to {excel_path}")


############################################################################################################
#### CLUSTERING USING KMEANS PER SEED ####
############################################################################################################
###########################################################################################################

# masks_dict, eroded_images_dict = calc_avg_ref(1, 'NONE')
    
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def K_Means_Per_Seed (image, mask, image_key, n_clusters=2, filter_type='NONE'):
 
    print(f"Processing image: {image_key}")
    labels = label(mask)
    props = regionprops(labels)
    
    # Initialize DataFrames to store reflectance data for each seed and cluster type
    embryo_reflectances = []
    endosperm_reflectances = []

    seed_count = 0  # Initialize the seed counter
   
    plt.figure(figsize=(20, 20))  # Create a larger figure to hold 64 subplots
    # Add the main title with the current image name, increase font size, and adjust position
    plt.suptitle(f'{image_key} - All Seeds', fontsize=25, y=0.9)   
    # Process each seed
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        width = maxc - minc
        if width > 20:
            seed_count += 1  # Increment the seed counter
#            if seed_count > 64:  # Prevent exceeding the number of subplots available
#                print("Reached the maximum number of seeds to display.")
#                break

            seed_image = image[minr:maxr, minc:maxc]
            seed_mask = mask[minr:maxr, minc:maxc]
            
            # Apply the mask to the seed image
            masked_seed_image = seed_image[seed_mask]
            
            # Flatten the masked seed image for PCA
            flattened_seed = masked_seed_image.reshape(-1, seed_image.shape[2])
            pca = PCA(n_components=2)  # Reduce to two dimensions
            reduced_data = pca.fit_transform(flattened_seed)
            
            # Apply k-means clustering only to the seed pixels
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
            cluster_labels = kmeans.fit_predict(reduced_data) + 1  # Adjust cluster labels to start from 1 instead of 0
            
            # Create an empty array to hold the clustering results for the full seed image
            cluster_image = np.zeros(seed_mask.shape, dtype=int)        
            
            # Place the clustering results back into the full seed image, at the locations of the seed
            seed_mask_indices = np.where(seed_mask)
            cluster_image[seed_mask_indices] = cluster_labels
                    
            # Apply filter to the original seed image before calculating reflectance
            filtered_seed_image = apply_filter(seed_image, filter_type) 
            
            # Calculate the centroid of each cluster
            cluster_1_centroid = np.mean(np.column_stack(np.where(cluster_image == 1)), axis=0)
            cluster_2_centroid = np.mean(np.column_stack(np.where(cluster_image == 2)), axis=0)
    
            # Determine which cluster is closer to the left (pointed end)
            if cluster_1_centroid[1] < cluster_2_centroid[1]:
                embryo_cluster = 1
                endosperm_cluster = 2
            else:
                embryo_cluster = 2
                endosperm_cluster = 1
    
            
            # Adjust cluster_image labels accordingly to preserve the background as 0
            cluster_image = np.where(cluster_image == embryo_cluster, 1, 
                                     np.where(cluster_image == endosperm_cluster, 2, 0))
            
            
            from matplotlib.colors import ListedColormap
            # Define custom colors
            custom_cmap = ListedColormap(['white', 'blue', 'red'])
    
            # Color the clusters for clear visualization
            colors = np.zeros_like(cluster_image, dtype=int)
            colors[cluster_image == 1] = 1
            colors[cluster_image == 2] = 2
            
            # Plot the original seed image with colored clusters
            plt.subplot(8, 8, seed_count)
            plt.imshow(np.max(seed_image, axis=2), cmap='gray')  # Max projection for visualization
            plt.imshow(colors, alpha=0.4, cmap=custom_cmap)  # Apply color overlay
            
            # Add contour for clarity
            plt.contour(cluster_image, levels=np.arange(n_clusters+1)-0.5, colors='red', linewidths=2)
    
            plt.title(f'Seed {seed_count}')
            plt.axis('off')  # Turn off axes for clarity
            
                                     
            cluster1_mask = (cluster_image == 1)
            filtered_cluster1_image = filtered_seed_image[cluster1_mask]
            reflectance_embryo = filtered_cluster1_image.mean(axis=0)
            if reflectance_embryo.size > 0:
                embryo_reflectances.append(reflectance_embryo)
            
            cluster2_mask = (cluster_image == 2)
            filtered_cluster2_image = filtered_seed_image[cluster2_mask]
            reflectance_endosperm = filtered_cluster2_image.mean(axis=0)
            if reflectance_endosperm.size > 0:
                endosperm_reflectances.append(reflectance_endosperm)
    
    print(f'Total seeds processed: {seed_count}')  # Print the total number of seeds processed    
    plt.show()
    return embryo_reflectances, endosperm_reflectances

def export_to_excel(eroded_images_dict, masks_dict, base_path, filter_types):

    for filter_type in filter_types:
        
        embryo_reflectance_data_all = pd.DataFrame()
        endosperm_reflectance_data_all = pd.DataFrame()

        # Sort keys for consistent processing
        keys_sorted = sorted(eroded_images_dict.keys(), key=lambda x: int(x.split('_')[-1].replace('.dat', '')))
    
        for key in keys_sorted:  # Use the sorted keys
            image = eroded_images_dict[key]
            mask = masks_dict[key]
            embryo_reflectances, endosperm_reflectances = K_Means_Per_Seed(image, mask, key, filter_type=filter_type)
            
            # Process embryo reflectances
            for idx, reflectance in enumerate(embryo_reflectances):
                row_name = f"{key.replace('.dat', '')}_seed_{idx+1}_embryo"
                data_df = pd.DataFrame([reflectance], columns=[f'Band_{i}' for i in range(len(reflectance))], index=[row_name])
                embryo_reflectance_data_all = pd.concat([embryo_reflectance_data_all, data_df])
            
            # Process endosperm reflectances
            for idx, reflectance in enumerate(endosperm_reflectances):
                row_name = f"{key.replace('.dat', '')}_seed_{idx+1}_endosperm"
                data_df = pd.DataFrame([reflectance], columns=[f'Band_{i}' for i in range(len(reflectance))], index=[row_name])
                endosperm_reflectance_data_all = pd.concat([endosperm_reflectance_data_all, data_df])

        # Save to Excel
        embryo_file_path = f'{base_path}/embryo_reflectances_{filter_type}.xlsx'
        endosperm_file_path = f'{base_path}/endosperm_reflectances_{filter_type}.xlsx'
        embryo_reflectance_data_all.to_excel(embryo_file_path)
        endosperm_reflectance_data_all.to_excel(endosperm_file_path)
        
        print(f"Data saved to {embryo_file_path} and {endosperm_file_path}")
    # Return the DataFrames for further use
    return embryo_reflectance_data_all, endosperm_reflectance_data_all

# Usage
base_path = r'G:\My Drive\Thesis\Temp_Work\excel_files\kmeans_per_seed'
filter_types = ['SG_FD', 'SG_SD', 'SG_SNV',
                'SG_MSC', 'SG_FD_SNV', 'MSC_SNV',
                'DE_SG_SD' ,'NONE']
embryo_df, endosperm_df = export_to_excel(eroded_images_dict, masks_dict, base_path, filter_types)

print("Embryo DataFrame:")
print(embryo_df.head())  # Display the first few rows for verification
print("Endosperm DataFrame:")
print(endosperm_df.head())


