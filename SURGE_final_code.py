import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

def cluster_image_exclude_black_hsv(image_path, n_clusters=3, threshold=10):
    """
    Cluster the image excluding black pixels in the HSV Value channel.

    Args:
        image_path (str): Path to the image file.
        n_clusters (int): Number of clusters for K-means.
        threshold (int): Threshold for excluding black pixels.

    Returns:
        tuple: Original RGB image, clustered RGB image, K-means model, and mask.
    """
    # Load the RGB image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask to exclude black pixels in the Value channel of HSV
    _, mask = cv2.threshold(image_hsv[:, :, 2], threshold, 255, cv2.THRESH_BINARY)

    # Extract non-black pixels using the mask
    masked_image = image_hsv[mask != 0]

    # Apply K-means clustering to the masked HSV pixels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(masked_image)
    clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]

    # Create the clustered image
    clustered_image = np.zeros_like(image_hsv)
    clustered_image[mask != 0] = clustered_pixels

    # Convert the clustered image back to RGB for display
    clustered_image_rgb = cv2.cvtColor(clustered_image.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return image, clustered_image_rgb, kmeans, mask

def get_largest_outer_contour(image, epsilon=0.01):
    """
    Find the largest outer contour in the image.

    Args:
        image (ndarray): Grayscale image.
        epsilon (float): Approximation accuracy parameter.

    Returns:
        tuple: Largest contour and approximated polygon.
    """
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the largest contour as a polygon
    perimeter = cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon * perimeter, True)
    
    return largest_contour, approx_polygon

def find_innermost_and_outermost_clusters(clustered_image, kmeans, mask):
    """
    Identify the innermost and outermost clusters in the clustered image.

    Args:
        clustered_image (ndarray): Clustered image in HSV.
        kmeans (KMeans): K-means model.
        mask (ndarray): Mask of non-black pixels.

    Returns:
        tuple: Innermost and outermost cluster images and masks.
    """
    h, w, _ = clustered_image.shape
    cluster_labels = np.full(mask.shape, -1)
    cluster_labels[mask != 0] = kmeans.labels_

    cluster_boundary_count = np.zeros(kmeans.n_clusters, dtype=int)

    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                current_label = cluster_labels[y, x]
                neighbors = cluster_labels[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)]
                if np.any(neighbors != current_label):
                    cluster_boundary_count[current_label] += 1

    innermost_cluster_idx = np.argmin(cluster_boundary_count)
    outermost_cluster_idx = np.argmax(cluster_boundary_count)

    innermost_cluster_mask = (cluster_labels == innermost_cluster_idx)
    outermost_cluster_mask = (cluster_labels == outermost_cluster_idx)

    innermost_cluster_image = np.zeros_like(clustered_image)
    outermost_cluster_image = np.zeros_like(clustered_image)
    innermost_cluster_image[innermost_cluster_mask] = clustered_image[innermost_cluster_mask]
    outermost_cluster_image[outermost_cluster_mask] = clustered_image[outermost_cluster_mask]

    innermost_cluster_image_rgb = cv2.cvtColor(innermost_cluster_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    outermost_cluster_image_rgb = cv2.cvtColor(outermost_cluster_image.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return innermost_cluster_image_rgb, outermost_cluster_image_rgb, innermost_cluster_mask, outermost_cluster_mask

def get_center_of_mass_of_innermost_cluster(innermost_cluster_mask):
    """
    Calculate the center of mass of the innermost cluster.

    Args:
        innermost_cluster_mask (ndarray): Mask of the innermost cluster.

    Returns:
        tuple: Coordinates of the center of mass (x, y).
    """
    indices = np.column_stack(np.where(innermost_cluster_mask))
    weights = innermost_cluster_mask[indices[:, 0], indices[:, 1]]

    if len(indices) > 0:
        center_of_mass = np.average(indices, axis=0, weights=weights).astype(int)
        centroid_y, centroid_x = center_of_mass[0], center_of_mass[1]
    else:
        centroid_x, centroid_y = 0, 0

    return centroid_x, centroid_y

def draw_largest_contour(image, mask, color=(0, 255, 0)):
    """
    Draw the largest contour in the mask.

    Args:
        image (ndarray): Original image.
        mask (ndarray): Binary mask.
        color (tuple): Color of the contour.

    Returns:
        tuple: Image with the largest contour drawn.
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Draw the largest contour on the image
    image_with_contour = image.copy()
    cv2.drawContours(image_with_contour, [largest_contour], -1, color, 2)
    
    return image_with_contour, largest_contour

def draw_angular_lines(image, center, num_lines=36, color=(255, 255, 255)):
    """
    Draw angular lines radiating from a center point.

    Args:
        image (ndarray): Original image.
        center (tuple): Center point (x, y).
        num_lines (int): Number of lines to draw.
        color (tuple): Color of the lines.

    Returns:
        tuple: Image with lines and their endpoints.
    """
    angle_step = 360 / num_lines
    h, w = image.shape[:2]
    line_endpoints = []

    for i in range(num_lines):
        angle = np.deg2rad(i * angle_step)
        x_end = int(center[0] + w * np.cos(angle))
        y_end = int(center[1] - h * np.sin(angle))
        cv2.line(image, center, (x_end, y_end), color, 1)
        line_endpoints.append((x_end, y_end))

    return image, line_endpoints

def find_intersections(line_endpoints, contour, image_shape):
    """
    Find intersections between lines and a contour. In case of multiple intersections, 
    return the farthest intersection point for each line. If no intersection is found,
    use the endpoint of the line.

    Args:
        line_endpoints (list): List of line endpoints.
        contour (ndarray): Contour points.
        image_shape (tuple): Shape of the image (height, width).

    Returns:
        list: Intersection points for each line.
    """
    intersections = []
    h, w = image_shape[:2]
    
    for (x1, y1), (x2, y2) in line_endpoints:
        farthest_intersection = None
        max_distance = -1
        
        for i in range(len(contour) - 1):
            x3, y3 = contour[i][0]
            x4, y4 = contour[i + 1][0]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            
            if not np.isclose(denom, 0):
                ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
                ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
                
                if 0 <= ua <= 1 and 0 <= ub <= 1:
                    x = x1 + ua * (x2 - x1)
                    y = y1 + ua * (y2 - y1)
                    distance = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
                    
                    if distance > max_distance:
                        max_distance = distance
                        farthest_intersection = (int(x), int(y))
        
        if farthest_intersection is None:
            farthest_intersection = (x2, y2)
        
        intersections.append(farthest_intersection)

    return intersections
def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1 (tuple): Coordinates of point 1 (x1, y1).
        point2 (tuple): Coordinates of point 2 (x2, y2).

    Returns:
        float: Euclidean distance between point1 and point2.
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def process_image(image_path):
    """
    Process an image to cluster pixels, find contours, draw shapes and lines,
    and calculate lengths between intersections.

    Args:
        image_path (str): Path to the image file.

    Returns:
        None
    """
    original_image, clustered_image, kmeans, mask = cluster_image_exclude_black_hsv(image_path)
    largest_contour_image, approx_polygon = get_largest_outer_contour(mask, epsilon=0.02)
    innermost_cluster_image, _, innermost_cluster_mask, _ = find_innermost_and_outermost_clusters(clustered_image, kmeans, mask)
    centroid_x, centroid_y = get_center_of_mass_of_innermost_cluster(innermost_cluster_mask)
    original_with_innermost_contour, innermost_contour = draw_largest_contour(original_image, innermost_cluster_mask, color=(255, 0, 0))

    # Draw angular lines from the center of mass
    center = (centroid_x, centroid_y)
    original_with_lines, line_endpoints = draw_angular_lines(original_image.copy(), center, num_lines=36, color=(255, 255, 255))

    # Find intersections of the angular lines with the enclosing shapes
    innermost_intersections = find_intersections([(center, endpoint) for endpoint in line_endpoints], innermost_contour, original_image.shape)
    outermost_intersections = find_intersections([(center, endpoint) for endpoint in line_endpoints], largest_contour_image, original_image.shape)

    # Store paired intersection points and calculate distances
    paired_intersections = list(zip(innermost_intersections, outermost_intersections))

    # Create a separate copy of the original image to draw lines and points
    lines_and_points_image = original_image.copy()

    # Arrays to store points p1 and p2 for each line
    #midpoints=[]
    p1_points = []
    p2_points=[]
    p3_points=[]
   

    for (inner_point, outer_point), (center, endpoint) in zip(paired_intersections, line_endpoints):
        # Calculate the mid-point on the line between inner_point and outer_point
        p1 = ((inner_point[0] + outer_point[0]) // 2, (inner_point[1] + outer_point[1]) // 2)
        p2 = ((inner_point[0] + p1[0]) // 2, (inner_point[1] + p1[1]) // 2)
        p3 = ((outer_point[0] + p1[0]) // 2, (outer_point[1] + p1[1]) // 2)
        # Store p1 and p2 points
        
        p1_points.append(p1)
        p2_points.append(p2)
        p3_points.append(p3)

        # Draw lines between innermost and outermost intersection points
        cv2.line(lines_and_points_image, inner_point, outer_point, (255, 255, 0), 1)

        # Draw points p1 and p2 on the image
        cv2.circle(lines_and_points_image, p1, 2, (255, 255, 255), -1)
        cv2.circle(lines_and_points_image, p2, 2, (255, 255, 255), -1)
        cv2.circle(lines_and_points_image, p3, 2, (255, 255, 255), -1)
        
    l1=len(p1_points)-1
    l2=len(p2_points)-1
    l3=len(p3_points)-1
    for i in range(len(p1_points)-1):
        cv2.line(lines_and_points_image, p1_points[i], p1_points[i+1], (0, 0, 0), 2)
        print(i)
    cv2.line(lines_and_points_image,p1_points[l1],p1_points[0],(0,0,0),2)
    for i in range(len(p2_points)-1):
        cv2.line(lines_and_points_image, p2_points[i], p2_points[i+1], (0, 0, 0), 2)
    cv2.line(lines_and_points_image,p2_points[l2],p2_points[0],(0,0,0),2)
    for i in range(len(p3_points)-1):
        cv2.line(lines_and_points_image, p3_points[i], p3_points[i+1], (0, 0, 0), 2) 
    cv2.line(lines_and_points_image,p3_points[l3],p3_points[0],(0,0,0),2)      
    print(l1)     
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(lines_and_points_image, cv2.COLOR_BGR2RGB))
    plt.title('Image with Lines and Equidistant Points')
    plt.axis('off')
    plt.show()


def process_images_in_folder(folder_path):
    """
    Process all images in a folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        None
    """
    for path in os.listdir(folder_path):
        image_path = os.path.join(folder_path, path)
        process_image(image_path)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
folder_path = r"C:\Users\visha\Desktop\New folder"
process_images_in_folder(folder_path)