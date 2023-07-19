import cv2
import numpy as np
import math

# resize image for im.show visualizing purposes
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def convex_hull_pointing_up(ch):
    #   Determines whether the path is directed upwards. If so, then this is a cone
        
    # contour points above the center and below
    points_above_center, points_below_center = [], []
    
    x, y, w, h = cv2.boundingRect(ch) # coordinates of the upper left corner of the circumscribing rectangle, width and height
    aspect_ratio = w / h # ratio of rectangle width to height
    # print("x:", x)
    # print("y:", y)
    # print("w:", w)
    # print("h:", h)
    # print("aspect ratio:", aspect_ratio)
    # if the rectangle is narrow, continue the definition. If not, then the circuit does not fit
    if aspect_ratio < 1.5:
        # each point of the contour is classified as lying above or below the center
        vertical_center = y + h / 2

        for point in ch:
            if point[0][1] < vertical_center: # if the y coordinate of a point is above the center, then add this point to the list of points above the center
                points_above_center.append(point)
            elif point[0][1] >= vertical_center:
                points_below_center.append(point)

        # determine the x coordinates of the extreme points lying below the center
        left_x = points_below_center[0][0][0]
        right_x = points_below_center[0][0][0]
        for point in points_below_center:
            if point[0][0] < left_x:
                left_x = point[0][0]
            if point[0][0] > right_x:
                right_x = point[0][0]

        #check if the top points of the contour lie outside the "base". If yes, then the circuit is not suitable
        for point in points_above_center:
            if (point[0][0] < left_x) or (point[0][0] > right_x):
                return False
    else:
        return False
        
    return True

def find_skew_angle(fiducial_coordinates):
        
    fiducial_1_x = round(fiducial_coordinates[0][0],2)
    fiducial_1_y = round(fiducial_coordinates[0][1],2)
    fiducial_2_x = round(fiducial_coordinates[1][0],2)
    fiducial_2_y = round(fiducial_coordinates[1][1],2)
            
    angle = 180+math.atan2(fiducial_2_y - fiducial_1_y, fiducial_2_x - fiducial_1_x)*(180/3.14159265) # finding the angle from the given fiducial coordinates
    print(angle)
    return angle

def adjust_homography(input_array, src_points, width, height):
    
    def sort_points(points):
        points = np.array(points)
        
        # Get x, y coordinates of each point
        x_coords, y_coords = points[:,0], points[:,1]
        
        # Sort points by x-coordinate
        sorted_x = np.argsort(x_coords)
        
        # Get leftmost and rightmost points
        leftmost = sorted_x[:2]
        rightmost = sorted_x[2:]
        
        # Sort leftmost and rightmost points by y-coordinate
        sorted_y_left = leftmost[np.argsort(y_coords[leftmost])]
        sorted_y_right = rightmost[np.argsort(y_coords[rightmost])]
        
        # Return sorted points as lower left, lower right, upper right, upper left
        return np.array([points[sorted_y_left[0]], points[sorted_y_left[1]], points[sorted_y_right[1]], points[sorted_y_right[0]]])
    
    for i,value in enumerate(input_array):
        input_array[i][0] = int(input_array[i][0]*width)
        input_array[i][1] = int(input_array[i][1]*height)

    # Calculate the homography matrix
    # span the lower coordinate to be the same Y values
    # span the left/right X coordinates to have the same X values
    src_points = sort_points(src_points)
    
    Upper_Left = src_points[0]
    Lower_Left = src_points[1]
    Upper_Right = src_points[2]
    Lower_Right = src_points[3]
    
    Upper_Left[0] = Lower_Left[0]
    Upper_Right[0] = Lower_Right[0]
    
    Lower_Left[1] = Lower_Right[1]
    Upper_Left[1] = Upper_Right[1]
    
    src_points_temp = src_points
    
    # Define the target points for the homography transformation
    target_points = np.array([Lower_Left, Lower_Right, Upper_Right, Upper_Left], np.float32)
    target_points = sort_points(target_points)
    
    src_points = np.array(src_points, np.float32)

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points_temp, target_points)

    # Apply the perspective transformation matrix to the input array
    input_array = np.array(input_array, np.float32)
    output_array = cv2.perspectiveTransform(input_array[None,:,:], M)[0]
    
    for i,value in enumerate(output_array):
        output_array[i][0] = output_array[i][0]/width
        output_array[i][1] = output_array[i][1]/height
    
    return output_array

def level_coordinates(coordinates, angle):
    
    # find left most point to set origin...based on left to right CV cone detection...needs to have the same origin to rotate around
    final_index = len(coordinates)-1
    x_0, y_0 = coordinates[0][0], coordinates[0][1]
    
    x_last, y_last = coordinates[final_index][0], coordinates[final_index][1]
    
    # print("checking origin assignment:")
    # print(x_0, y_0)
    # print(x_last, y_last)
    
    if x_0 > x_last:
        x_0 = x_last
        y_0 = y_last
    
    # print("checking origin re-assignment:")
    # print(x_0, y_0)
    # print(x_last, y_last)
    
    # convert angle to radians
    theta = np.deg2rad(angle)
    rotational_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    # translate points so that the origin is at the rotation point
    # points = np.column_stack((coordinates[0]))
    
    leveled_points = [[],[]]
    for x in range(0, len(coordinates)):
        leveled_points[0].append(coordinates[x][0] - x_0)        
        leveled_points[1].append(coordinates[x][1] - y_0)

    # rotate the points
    leveled_points = np.dot(rotational_matrix, np.array([leveled_points[0], leveled_points[1]]))

    # translate the points back
    leveled_points[0] += x_0
    leveled_points[1] += y_0
    
    leveled_points = np.column_stack((leveled_points[0], leveled_points[1]))

    return leveled_points


def get_meters_per_pixel_and_angle_from_fiducials(blob_path):

    if ".PNG" not in blob_path:
        cap = cv2.VideoCapture(blob_path)    
        rval, image = cap.read()
    else:
        image = cv2.imread(blob_path)
       
    width_multiple = 2
    width_blown_up = int(width_multiple*image.shape[1])
    height_blown_up = int(width_multiple*image.shape[0])
    
    image = ResizeWithAspectRatio(image, width=width_blown_up) # blow up the image to 2-4x to make sure any small cones are captured
        
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_HSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_thresh_low = cv2.inRange(img_HSV, np.array([0, 135, 135]), np.array([15, 255, 255])) #everything that is included in the "left red"
    img_thresh_high = cv2.inRange(img_HSV, np.array([159, 135, 135]), np.array([179, 255, 255])) #everything that is included in the "right red"
    
    img_thresh = cv2.bitwise_or(img_thresh_low, img_thresh_high) #combine obtained from
    kernel = np.ones((7, 7))
    img_thresh_opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 11)
    
    img_edges = cv2.Canny(img_thresh_blurred, 110, 180)
    contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
    # simplification of contours (reducing the number of points in them) using the Douglas-Peker algorithm
    approx_contours = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)

    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv2.convexHull(ac))

    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv2.convexHull(ch))

    cones = []
    bounding_rects = []
    for ch in convex_hulls_3to10:
        if convex_hull_pointing_up(ch):
            cones.append(ch)
            rect = cv2.boundingRect(ch)
            bounding_rects.append(rect)
                
    img_res = image

    # img_thresh_blurred = ResizeWithAspectRatio(image, width=640)
    # cv2.imshow('cones', img_thresh_blurred)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # img_edges = ResizeWithAspectRatio(img_edges, width=640)
    # cv2.imshow('cones', img_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    
    rect_number = 10
    
    for i, rect1 in enumerate(bounding_rects):
        for j, rect2 in enumerate(bounding_rects[i+1:]):
            # extract the top and bottom coordinates of each rectangle
            top1, bottom1 = rect1[1], rect1[1] + rect1[3]
            top2, bottom2 = rect2[1], rect2[1] + rect2[3]
            
            # extract the left and right coordinates of each rectangle
            left1, right1 = rect1[0], rect1[0] + rect1[2]
            left2, right2 = rect2[0], rect2[0] + rect2[2]
            
            # check if the top or bottom of either rectangle is intersected by the horizontal sides of the other rectangle
            if (top1 <= top2 <= bottom1 or top1 <= bottom2 <= bottom1) and (left1 <= right2 or left2 <= right1):
                bounding_rects = []                     
                bounding_rects.append(rect1)
                bounding_rects.append(rect2)
    
    if len(bounding_rects) >= rect_number:
        print(bounding_rects)
        fiducial_coordinates = [[],[],[],[]]
        fiducial_coordinates[0] = [bounding_rects[0][0]/width_blown_up, bounding_rects[0][1]/height_blown_up, bounding_rects[0][2]/width_blown_up, bounding_rects[0][3]/height_blown_up]
        fiducial_coordinates[1] = [bounding_rects[1][0]/width_blown_up, bounding_rects[1][1]/height_blown_up, bounding_rects[1][2]/width_blown_up, bounding_rects[1][3]/height_blown_up]
        
        fiducial_1_x = fiducial_coordinates[0][0] + (fiducial_coordinates[0][2]/2)
        fiducial_1_y = ((round(fiducial_coordinates[0][1],2) + round(fiducial_coordinates[0][3],2)/2))
        fiducial_2_x = fiducial_coordinates[1][0] + (fiducial_coordinates[1][2]/2)
        fiducial_2_y = ((round(fiducial_coordinates[1][1],2) + round(fiducial_coordinates[1][3],2)/2))
        
        fiducial_coordinates = [[],[]]
        fiducial_coordinates[0] = [fiducial_1_x,fiducial_1_y]
        fiducial_coordinates[1] = [fiducial_2_x,fiducial_2_y]
        
        angle = find_skew_angle(fiducial_coordinates)
        
        for rect in bounding_rects:
            cv2.rectangle(img_res, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 3)
                    
        img_res = ResizeWithAspectRatio(image, width=800) # blow up the image to 4x to make sure any small cones are captured

        cv2.imshow('cones', img_res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # draw only the pairs of rectangles whose vertical spans overlap on the Y axis    
        x_span = []
        if len(bounding_rects) == 2:        
            for rect in bounding_rects:  
                print("rectangle: ", rect)
                x_span.append(rect[0]/width_multiple)
            
            cap = cv2.VideoCapture(blob_path)
            rval, image = cap.read()
            cap.release()
            print(x_span)
            
            x_span = int(abs(x_span[1] - x_span[0]))
            print("x span: ", x_span)
            cone_distance = 20 #4.572 #6.4008 # #8.5
            meters_to_pixels = abs(cone_distance/x_span) # meter/pixel conversion
        else:
            x_span = 0
            meters_to_pixels = 0
    
    # if there are no bounding rectangles, prompt user for manual coordinate input + adjust for homography
    else:
        # Define the mouse callback function
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add the selected point to the list
                points.append((x, y))
                print("Selected point ({}, {})".format(x, y))
                # Draw a small circle at the selected point
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
                cv2.imshow("image", img)

        # Load the image
        img = image
        
        image_width = 1200
        img = ResizeWithAspectRatio(img, width=image_width)

        # Create a window to display the image
        cv2.namedWindow("image")

        # Initialize the list of selected points
        points = []

        # Set the mouse callback function
        cv2.setMouseCallback("image", mouse_callback)

        # Display the image
        cv2.imshow("image", img)

        # Wait for the user to select 4 points
        while len(points) < 4:
            cv2.waitKey(1)

        # Print the selected points
        print("Selected points:")
        for point in points:
            print(point)

        # Close the window
        cv2.destroyAllWindows()
        
        # Sort the points by their X-coordinate
        points.sort(key=lambda p: p[0])

        # Split the points into two groups based on the X-coordinate
        left_points = points[:2]
        right_points = points[2:]

        # Draw lines connecting the left points and right points
        cv2.line(img, left_points[0], left_points[1], (0, 0, 255), 2)
        cv2.line(img, right_points[0], right_points[1], (0, 0, 255), 2)

        # Calculate the midpoints of the two lines
        left_midpoint = tuple(np.mean(left_points, axis=0).astype(int))
        right_midpoint = tuple(np.mean(right_points, axis=0).astype(int))

        # Draw a line connecting the two midpoints
        cv2.line(img, left_midpoint, right_midpoint, (0, 0, 255), 2)

        # Calculate the span on the X-axis
        x_span_percent = abs(right_midpoint[0] - left_midpoint[0])/img.shape[1] # as a percent
              
        x_span = x_span_percent*(width_blown_up/width_multiple) # multiple x_span percentage by original image width
        
        fiducial_coordinates = [[],[]]
        fiducial_coordinates[0] = right_midpoint
        fiducial_coordinates[1] = left_midpoint
        angle = find_skew_angle(fiducial_coordinates)

        # Show the image and wait for a keypress
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Print the span on the X-axis
        print("Span on the X-axis:", x_span)
        
        cone_distance = 10 #4.572 #6.4008 # #8.5
        meters_to_pixels = abs(cone_distance/x_span) # meter/pixel conversion
    
    points = np.array(points, np.float32)
    for i in range(0,len(points)):
        points[i][0] = int(points[i][0]/(img.shape[1])*image.shape[1])
        points[i][1] = int(points[i][1]/(img.shape[0])*image.shape[0])

    # bounding rects is the resulting array of cone coordinates. 
    print(meters_to_pixels, angle, points)
    return meters_to_pixels, angle, points

    # x_1 = bounding_rects[0][0] # the intersecting point 1 X position
    # x_2 = bounding_rects[1][0] # the intersecting point 2 X position
    # x_coords = [x_1,x_2]
        
    # return x_coords