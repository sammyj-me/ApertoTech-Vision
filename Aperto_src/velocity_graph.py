import cv2
import numpy as np
from resize_image import*

def add_velocity_graph(current_frame_img, frame_num, calc_array, total_frames):
    
    # Create an array of values for X and Y coordinates
    x_values = np.linspace(1,total_frames+1,10)
    x_values = np.round(x_values)
    
    y_values = np.linspace(0, max(calc_array[3]), 8)
    # y_values = np.round(y_values)
    
    # Determine the size of the image
    y_size = current_frame_img.shape[0]
    x_size = current_frame_img.shape[1]
    img_size = (int(x_size*.5), int(y_size*.25))
    
    # Create a blank image
    img = np.full((img_size[1], img_size[0], 3), 255, np.uint8)
    
    # Determine the range of values for X and Y axes
    x_min = x_values.min()
    x_max = x_values.max()
    y_min = y_values.min()
    y_max = y_values.max()

    # Determine the size of each axis
    x_axis_size = img_size[0] - 150
    y_axis_size = img_size[1] - 110

    # Determine the scale factor for X and Y axes
    x_scale = x_axis_size / (x_max - x_min)
    y_scale = y_axis_size / (y_max - y_min)

    # Define the start points for the lines
    x_start = 100
    y_start = img_size[1] - 50

    # Draw the X and Y axes
    cv2.line(img, (x_start, y_start), (x_start + x_axis_size, y_start), (0, 0, 0), 2)
    cv2.line(img, (x_start, y_start), (x_start, 50), (0, 0, 0), 2)

    # Draw the tick marks on the X and Y axes
    for i in range(x_values.size):
        x = int((x_values[i] - x_min) * x_scale) + x_start
        y = img_size[1] - 50
        cv2.line(img, (x, y), (x, y - 10), (0, 0, 0), 2)
        # cv2.putText(img, str(round(x_values[i],2)), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 0, 0), 2)

    for i in range(y_values.size):
        x = x_start
        y = img_size[1] - int((y_values[i] - y_min) * y_scale) - 50
        cv2.line(img, (x, y), (x + 10, y), (0, 0, 0), 2)
        cv2.putText(img, str(round(y_values[i],2)), (x - 62, y), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2)

    # Draw the scatter plot
    y_axis_velocity_data = np.array(calc_array[3])
    x_axis_velocity_data = np.array(calc_array[0])
    x_axis_velocity_data = x_axis_velocity_data[-len(y_axis_velocity_data):]
    # print(x_axis_velocity_data)    
    
    for i in range(x_axis_velocity_data.size - 1):
        x1 = int((x_axis_velocity_data[i] - x_min) * x_scale) + x_start
        y1 = img_size[1] - int((y_axis_velocity_data[i] - y_min) * y_scale) - 50
        x2 = int((x_axis_velocity_data[i+1] - x_min) * x_scale) + x_start
        y2 = img_size[1] - int((y_axis_velocity_data[i+1] - y_min) * y_scale) - 50
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
    # Set the initial position of the vertical bar
    bar_pos = int((frame_num - x_min) * x_scale) + x_start
    
    # the current frame number loops through every frame of the array
    # calc_array shows velocity at certain points of the frame, as it correlates to calc_array[0].
    # there are always less values in calc_array[3] than frames listed in calc_array[0]
    # 
    # How do I know when to start plotting velocity?
    # I can search the x_axis_velocity_data and y_axis_velocity data 
    # for the velocity values that correlate based on frame_num
    
    if frame_num in x_axis_velocity_data:
        index = np.where(x_axis_velocity_data == frame_num)[0]
        index = index[0]
        instant_velocity = y_axis_velocity_data[index]
        velocity_string = str(round(instant_velocity,2))
        velocity_string = velocity_string + " m/s" #"Inst. Vel: "
        cv2.putText(img, velocity_string, (bar_pos-55, 46), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 1)

    # Draw the vertical bar
    cv2.line(img, (bar_pos, 50), (bar_pos, img_size[1] - 50), (0, 0, 255), 2)
    
    print(current_frame_img.shape[0])
    percent = .25
    if (current_frame_img.shape[0] < 1000):
        percent = 0.35
    print(percent)
    x_offset_data=int(current_frame_img.shape[1]*percent)#-int(img.shape[1])
    y_offset_data=current_frame_img.shape[0]-(img.shape[0])
    
    # Define the Title text and position---------------------------------
    text = 'Instant Velocity Graph'
    
    x = int(img_size[0]/2)-135
    y = int(img_size[1]/14)
    position = (x, y)

    # Define the font and other properties of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 0, 0) # (B, G, R)
    thickness = 2
    
    # Add the text to the image
    cv2.putText(img, text, position, font, font_scale, color, thickness)
    
    #----------------------------------------------------------------------
    
    # Create Y Axis Label:-------------------------------------------------
    
    # Create a small blank white image
    text_img = np.zeros((40, 155, 3), dtype=np.uint8)
    text_img.fill(255)

    # Define the text and position
    text = 'Velocity (m/s)'
    position = (1, 20) # (x, y)

    # Define the font and other properties of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .65
    color = (0, 0, 0) # (B, G, R)
    thickness = 2

    # Add the text to the small image
    cv2.putText(text_img, text, position, font, font_scale, color, thickness)

    # Rotate the small image by 90 degrees
    rotated_text_img = cv2.rotate(text_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Define the position to place the small image on the bigger image
    x_offset = 0
    y_offset = int(img.shape[0]/4)

    # Place the rotated small image on the bigger image
    img[y_offset:y_offset+rotated_text_img.shape[0], x_offset:x_offset+rotated_text_img.shape[1]] = rotated_text_img        

    #----------------------------------------------------------------------
        
    # Add the graph to bottom of the original frame:
    current_frame_img[y_offset_data:y_offset_data+img.shape[0], x_offset_data:x_offset_data+img.shape[1]] = img
        
    return current_frame_img

def test_velocity_graph():
    width = 1280
    height = 1080
    current_frame = np.full((height, width, 3), 255, np.uint8)
    current_frame = add_velocity_graph(current_frame,3,[[1,2,3,4,5,6],[1,2,3,4,5,6],[3,3,3,3,3,3],[3,3,3,3,3,3]],4)
    cv2.imshow("Frame", current_frame)
    cv2.waitKey(0)
    return current_frame

# test_velocity_graph()