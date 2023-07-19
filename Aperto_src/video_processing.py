import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from aperto_object_detection import*
from Body_Coordinates import Frame
from moviepy.editor import VideoFileClip

from draw_image_stats import*
from velocity_graph import*
from resize_image import*

from tkinter import filedialog

def open_file_dialog():
    file_path = filedialog.askopenfilename()
    return file_path

class Analyzer:
    def __init__(self, file):
        self.mp_drawing = mp.solutions.drawing_utils     # Connecting Keypoints Visuals
        self.mp_pose = mp.solutions.pose                 # Keypoint detection model

        self.cap = cv2.VideoCapture(file)
        
        # Recolor image to RGB
        self.ret, self.frame = self.cap.read()
        self.image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.video_width = self.frame.shape[1]
        self.video_height = self.frame.shape[0]
        self.frame_rate = 0

        # Process Scale From First Frame of Video
        self.meters_per_pixel, self.angle, self.points = get_meters_per_pixel_and_angle_from_fiducials(file)

        # print("meters per pixel:", meters_per_pixel)

        self.image, ResizeWithAspectRatio(self.image, 256)
        self.image.flags.writeable = False

        #image.flags.writeable = False
        print("found image frame")
        self.nose_velocity = [[],[],[]]
        self.toe_ankle_heel_left = [[],[],[],[]]
        self.toe_ankle_heel_right = [[],[],[],[]]
        self.frames = []
        self.file = file
        
        self.toe_ankle_heel_aggregated = [[], []]
        
        self.max_v = 0
        self.max_v_meters_per_second = 0
        self.max_v_miles_per_hour = 0
        
        self.calc_array = [[],[],[],[],[]]
    
    def create_smooth_velocity_points_array(self, coordinates_list, frame_rate, video_width, meters_per_pixel):        
        
        if frame_rate == 240:
            frame_sample_rate = 3
        else:
            frame_sample_rate = 1
            
        # drop frames before toe off frame 1 and after full support frame 2 
        subtract_val = coordinates_list[0][0]
        
        for i in range(0, len(coordinates_list[0])):        
            # cutoff_frame = int(cutoff_percent*len(coordinates_list[0]))
            # focus_frame = coordinates_list[0][i] - subtract_val

            # if (focus_frame >= cutoff_frame and
            #     focus_frame <= len(coordinates_list[0]) - cutoff_frame):
                #                                                        and
                # coordinates_list[0][i]%frame_sample_rate == 0)):
            
            self.calc_array[0].append(coordinates_list[0][i])       # calc array[0] is the frame # array
            # print(coordinates_list[1][i][0])
            self.calc_array[1].append(coordinates_list[1][i][0])    # calc array[1] is the position array
            
        window_size = 1 #int(len(self.calc_array[0])/2) # define window size for moving average

        self.calc_array[2] = np.convolve(self.calc_array[1], np.ones(window_size)/window_size, "valid") # convolve -> creates moving average of position...this is of the position values
        
        # Now creating velocity column:
        time_prev = coordinates_list[0][0]/frame_rate # time 0
        position_prev = self.calc_array[2][0] # position 0
        self.calc_array[2] = list(self.calc_array[2])

        velocity_lower_threshold = 0
        velocity_upper_threshold = 13

        for x in range(1,len(self.calc_array[2])): # moving average velocity calculation 
            time = self.calc_array[0][x]/frame_rate
            
            delta_t = abs(time - time_prev)
            position = self.calc_array[2][x]
            delta_x = (abs(position - position_prev)*video_width)*meters_per_pixel # pixel position after moving average is applied

            velocity = delta_x/delta_t
            if velocity > velocity_lower_threshold and velocity < velocity_upper_threshold:
                self.calc_array[3].append(velocity) # convert to velocity
            
            time_prev = time
            position_prev = position
        
        window_size_2 = int(len(self.calc_array[2])/4)
        moving_averaged_velocity = np.convolve(self.calc_array[3], np.ones(window_size_2)/window_size_2, "valid") # convolve -> creates moving average of position...this is of the position values
        self.calc_array[3] = moving_averaged_velocity
        
        return self.calc_array
        

    def calculate_max_velocity(self, calc_array):    
        max_v = max(self.calc_array[3]) # return highest value in coordinates list
        return max_v

    def find_stride_length(data, video_width, meters_per_pixel):
        # assuming your data set is stored in a list called 'data'
        coords = np.array(data)
        
        # extract x and y coordinates
        x = [abs(coord) for coord in coords[0]]
        y = [abs(coord) for coord in coords[1]]
    
        n, bins, patches = plt.hist(x, bins=200, range=(0, 1))
        standard_deviation = np.std(n)
        mean = np.mean(n)
        
        foot_placement_points_percent = [[],[]]
        for i, value in enumerate(n):
            if value >= 5:
                foot_placement_points_percent[0].append(i/len(n))
                foot_placement_points_percent[1].append(value)
        
        # Check for values less than 0.2 apart and average them
        for i in range(0, len(foot_placement_points_percent[0]) - 1):
            try:
                if abs(foot_placement_points_percent[0][i] - foot_placement_points_percent[0][i+1]) < 0.125:
                    val = (foot_placement_points_percent[1][i] - foot_placement_points_percent[1][i+1])
                    if val > 0:                        
                        # foot_placement_points_percent[0][i] = (foot_placement_points_percent[i] + foot_placement_points_percent[i+1]) / 2
                        # foot_placement_points_percent[1][i] = (foot_placement_points_percent[i] + foot_placement_points_percent[i+1]) / 2
                        foot_placement_points_percent[0].pop(i)
                        foot_placement_points_percent[1].pop(i)
                    elif val < 0:
                        foot_placement_points_percent[0].pop(i+1)
                        foot_placement_points_percent[1].pop(i+1)
                    elif val == 0:
                        foot_placement_points_percent[0][i] = (foot_placement_points_percent[i] + foot_placement_points_percent[i+1]) / 2
                        foot_placement_points_percent[1][i] = (foot_placement_points_percent[i] + foot_placement_points_percent[i+1]) / 2
                        foot_placement_points_percent[0].pop(i)
                        foot_placement_points_percent[1].pop(i)
            except:
                pass
        
        # Check for values less than 0.2 apart and average them
        for i in range(0, len(foot_placement_points_percent[0]) - 1):
            try:
                if abs(foot_placement_points_percent[0][i] - foot_placement_points_percent[0][i+1]) < 0.125:
                    val = (foot_placement_points_percent[1][i] - foot_placement_points_percent[1][i+1])
                    if val > 0:                        
                        # foot_placement_points_percent[0][i] = (foot_placement_points_percent[i] + foot_placement_points_percent[i+1]) / 2
                        # foot_placement_points_percent[1][i] = (foot_placement_points_percent[i] + foot_placement_points_percent[i+1]) / 2
                        foot_placement_points_percent[0].pop(i)
                        foot_placement_points_percent[1].pop(i)
                    elif val < 0:
                        foot_placement_points_percent[0].pop(i+1)
                        foot_placement_points_percent[1].pop(i+1)
                    elif val == 0:
                        foot_placement_points_percent[0][i] = (foot_placement_points_percent[i] + foot_placement_points_percent[i+1]) / 2
                        foot_placement_points_percent[1][i] = (foot_placement_points_percent[i] + foot_placement_points_percent[i+1]) / 2
                        foot_placement_points_percent[0].pop(i)
                        foot_placement_points_percent[1].pop(i)
            except:
                pass
        
        plt.show()
        # print(foot_placement_points_percent)
        # print("foot placement percents^")
        
        stride_length = 0
        for i in range(0, len(foot_placement_points_percent[0])):
            try:
                stride_length = stride_length + ((foot_placement_points_percent[0][i+1]-foot_placement_points_percent[0][i])*video_width*meters_per_pixel)
            except:
                pass
            
        stride_length = stride_length/(len(foot_placement_points_percent[0])-1)
        
        return stride_length
        
    def capture_body_coordinate_data(self, frame_data):                    
        self.nose_velocity[0].append(self.cap.get(1))
        self.nose_velocity[1].append(frame_data.nose)

        self.toe_ankle_heel_left[0].append(self.cap.get(1))
        self.toe_ankle_heel_left[1].append(frame_data.left_ankle)
        self.toe_ankle_heel_left[2].append(frame_data.left_heel)

        self.toe_ankle_heel_right[0].append(self.cap.get(1))
        self.toe_ankle_heel_right[1].append(frame_data.right_ankle)
        self.toe_ankle_heel_right[2].append(frame_data.right_heel)
                            
    def aggregate_toe_ankle_heel(self):
        
        for x in range(1, 3):
            if len(self.toe_ankle_heel_left[x]) == 0:
                break

            self.toe_ankle_heel_right[x] = np.array(self.toe_ankle_heel_right[x])
            self.toe_ankle_heel_left[x] = np.array(self.toe_ankle_heel_left[x])

            self.toe_ankle_heel_right[x] = level_coordinates(self.toe_ankle_heel_right[x], self.angle)
            self.toe_ankle_heel_left[x] = level_coordinates(self.toe_ankle_heel_left[x], self.angle)

            plt.scatter(self.toe_ankle_heel_right[x][:, 0], self.toe_ankle_heel_right[x][:, 1], label='skewed points')

            self.toe_ankle_heel_aggregated[0].extend(value[0] for value in self.toe_ankle_heel_right[x])
            self.toe_ankle_heel_aggregated[1].extend(value[1] for value in self.toe_ankle_heel_left[x])

        plt.legend()
        plt.show()

        # return toe_ankle_heel_aggregated
    
    def calculate_stride_length_and_frequency(self, toe_ankle_heel_aggregated, video_width, meters_per_pixel, max_v_meters_per_second):
        try:
            stride_length =  round(self.find_stride_length(toe_ankle_heel_aggregated, video_width, meters_per_pixel), 2)
            stride_frequency =  round(max_v_meters_per_second / stride_length, 2)
            print("stride length:", round(stride_length, 2))
            print("stride frequency:", round(stride_frequency, 2))
        except:
            stride_length = "VNC"
            stride_frequency = "VNC"
        return stride_length, stride_frequency

    def create_values_array(self, max_v_miles_per_hour, max_v_meters_per_second, stride_length, stride_frequency):
        values = [
            "Peak Velocity (mph):", str(max_v_miles_per_hour),
            "Peak Velocity (m/s):", str(max_v_meters_per_second), 
            "Flying 10m (sec):", str(round((10 / max_v_meters_per_second), 2)), 
            "Strides/Sec:", str(stride_frequency),
            "Stride Length (m):", str(stride_length),
        ]
        print(values)
        return values

    def get_output_path(self, file):
        filename = os.path.basename(file)
        filename_without_ext = os.path.splitext(filename)[0]
        filename_without_ext= filename_without_ext + ".mp4"
        output_path = "output/" + filename_without_ext
        return output_path
        
    def add_logo_and_stats_to_frame(self, frame, aperto_overlay, aperto_stats):
        y_offset_logo=0
        x_offset_logo=0
        x_offset_data=0
        y_offset_data=frame.shape[0] - aperto_stats.shape[0]
        frame[y_offset_logo:y_offset_logo + aperto_overlay.shape[0], x_offset_logo:x_offset_logo + aperto_overlay.shape[1]] = aperto_overlay
        frame[y_offset_data:y_offset_data + aperto_stats.shape[0], x_offset_data:x_offset_data + aperto_stats.shape[1]] = aperto_stats
        return frame
    
    def write_frames_to_output(self, frames, aperto_overlay, aperto_stats, out):
        total_frames = len(frames)
        for frame_num, current_frame in enumerate(frames): 
            current_frame = ResizeWithAspectRatio(current_frame, width=1920)    
            current_frame = self.add_logo_and_stats_to_frame(current_frame, aperto_overlay, aperto_stats)
            current_frame = add_velocity_graph(current_frame, frame_num+1, self.calc_array, total_frames)        

            out.write(current_frame)  # Write the frame to the video file

            cv2.imshow("Frame", current_frame)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
    
    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image = ResizeWithAspectRatio(image, width=int(self.video_width*3))
        return image

    def process_results(self, results):
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_data = Frame(landmarks, self.mp_pose)
            self.capture_body_coordinate_data(frame_data)
            
    def handle_results(self, results, image):
        if results:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            image = ResizeWithAspectRatio(image, 900)
            cv2.imshow('Sprints', image)

    def post_processing(self):
        clip = VideoFileClip(self.file)
        num_frames = sum(1 for _ in clip.iter_frames())
        
        self.frame_rate = self.calculate_frame_rate(num_frames, clip.duration)
        print("duration: ", clip.duration)
        print("video fps: ", self.frame_rate)
        
        meters_per_second_to_miles_per_hour = 2.23694
        
        self.nose_velocity[1] = level_coordinates(np.array(self.nose_velocity[1]), self.angle)
        
        self.calc_array = self.create_smooth_velocity_points_array(self.nose_velocity, self.frame_rate, self.video_width, self.meters_per_pixel)
        self.max_v = self.calculate_max_velocity(self.calc_array)
        self.max_v_meters_per_second = round(self.max_v,2)
        self.max_v_miles_per_hour = round(self.max_v * meters_per_second_to_miles_per_hour,2)
        # toe_ankle_heel_aggregated = self.aggregate_toe_ankle_heel()
    
    def calculate_frame_rate(self, num_frames, duration):
        return 1 / (duration / num_frames)
    
    def analyze_file(self):
        with self.mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.6, model_complexity = 1) as pose:
            past_frame = None
            while self.ret:
                try:
                    print("running loop")
                    self.frames.append(self.frame)
                    image = self.process_frame(self.frame)

                    results = pose.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    results = pose.process(image)
                    self.process_results(results)

                    print(self.cap.get(1))

                    if cv2.waitKey(3) & 0xFF == ord('q'):
                        break
                    elif past_frame != self.cap.get(1):
                        past_frame = self.cap.get(1)
                    else:
                        cv2.destroyAllWindows()
                        break

                except Exception as e:
                    print("Error:", e)

                self.handle_results(results, image)
                self.ret, self.frame = self.cap.read()

            self.post_processing()
            self.aggregate_toe_ankle_heel()
            stride_length, stride_frequency = self.calculate_stride_length_and_frequency(self.toe_ankle_heel_aggregated, self.video_width, self.meters_per_pixel, self.max_v_meters_per_second)
            values = self.create_values_array(self.max_v_miles_per_hour, self.max_v_meters_per_second, stride_length, stride_frequency)
            
            aperto_overlay = url_to_image('https://i.ibb.co/K6Wzt3N/aperto-logo.jpg')
            aperto_overlay = ResizeWithAspectRatio(aperto_overlay, width=400)
            aperto_stats = draw_image_stats(values)
            aperto_stats = ResizeWithAspectRatio(aperto_stats, width=400)
            
            output_path = self.get_output_path(self.file)
            fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
            out = cv2.VideoWriter(output_path, fourcc, int(self.frame_rate), (1920, 1080), True)

            self.write_frames_to_output(self.frames, aperto_overlay, aperto_stats, out)
            
            self.cap.release()
            out.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    path = open_file_dialog()
    file_to_analyze = Analyzer(path)
    file_to_analyze.analyze_file()