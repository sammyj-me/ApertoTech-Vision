class Frame:
    def __init__(self, landmarks, mp_pose):
        
        def get_nose():       
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            return nose

        def get_left_eye():
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
            return left_eye
        
        def get_right_eye():
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
            return right_eye
        
        def get_left_ear():
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            return left_ear
        
        def get_right_ear():
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            return right_ear
        
        def get_right_index():
            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
            return right_index
        
        def get_left_index():
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            return left_index
        
        def get_left_thumb():
            left_thumb = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y]
            return left_thumb
        
        def get_right_thumb():
            right_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]
            return right_thumb
        
        def get_left_wrist():
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            return left_wrist
        
        def get_right_wrist():
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            return right_wrist
        
        def get_left_pinky():
            left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
            return left_pinky
        
        def get_right_pinky():
            right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
            return right_pinky
        
        def get_left_shoulder():
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            return left_shoulder
        
        def get_left_elbow():
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            return left_elbow
        
        def get_wrist():
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            return wrist
       
        def get_left_foot_index():
            left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            return left_foot_index
        
        def get_right_foot_index():
            right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            return right_foot_index
    
        def get_left_hip():
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            return left_hip    

        def get_right_hip():
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            return right_hip
        
        def get_hip_avg():
            left_hip = get_left_hip()
            right_hip = get_right_hip()
            # print("left and right hip:")
            # print(left_hip,right_hip)
            
            left_hip_x = left_hip[0]
            right_hip_x = right_hip[0]
            
            left_hip_y = left_hip[1]
            right_hip_y = right_hip[1]
            num = 2
            hip_avg = [(left_hip_x + right_hip_x)/num,(left_hip_y + right_hip_y)/num]
            return hip_avg
    
        def get_right_knee():
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            return right_knee
    
        def get_left_knee():
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            return left_knee
        
        def get_left_ankle():
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            return left_ankle
        
        def get_right_ankle():        
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            return right_ankle
        
        def get_left_heel():
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            return left_heel
        
        def get_right_heel():
            right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            return right_heel
        
        def get_bounding_box(coordinates_list):
            
            left_x = 10000000            
            lower_y = 10000000
            right_x = 0
            upper_y = 0
            
            for attr in coordinates_list:
                x = attr[0]
                y = attr[1]
                
                if x <= left_x:
                    left_x = x
                
                if y <= lower_y:
                    lower_y = y
                    
                if x >= right_x:
                    right_x = x
                            
                if y >= upper_y:
                    upper_y = y
            
            lower_y =(lower_y-0.1)  # with 10% padding            
            upper_y = (upper_y+0.1)  # with 10% padding
            
            # Width aspect ratio correction:
            # the ratio between height to width must be ~1.4 (width is larger) for the 
            # 10-imagegallery view to fit an instagram story or equivalent
            # find pixels of height and width            
            height = abs(upper_y-lower_y) 
            # print("height: ", height)
            req_left_right_width_addition = (height*1.5)*0.5
            # print("req_left_right_width_addition: ", req_left_right_width_addition)
            
            right_x = (right_x+(req_left_right_width_addition/2))  # with aspect ratio padding
            left_x = (left_x-(req_left_right_width_addition/2))  # with aspect ratio padding   

            if (right_x > 1.00): #
                difference = abs(right_x-1.00)
                right_x = 1.00
                left_x = left_x-difference                
            
            if (left_x < 0.00):
                difference = abs(left_x)
                right_x = right_x+difference
                left_x = 0.00                
            
            bounding_box = (left_x, lower_y, right_x, upper_y) #
            return bounding_box
        
        self.left_shoulder = get_left_shoulder()
        self.left_elbow = get_left_elbow()
        self.wrist = get_wrist()
        self.left_foot_index = get_left_foot_index()
        self.right_foot_index = get_right_foot_index()
        
        self.left_hip = get_left_hip()
        self.right_hip = get_right_hip()
        self.hip_avg = get_hip_avg()
        
        self.left_knee = get_left_knee()
        self.right_knee = get_right_knee()
        
        self.left_ankle = get_left_ankle()
        self.right_ankle = get_right_ankle()
        self.left_heel = get_left_heel()
        self.right_heel = get_right_heel()
                
        self.nose = get_nose()
        
        self.left_eye = get_left_eye()
        self.right_eye = get_right_eye()
        
        self.left_ear = get_left_ear()
        self.right_ear = get_right_ear()
        
        self.right_index = get_right_index()
        self.left_index = get_left_index()
        
        self.left_thumb = get_left_thumb()
        self.right_thumb = get_right_thumb()
        
        self.left_wrist = get_left_wrist()
        self.right_wrist = get_right_wrist()
        
        self.left_pinky = get_left_pinky()
        self.right_pinky = get_right_pinky()
        
        self.all_points = [
        
        get_left_shoulder(),
        get_left_elbow(),
        get_wrist(),
        get_left_foot_index(),
        get_right_foot_index(),
        get_left_hip(),
        get_right_hip(),
        
        get_left_knee(),
        get_right_knee(),
        
        get_left_ankle(),
        get_right_ankle(),
        get_left_heel(),
        get_right_heel(),
                
        get_nose(),
        get_left_eye(),
        get_right_eye(),
        get_left_ear(),
        get_right_ear(),
        
        get_right_index(),
        get_left_index(),
        
        get_left_thumb(),
        get_right_thumb(),
        
        get_left_wrist(),
        get_right_wrist(),
        
        get_left_pinky(),
        get_right_pinky()]
        self.bounding_box = get_bounding_box(self.all_points)