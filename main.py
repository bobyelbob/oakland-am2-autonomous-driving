import motor_controller as mc
import image_processing as ip
import cv2
import numpy as np
import time
import math

# video = cv2.VideoCapture(0)
# video.set(cv2.CAP_PROP_FRAME_WIDTH,320) # width = 320
# video.set(cv2.CAP_PROP_FRAME_HEIGHT,240) # height = 240

file_size = (320,240) # Assumes 1280x720 mp4
output_filename = 'lane_following_55.mp4'
output_frames_per_second = 20.0 

if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(output_filename,  
                        fourcc, 
                        output_frames_per_second, 
                        file_size) 

    while cap.isOpened():

        # Capture one frame only
        success, frame = cap.cv2.imread()
        
        # frame = cv2.flip(frame,-1)
        frame = cv2.imread()
        original_frame = frame.copy

        processed_frame = ip.process_frame(original_frame)

        result.write(processed_frame)     

        # if success:
        #     original_frame = frame.copy

        #     processed_frame = ip.process_frame(original_frame)

        #     result.write(processed_frame)     
        
        # else:
        #     break
    
    cap.release()
    result.release()
    cv2.destroyAllWindows()

    # while True:
    #     ret,frame = video.read()
    #     frame = cv2.flip(frame, -1) #flip image vertically
    #     cv2.imshow('original',frame)
    #     # cv2.imwrite('original.jpg',frame)
        
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break
        
    # video.release()
    # cv2.destroyAllWindows()
        