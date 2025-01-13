import cv2
import supervision as sv
import sys 
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=30 # 24
    
    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance= {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,number_of_frames-1)

                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position,end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed',None)
                       distance = track_info.get('distance',None)
                       if speed is None or distance is None:
                           continue
                       
                       bbox = track_info['bbox']
                       position = get_foot_position(bbox)
                       position = list(position)
                       position[1]+=40

                       position = tuple(map(int,position))

                    #    annotated_frame = frame.copy()
                    #    labels = [str(tracker_id) for tracker_id in detections.tracker_id]

                    #    annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
                    #    annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                    #    annotated_frame, detections, labels=labels)
                    #    labels = [f"{class_name,} {confidence:.2f}" for class_name, confidence in zip(detections['class_name'], detections.confidence)

                    #    annotated_frame = frame.copy()
                    #    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
                    #    label_annotator.annotate(scene=annotated_frame, detections=position,labels=f"{speed:.2f} km/h")
                    #    label_annotator.annotate(scene=annotated_frame, detections=(position[0],position[1]+20),labels=f"{distance:.2f} km/h")

                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            output_frames.append(frame)
        
        return output_frames