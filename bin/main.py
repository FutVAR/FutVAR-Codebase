import argparse
from enum import Enum
from typing import Iterator, List, Union
from collections import deque

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import View_Transformer 
from speed_and_distance_estimator import SpeedAndDistance_Estimator

from inklet.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch, draw_pitch_voronoi_diagram
from inklet.common.ball import BallTracker, BallAnnotator
from inklet.common.team import TeamClassifier
from inklet.common.view import ViewTransformer
from inklet.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'models/best-dataset-v7.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'models/football-pitch-detection-v9.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'models/football-ball-detection-v2.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)
TRACE_LABEL_ANNOTATOR = sv.TraceAnnotator(
    # color=sv.ColorPalette.DEFAULT, 
    color=sv.Color.from_hex('#FF1493'), 
    position=sv.Position.CENTER, 
    trace_length=30, 
    thickness=2, 
    color_lookup=sv.ColorLookup.CLASS
)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    text_position=sv.Position.CENTER
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    VORONOI = 'VORONOI'
    BALL_TRACK = 'BALL_TRACK'
    LINE_PROJECTION = 'LINE_PROJECTION'
    CAMERA_ESTIMATOR = 'CAMERA_ESTIMATOR'
    POSSESSION = 'POSSESSION'
    SPEED_DISTANCE = 'SPEED_DISTANCE'
    ACTION = 'ACTION'



def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(ball_detections: sv.Detections, detections: sv.Detections,keypoints: sv.KeyPoints,color_lookup: np.ndarray) -> np.ndarray:
    # mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    # transformer = ViewTransformer(
    #     source=keypoints.xy[0][mask].astype(np.float32),
    #     target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    # )
    filter = keypoints.confidence[0] > 0.5
    frame_reference_points = keypoints.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )

    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    frame_ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=radar)
    
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar

def render_radar_run(ball_detections: sv.Detections,players_detections: sv.Detections,referees_detections: sv.Detections,keypoints: sv.KeyPoints) -> np.ndarray:
    filter = keypoints.confidence[0] > 0.5
    frame_reference_points = keypoints.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
    )

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_referees_xy = transformer.transform_points(points=referees_xy)

    radar = draw_pitch(CONFIG)
    radar = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=radar)
    radar = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 0],
            face_color=sv.Color.from_hex('00BFFF'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=radar)
    radar = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 1],
            face_color=sv.Color.from_hex('FF1493'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=radar)
    radar = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_referees_xy,
            face_color=sv.Color.from_hex('FFD700'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=radar)
    return radar

def render_voronoi(players_detections: sv.Detections,keypoints: sv.KeyPoints) -> np.ndarray:
    filter = keypoints.confidence[0] > 0.5
    frame_reference_points = keypoints.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    voronoi = draw_pitch(CONFIG)
    voronoi = draw_pitch_voronoi_diagram(
        config=CONFIG,
        team_1_xy=pitch_players_xy[players_detections.class_id == 0],
        team_2_xy=pitch_players_xy[players_detections.class_id == 1],
        team_1_color=sv.Color.from_hex('00BFFF'),
        team_2_color=sv.Color.from_hex('FF1493'),
        pitch=voronoi)
    return voronoi 

def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack()
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        ball_detections = detections[detections.class_id == BALL_CLASS_ID]
        player_detetions = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeeper_detections = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referee_detections = detections[detections.class_id == REFEREE_CLASS_ID]

        detector = sv.Detections.merge([player_detetions, goalkeeper_detections, referee_detections])

        annotated_frame = frame.copy()
        annotated_frame = TRACE_LABEL_ANNOTATOR.annotate(annotated_frame, ball_detections)
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detector)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detector)
        yield annotated_frame
        


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    tracker = sv.ByteTrack()
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    # ball_tracker = BallTracker(buffer_size=20)
    # ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    # def callback(image_slice: np.ndarray) -> sv.Detections:
    #     result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
    #     return sv.Detections.from_ultralytics(result)

    # slicer = sv.InferenceSlicer(
    #     callback=callback,
    #     overlap_filter_strategy=sv.OverlapFilter.NONE,
    #     slice_wh=(640, 640),
    # )
   

    for frame in frame_generator:
        result = ball_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        # detections = slicer(frame).with_nms(threshold=0.1)
        # detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = TRACE_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame

def resolve_goalkeepers_team_id_radar(players: sv.Detections,goalkeepers: sv.Detections) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)

def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run radar on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0] #1280
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # player detection
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        ball_detections = detections[detections.class_id == BALL_CLASS_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        detections = sv.Detections.merge([players, goalkeepers, referees])

        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame=ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)
        annotated_frame = TRIANGLE_ANNOTATOR.annotate(annotated_frame,ball_detections)

        h, w, _ = frame.shape
        radar = render_radar(ball_detections, detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame

def run_radar_run(source_video_path: str, device: str) -> Iterator[np.ndarray]:

    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    frame_generator = sv.get_video_frames_generator(
    source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(result)
        players_detections = detections[detections.class_id == PLAYER_CLASS_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops

    team_classifier = TeamClassifier(device="cuda")
    team_classifier.fit(crops)

    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=20, height=17
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        # ball, goalkeeper, player, referee detection

        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        ball_detections = detections[detections.class_id == BALL_CLASS_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_CLASS_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_CLASS_ID]
        players_detections = all_detections[all_detections.class_id == PLAYER_CLASS_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_CLASS_ID]

        # team assignment

        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)

        goalkeepers_detections.class_id = resolve_goalkeepers_team_id_radar(players_detections, goalkeepers_detections)

        referees_detections.class_id -= 1

        all_detections = sv.Detections.merge([
            players_detections, goalkeepers_detections, referees_detections])

        # frame visualization

        labels = [
            f"#{tracker_id}"
            for tracker_id
            in all_detections.tracker_id
        ]

        all_detections.class_id = all_detections.class_id.astype(int)

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate( scene=annotated_frame,
            detections=all_detections)
        annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
            detections=all_detections,
            labels=labels)
        annotated_frame = triangle_annotator.annotate(scene=annotated_frame,
            detections=ball_detections)

        # sv.plot_image(annotated_frame)

        players_detections = sv.Detections.merge([
            players_detections, goalkeepers_detections
        ])

        # detect pitch key points

        result = pitch_detection_model(frame, verbose=False)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)
        # annotated_frame = draw_points_on_pitch(
        #     config=CONFIG,
        #     xy=pitch_players_xy[players_detections.class_id == 0],
        #     face_color=sv.Color.from_hex('00BFFF'),
        #     edge_color=sv.Color.BLACK,
        #     radius=16,
        #     pitch=annotated_frame)
        # annotated_frame = draw_points_on_pitch(
        #     config=CONFIG,
        #     xy=pitch_players_xy[players_detections.class_id == 1],
        #     face_color=sv.Color.from_hex('FF1493'),
        #     edge_color=sv.Color.BLACK,
        #     radius=16,
        #     pitch=annotated_frame)
        # annotated_frame = draw_points_on_pitch(
        #     config=CONFIG,
        #     xy=pitch_referees_xy,
        #     face_color=sv.Color.from_hex('FFD700'),
        #     edge_color=sv.Color.BLACK,
        #     radius=16,
        #     pitch=annotated_frame)

        # sv.plot_image(annotated_frame)

        # annotated_frame = draw_pitch(CONFIG)
        # annotated_frame = draw_pitch_voronoi_diagram(
        #     config=CONFIG,
        #     team_1_xy=pitch_players_xy[players_detections.class_id == 0],
        #     team_2_xy=pitch_players_xy[players_detections.class_id == 1],
        #     team_1_color=sv.Color.from_hex('00BFFF'),
        #     team_2_color=sv.Color.from_hex('FF1493'),
        #     pitch=annotated_frame)

        # sv.plot_image(annotated_frame)
            # project ball, players and referies on pitch
        h, w, _ = frame.shape
        radar = render_radar_run(ball_detections,players_detections,referees_detections,key_points)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        # sv.plot_image(annotated_frame)
        yield annotated_frame

def resolve_goalkeepers_team_id_voronoi(players: sv.Detections,goalkeepers: sv.Detections) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)

def run_voronoi(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    frame_generator = sv.get_video_frames_generator(
    source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(result)
        players_detections = detections[detections.class_id == PLAYER_CLASS_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops

    team_classifier = TeamClassifier(device="cuda")
    team_classifier.fit(crops)

    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=20, height=17
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        # ball, goalkeeper, player, referee detection

        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        ball_detections = detections[detections.class_id == BALL_CLASS_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_CLASS_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_CLASS_ID]
        players_detections = all_detections[all_detections.class_id == PLAYER_CLASS_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_CLASS_ID]

        # team assignment

        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)

        goalkeepers_detections.class_id = resolve_goalkeepers_team_id_voronoi(players_detections, goalkeepers_detections)

        referees_detections.class_id -= 1

        all_detections = sv.Detections.merge([
            players_detections, goalkeepers_detections, referees_detections])

        # frame visualization

        labels = [
            f"#{tracker_id}"
            for tracker_id
            in all_detections.tracker_id
        ]

        all_detections.class_id = all_detections.class_id.astype(int)

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(scene=annotated_frame,
            detections=all_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame,
            detections=all_detections,
            labels=labels)
        annotated_frame = triangle_annotator.annotate(scene=annotated_frame,
            detections=ball_detections)

        # sv.plot_image(annotated_frame)
        players_detections = sv.Detections.merge([
            players_detections, goalkeepers_detections
        ])

        # detect pitch key points
        result = pitch_detection_model(frame, verbose=False)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)

        h, w, _ = frame.shape
        voronoi = render_voronoi(players_detections, key_points)
        voronoi = sv.resize_image(voronoi, (w // 2, h // 2))
        radar_h, radar_w, _ = voronoi.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, voronoi, opacity=0.5, rect=rect)
        yield annotated_frame

def replace_outliers_based_on_distance(positions: List[np.ndarray],distance_threshold: float) -> List[np.ndarray]:
    last_valid_position: Union[np.ndarray, None] = None
    cleaned_positions: List[np.ndarray] = []

    for position in positions:
        if len(position) == 0:
            # If the current position is already empty, just add it to the cleaned positions
            cleaned_positions.append(position)
        else:
            if last_valid_position is None:
                # If there's no valid last position, accept the first valid one
                cleaned_positions.append(position)
                last_valid_position = position
            else:
                # Calculate the distance from the last valid position
                distance = np.linalg.norm(position - last_valid_position)
                if distance > distance_threshold:
                    # Replace with empty array if the distance exceeds the threshold
                    cleaned_positions.append(np.array([], dtype=np.float64))
                else:
                    cleaned_positions.append(position)
                    last_valid_position = position

    return cleaned_positions

def interpolate_empty_positions(path: List[np.ndarray]) -> List[np.ndarray]:
    # Find indices of empty arrays
    empty_indices = [i for i, arr in enumerate(path) if arr.size == 0]

    for idx in empty_indices:
        # Find the previous non-empty array
        prev_idx = max(i for i in range(idx) if path[i].size > 0)

        # Find the next non-empty array (if it exists)
        next_indices = [i for i in range(idx + 1, len(path)) if path[i].size > 0]
        if not next_indices:
            # If no next non-empty array exists, skip interpolation
            continue

        next_idx = min(next_indices)

        # Linear interpolation between the previous and next non-empty arrays
        weight = (idx - prev_idx) / (next_idx - prev_idx)
        interpolated_value = (1 - weight) * path[prev_idx] + weight * path[next_idx]

        # Replace the empty array with the interpolated value
        path[idx] = interpolated_value

    return path

def run_balltrack(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    MAXLEN = 5
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    frame_generator = sv.get_video_frames_generator(source_video_path)

    path_raw = []
    M = deque(maxlen=MAXLEN)

    for frame in tqdm(frame_generator, total=video_info.total_frames):

        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        ball_detections = detections[detections.class_id == BALL_CLASS_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        result = pitch_detection_model(frame, verbose=False)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)

        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        pitch_reference_points = np.array(CONFIG.vertices)[filter]

        transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )
        M.append(transformer.m)
        transformer.m = np.mean(np.array(M), axis=0)

        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

        path_raw.append(pitch_ball_xy)

        path = [np.empty((0, 2), dtype=np.float32) if coorinates.shape[0] >= 2 else coorinates for coorinates in path_raw]

        path = [coorinates.flatten() for coorinates in path]

        # replace_outliers_based_on_distance

        MAX_DISTANCE_THRESHOLD = 500
        path = replace_outliers_based_on_distance(path, MAX_DISTANCE_THRESHOLD)

        # interpolate_empty_positions
        path = interpolate_empty_positions(path)
        # sv.plot_image(annotated_frame)

        h, w, _ = frame.shape
        radar = draw_pitch(CONFIG)
        radar = draw_paths_on_pitch(
            config=CONFIG,
            paths=[path],
            color=sv.Color.WHITE,
            pitch=radar)

        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)

        # annotated_frame = draw_pitch(CONFIG)
        # annotated_frame = draw_paths_on_pitch(
        #     config=CONFIG,
        #     paths=[path],
        #     color=sv.Color.WHITE,
        #     pitch=annotated_frame)
        yield annotated_frame

def run_projector(source_video_path: str, device: str) -> Iterator[np.ndarray]:

    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        thickness=2, edges=CONFIG.edges)
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex('#FF1493'),
        radius=8)
    vertex_annotator_2 = sv.VertexAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        radius=8)

    # frame_generator = sv.get_video_frames_generator(source_video_path, start=200)
    # frame = next(frame_generator)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)

        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        frame_reference_key_points = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])

        pitch_reference_points = np.array(CONFIG.vertices)[filter]
        transformer = ViewTransformer(
            source=pitch_reference_points,
            target=frame_reference_points
        )

        pitch_all_points = np.array(CONFIG.vertices)
        frame_all_points = transformer.transform_points(points=pitch_all_points)

        frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

        annotated_frame = frame.copy()
        annotated_frame = edge_annotator.annotate(
            scene=annotated_frame,
            key_points=frame_all_key_points)
        annotated_frame = vertex_annotator_2.annotate(
            scene=annotated_frame,
            key_points=frame_all_key_points)
        annotated_frame = vertex_annotator.annotate(
            scene=annotated_frame,
            key_points=frame_reference_key_points)

        # sv.plot_image(annotated_frame)
        return annotated_frame

def run_camera(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    # Read Video
    video_frames = read_video(source_video_path)

    # Initialize Tracker
    tracker = Tracker(PLAYER_DETECTION_MODEL_PATH)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # View Transformer
    view_transformer = View_Transformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    # save_video(output_video_frames, 'output/output_video.avi')
    return output_video_frames


def run_possession(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    pass

def run_speed_distance(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    pass

def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
        
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
        
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
        
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
        
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
        
    elif mode == Mode.RADAR:
        frame_generator = run_radar_run(
            source_video_path=source_video_path, device=device)
    
    elif mode == Mode.VORONOI:
        frame_generator = run_voronoi(
            source_video_path=source_video_path, device=device)
        
    elif mode == Mode.BALL_TRACK:
        frame_generator = run_balltrack(
            source_video_path=source_video_path, device=device)
  
    elif mode == Mode.LINE_PROJECTION:
        frame_generator = run_projector(
            source_video_path=source_video_path, device=device)
        
    elif mode == Mode.CAMERA_ESTIMATOR:
        frame_generator = run_camera(
            source_video_path=source_video_path, device=device)
        
    elif mode == Mode.POSSESSION:
        frame_generator = run_possession(
            source_video_path=source_video_path, device=device)
        
    elif mode == Mode.SPEED_DISTANCE:
        frame_generator = run_speed_distance(
            source_video_path=source_video_path, device=device)
        
    elif mode == Mode.ACTION:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device)
        
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    args = parser.parse_args()
    main(source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode
    )
