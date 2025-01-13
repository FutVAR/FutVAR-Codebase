
def render_vonoroi(detections: sv.Detections,keypoints: sv.KeyPoints,color_lookup: np.ndarray) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    # filter = key_points.confidence[0] > 0.5
    # frame_reference_points = key_points.xy[0][filter]
    # pitch_reference_points = np.array(CONFIG.vertices)[filter]

    # transformer = ViewTransformer(
    #     source=frame_reference_points,
    #     target=pitch_reference_points
    # )

    # frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    # pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    # players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    # pitch_players_xy = transformer.transform_points(points=players_xy)

    # referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    # pitch_referees_xy = transformer.transform_points(points=referees_xy)

    vonoroi = draw_pitch(CONFIG)
    vonoroi = draw_pitch_voronoi_diagram(
        config=CONFIG,
        team_1_xy=transformed_xy[color_lookup == 0],
        team_2_xy=transformed_xy[color_lookup == 1],
        team_1_color=sv.Color.from_hex('00BFFF'),
        team_2_color=sv.Color.from_hex('FF1493'),
        pitch=vonoroi)
    
    return vonoroi

=============

    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    # ball, goalkeeper, player, referee detection
    # tracker = sv.ByteTrack()
    # tracker.reset()

    # frame_generator = sv.get_video_frames_generator(source_video_path)
    # frame = next(frame_generator)
    # result = player_detection_model(frame, imgsz=1920, verbose=False)[0]
    # detections = sv.Detections.from_ultralytics(result)

    # ball_detections = detections[detections.class_id == BALL_CLASS_ID]
    # ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    # all_detections = detections[detections.class_id != BALL_CLASS_ID]
    # all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    # all_detections = tracker.update_with_detections(detections=all_detections)

    # goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_CLASS_ID]
    # players_detections = all_detections[all_detections.class_id == PLAYER_CLASS_ID]
    # referees_detections = all_detections[all_detections.class_id == REFEREE_CLASS_ID]

# radar
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1920, verbose=False)[0] #1280
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1920, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)

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
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)
        
        annotated_frame = TRIANGLE_ANNOTATOR.annotate(annotated_frame, detections, custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        vonoroi = render_vonoroi(detections, keypoints, color_lookup)
        vonoroi = sv.resize_image(vonoroi, (w // 2, h // 2))
        vonoroi_h, vonoroi_w, _ = vonoroi.shape
        rect = sv.Rect(
            x=w // 2 - vonoroi_w // 2,
            y=h - vonoroi_h,
            width=vonoroi_w,
            height=vonoroi_h
        )
        annotated_frame = sv.draw_image(annotated_frame, vonoroi, opacity=0.5, rect=rect)



        ====== edited ======
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
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)

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
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)
        annotated_frame = TRIANGLE_ANNOTATOR.annotate(annotated_frame, detections, custom_color_lookup=color_lookup)
        
        # all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])
        # all_detections.class_id = all_detections.class_id.astype(int)

        # annotated_frame = frame.copy()
        # annotated_frame = ellipse_annotator.annotate(
        #     scene=annotated_frame,
        #     detections=all_detections)
        # annotated_frame = label_annotator.annotate(
        #     scene=annotated_frame,
        #     detections=all_detections,
        #     labels=labels)
        # annotated_frame = triangle_annotator.annotate(
        #     scene=annotated_frame,
        #     detections=ball_detections)

        players = sv.Detections.merge([
            players, goalkeepers])


        # project ball, players and referies on pitch
        filter = keypoints.confidence[0] > 0.5
        frame_reference_points = keypoints.xy[0][filter]
        pitch_reference_points = np.array(CONFIG.vertices)[filter]

        transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )

        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_players_xy = transformer.transform_points(points=players_xy)

        h, w, _ = frame.shape
        radar = draw_pitch(CONFIG)
        radar = draw_pitch_voronoi_diagram(
            config=CONFIG,
            team_1_xy=pitch_players_xy[players.class_id == 0],
            team_2_xy=pitch_players_xy[players.class_id == 1],
            team_1_color=sv.Color.from_hex('00BFFF'),
            team_2_color=sv.Color.from_hex('FF1493'),
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