   player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    
    tracker = sv.ByteTrack()
    tracker.reset()

    frame_generator = sv.get_video_frames_generator(source_video_path)
    frame = next(frame_generator)

    # ball, goalkeeper, player, referee detection

    result = player_detection_model(frame, verbose=False)[0]
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
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1920, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        players_detections = detections[detections.class_id == PLAYER_CLASS_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops

    team_classifier = TeamClassifier(device="cpu")
    team_classifier.fit(crops)

    players = detections[detections.class_id == PLAYER_CLASS_ID]
    crops = get_crops(frame, players)

    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players, players_detections, goalkeepers_detections)

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



@@@@
    annotated_frame = frame.copy()
    annotated_frame = ELLIPSE_ANNOTATOR.annotate(
        scene=annotated_frame,
        detections=all_detections)
    annotated_frame = LABEL_ANNOTATOR.annotate(
        scene=annotated_frame,
        detections=all_detections,
        labels=labels)
    annotated_frame = TRIANGLE_ANNOTATOR.annotate(
        scene=annotated_frame,
        detections=ball_detections)

    sv.plot_image(annotated_frame)

    players_detections = sv.Detections.merge([
        players_detections, goalkeepers_detections
    ])

    # detect pitch key points
    result = pitch_detection_model(frame,imgsz=1920, verbose=False)[0]
    key_points = sv.KeyPoints.from_ultralytics(result)

    # project ball, players and referies on pitch

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
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

    annotated_frame = draw_pitch(CONFIG)
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=annotated_frame)
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_players_xy[players_detections.class_id == 0],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame)
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_players_xy[players_detections.class_id == 1],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame)
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_referees_xy,
        face_color=sv.Color.from_hex('FFD700'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame)

    annotated_frame = draw_pitch(CONFIG)
    annotated_frame = draw_pitch_voronoi_diagram(
        config=CONFIG,
        team_1_xy=pitch_players_xy[players_detections.class_id == 0],
        team_2_xy=pitch_players_xy[players_detections.class_id == 1],
        team_1_color=sv.Color.from_hex('00BFFF'),
        team_2_color=sv.Color.from_hex('FF1493'),
        pitch=annotated_frame)