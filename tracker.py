from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np

class Tracker:
    """
    A tracking class that encapsulates object tracking functionality using DeepSORT.
    It initializes the DeepSORT tracker and encoder with specific parameters.
    """
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        """
        Initializes the Tracker class.

        It sets up the DeepSORT tracker with a specified maximum cosine distance and budget for the nearest neighbor metric.
        Additionally, it initializes the feature encoder using a pre-trained model.
        """
        # Maximum cosine distance for track matching
        max_cosine_distance = 0.4
        # Budget for the nearest neighbor metric, None means unlimited
        nn_budget = None

        # File path for the feature encoder model
        encoder_model_filename = 'model_data/mars-small128.pb'

        # Initialize the nearest neighbor metric with the specified maximum cosine distance and budget
        metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
        # Initialize the DeepSORT tracker with the above metric
        self.tracker = DeepSortTracker(metric)
        # Initialize the feature encoder with the specified model file and batch size
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)


    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            id = track.track_id

            track.append(Track(id, bbox))

        self.tracks = tracks

class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox