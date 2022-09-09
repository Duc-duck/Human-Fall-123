# -*- coding: utf-8 -*-
"""
This script to extract skeleton joints position and score.

- This 'annot_folder' is a action class and bounding box for each frames that came with dataset.
    Should be in format of [frame_idx, action_cls, xmin, ymin, xmax, ymax]
        Use for crop a person to use in pose estimation model.
- If have no annotation file you can leave annot_folder = '' for use Detector model to get the
    bounding box.
"""

import os
import cv2
import time
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

from Track.Tracker import Detection, Tracker
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import vis_frame_fast

save_path = 'C:/Users/Victus/Desktop/Final_project/Home_new-pose+score_tracking_updated.csv'

annot_file = 'C:/Users/Victus/Desktop/Final_project/Annotations(1).csv'  # from create_dataset_1.py
video_folder = 'C:/Users/Victus/Desktop/Final_project/Videos'
annot_folder = ''  # bounding box annotation for each frame.

# DETECTION MODEL.
detector = TinyYOLOv3_onecls()

# POSE MODEL.
inp_h = 320
inp_w = 256
pose_estimator = SPPE_FastPose('resnet50', inp_h, inp_w,'cuda')

# Tracking
max_age = 30
tracker = Tracker(max_age=max_age, n_init=3)
    
# with score.
columns = ['video', 'frame', 'Nose_x', 'Nose_y', 'Nose_s', 'LShoulder_x', 'LShoulder_y', 'LShoulder_s',
           'RShoulder_x', 'RShoulder_y', 'RShoulder_s', 'LElbow_x', 'LElbow_y', 'LElbow_s', 'RElbow_x',
           'RElbow_y', 'RElbow_s', 'LWrist_x', 'LWrist_y', 'LWrist_s', 'RWrist_x', 'RWrist_y', 'RWrist_s',
           'LHip_x', 'LHip_y', 'LHip_s', 'RHip_x', 'RHip_y', 'RHip_s', 'LKnee_x', 'LKnee_y', 'LKnee_s',
           'RKnee_x', 'RKnee_y', 'RKnee_s', 'LAnkle_x', 'LAnkle_y', 'LAnkle_s', 'RAnkle_x', 'RAnkle_y',
           'RAnkle_s', 'label']

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy


annot = pd.read_csv(annot_file)
vid_list = annot['video'].unique()
#print('vid_list:',vid_list)
#vid_list1 = ['video(59).avi']

for vid in vid_list:
    print(f'Process on: {vid}')
    df = pd.DataFrame(columns=columns)
    cur_row = 0
    
    # Pose Labels.
    frames_label = annot[annot['video'] == vid].reset_index(drop=True)

    cap = cv2.VideoCapture(os.path.join(video_folder, vid))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Bounding Boxs Labels.
    annot_file1 = os.path.join(annot_folder, vid.split('.')[0], '.txt')
    #print('annot_file:',annot_file)
    annot1 = None
    if os.path.exists(annot_file1):
        annot = pd.read_csv(annot_file, header=None,
                                  names=['frame_idx', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        annot = annot.dropna().reset_index(drop=True)

        assert frames_count == len(annot), 'frame count not equal! {} and {}'.format(frames_count, len(annot))

    fps_time = 0
    i = 1
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cls_idx = int(frames_label[frames_label['frame'] == i]['label'])
            detected = detector.detect(frame)
            #print('detected:',i,'',detected)
            tracker.predict()
            for track in tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det
            detections = []
            
            if annot1:
                bb = np.array(annot.iloc[i-1, 2:].astype(int))
            else:
              if detected is not None:
                bb = detected[0, :4].numpy().astype(int)
                print('bb:',i,' ',bb,' ',type(bb))
              else:
                #continue
                bb = np.array((0, 0, 0, 0))
                print('bb:',i,' ',bb,' ',type(bb))
            bb[:2] = np.maximum(0, bb[:2] - 5)
            bb[2:] = np.minimum(frame_size, bb[2:] + 5) if bb[2:].any() != 0 else bb[2:]
            result = []
            if bb.any() != 0:
              result = pose_estimator.predict(frame, torch.tensor(bb[None, ...]),
                                              torch.tensor([[1.0]]))
              detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                      np.concatenate((ps['keypoints'].numpy(),
                                                      ps['kp_score'].numpy()), axis=1),
                                      ps['kp_score'].mean().numpy()) for ps in result]
            tracker.update(detections)
            if len(result) > 0:
                pt_norm = normalize_points_with_size(result[0]['keypoints'].numpy().copy(),
                                                      frame_size[0], frame_size[1])
                pt_norm = np.concatenate((pt_norm, result[0]['kp_score']), axis=1)

                #idx = result[0]['kp_score'] <= 0.05
                #pt_norm[idx.squeeze()] = np.nan
                row = [vid, i, *pt_norm.flatten().tolist(), cls_idx]
                scr = result[0]['kp_score'].mean()
            else:
                row = [vid, i, *[np.nan] * (13 * 3), cls_idx]
                scr = 0.0

            df.loc[cur_row] = row
            cur_row += 1

            # VISUALIZE.
            frame = vis_frame_fast(frame, result)
            frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
            frame = cv2.putText(frame, 'Frame: {}, Pose: {}, Score: {:.4f}'.format(i, cls_idx, scr),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame = frame[:, :, ::-1]
            fps_time = time.time()
            i += 1

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, mode='w', index=False)

