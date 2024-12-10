import cv2
import rerun as rr              # pip install rerun-sdk
import rerun.blueprint as rrb

from tqdm import tqdm
from glob import glob
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import rbd, numpy_image_to_torch
from lightglue import viz2d

N = 1024
imgs = sorted(glob('/data/slammy/color/*.jpg'))[300::30]


# Load the extractor and matcher
extractors, matchers = {}, {}
for feature in ['sift', 'superpoint', 'aliked']:
    if feature == 'superpoint':
        extractors[feature] = SuperPoint(max_num_keypoints=N).eval()  # load the extractor
    elif feature == 'aliked':
        extractors[feature] = ALIKED(max_num_keypoints=N).eval()
    elif feature == 'sift':
        extractors[feature] = SIFT(max_num_keypoints=N).eval()
    matchers[feature] = LightGlue(feature).eval()  # load the matcher



# Setup the rerun viewer
blueprint = rrb.Vertical(
    *[rrb.Horizontal(
        rrb.Spatial2DView(name=f"features_{f}", origin=f"/world/camera/features_{f}"),
        rrb.Spatial2DView(name=f"matches_{f}", origin=f"/world/camera/matches_{f}"))
    for f in ['sift', 'superpoint', 'aliked']])
rr.init("SLAM_lab", spawn=True, default_blueprint= blueprint)


# Extract and match the features
last_feats = {}
for img_id, img in enumerate(tqdm(imgs)):
    rr.set_time_sequence("time", img_id)
    curr_img_np = cv2.imread(img)
    curr_img_np = cv2.cvtColor(curr_img_np, cv2.COLOR_BGR2RGB)
    curr_img = numpy_image_to_torch(curr_img_np)

    for feature, extractor in extractors.items():
        # extract the features
        if img_id == 0:
            last_img = curr_img
            last_feats[feature] = extractors[feature].extract(last_img)  # auto-resize the image, disable with resize=None
            continue
        feats1r = extractor.extract(curr_img)

        # match the features
        matches01 = matchers[feature]({'image0': last_feats[feature], 'image1': feats1r})
        kpts0, kpts1, matches = last_feats[feature]["keypoints"].squeeze(), feats1r["keypoints"].squeeze(), matches01["matches"][0]

        # plot the matches
        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"].squeeze()), viz2d.cm_prune(matches01["prune1"].squeeze())
        viz2d.plot_images([last_img, curr_img])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        rr.log(f"world/camera/features_{feature}", rr.Image(viz2d.plot3()).compress(jpeg_quality=85))
    
        viz2d.plot_images([last_img, curr_img])
        viz2d.plot_matches(kpts0[matches[...,0]], kpts1[matches[...,1]], color="lime", lw=0.2)
        rr.log(f"world/camera/matches_{feature}", rr.Image(viz2d.plot3()).compress(jpeg_quality=85))


        last_feats[feature] = feats1r
    last_img = curr_img
