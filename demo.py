import cv2
import rerun as rr              # pip install rerun-sdk
import rerun.blueprint as rrb

from tqdm import tqdm
from glob import glob
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import rbd, numpy_image_to_torch
from lightglue import viz2d

N = 1024
feature = 'sift'  # or 'disk', 'sift', 'aliked', 'doghardnet'
if feature == 'superpoint':
    # SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=N).eval()  # load the extractor
elif feature == 'disk':
    # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
    extractor = DISK(max_num_keypoints=N).eval()  # load the extractor
elif feature == 'aliked':
    extractor = ALIKED(max_num_keypoints=N).eval()
elif feature == 'sift':
    extractor = SIFT(max_num_keypoints=N).eval()

matcher = LightGlue(feature).eval()  # load the matcher

imgs = sorted(glob('/data/slammy/color/*.jpg'))[::30]

# Setup the blueprint
blueprint = rrb.Vertical(
    rrb.Horizontal(
        # rrb.Spatial3DView(name="3D", origin="/world"),
        rrb.Spatial2DView(name="Camera", origin="/world/camera/image"),
    ),
    rrb.Horizontal(
        rrb.Spatial2DView(name="Camera2", origin="/world/camera/image2"),
    ),            
    row_shares=[3,2],  # 3 "parts" in the first Horizontal, 2 in the second
)
rr.init("SLAM_lab", spawn=True, default_blueprint= blueprint)

for img_id, img in tqdm(enumerate(imgs)):
    curr_img_np = cv2.imread(img)
    curr_img_np = cv2.cvtColor(curr_img_np, cv2.COLOR_BGR2RGB)
    curr_img = numpy_image_to_torch(curr_img_np)

    # extract the features
    if img_id == 0:
        last_img = curr_img
        feats0r = extractor.extract(last_img)  # auto-resize the image, disable with resize=None
        continue
    feats1r = extractor.extract(curr_img)

    # match the features
    matches01 = matcher({'image0': feats0r, 'image1': feats1r})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0r, feats1r, matches01]]  # remove batch dimension
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    axes = viz2d.plot_images([last_img, curr_img])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    rr.log("world/camera/image", rr.Image(viz2d.plot3()).compress(jpeg_quality=85))

    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([last_img, curr_img])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    rr.log("world/camera/image2", rr.Image(viz2d.plot3()).compress(jpeg_quality=85))
    feats0r = feats1r
    last_img = curr_img
