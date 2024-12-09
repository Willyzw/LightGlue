import cv2
from pathlib import Path
from tqdm import tqdm

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d

from dataset import VideoDataset
import rerun as rr              # pip install rerun-sdk
import rerun.blueprint as rrb

feature = 'aliked'  # or 'disk', 'sift', 'aliked', 'doghardnet'
if feature == 'superpoint':
    # SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher
elif feature == 'disk':
    # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
    extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher = LightGlue(features='disk').eval().cuda()  # load the matcher
elif feature == 'aliked':
    extractor = ALIKED(max_num_keypoints=2048).eval().cuda()
    matcher = LightGlue(features='aliked').eval().cuda()
elif feature == 'sift':
    extractor = SIFT(max_num_keypoints=2048).eval().cuda()
    matcher = LightGlue(features='sift').eval().cuda()

dataset = VideoDataset('./videos/kitti00', 'video.mp4')   

pbar = tqdm(total=dataset.num_frames)
img_id = 0

# Setup the blueprint
blueprint = rrb.Vertical(
    rrb.Horizontal(
        # rrb.Spatial3DView(name="3D", origin="/world"),
        rrb.Spatial2DView(name="Camera", origin="/world/camera/image"),
    ),
    rrb.Horizontal(
        rrb.Horizontal(
            rrb.TimeSeriesView(origin="/trajectory_error"),
            rrb.TimeSeriesView(origin="/trajectory_stats"),
            column_shares = [1,1]
        ),
        rrb.Spatial2DView(name="Trajectory 2D", origin="/trajectory_img/2d"),
        column_shares = [3,2],
    ),                
    row_shares=[3,2],  # 3 "parts" in the first Horizontal, 2 in the second
)        
rr.init("pyslam",  spawn=True, default_blueprint= blueprint)

while dataset.isOk():

    timestamp = dataset.getTimestamp()          # get current timestamp 
    curr_img_np = dataset.getImage(img_id)
    pbar.update(1)

    curr_img = numpy_image_to_torch(curr_img_np).cuda()
    if img_id == 0:
        last_img = curr_img
        feats0r = extractor.extract(last_img)  # auto-resize the image, disable with resize=None
    feats1r = extractor.extract(curr_img)

    # match the features
    matches01 = matcher({'image0': feats0r, 'image1': feats1r})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0r, feats1r, matches01]]  # remove batch dimension
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    pbar.set_description(f"Matches: {matches.shape[0]}")

    axes = viz2d.plot_images([last_img, curr_img])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    viz2d.save_plot(f"outputs/demo_{feature}_{img_id}_match.png")
    img = cv2.imread(f"outputs/demo_{feature}_{img_id}_match.png")
    rr.log("world/camera/image", rr.Image(img).compress(jpeg_quality=85))

    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([last_img, curr_img])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    viz2d.save_plot(f"outputs/demo_{feature}_{img_id}_point.png")

    img_id += 1
    last_img = curr_img
    