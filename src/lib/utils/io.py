import pprint
import json
import glob
import cv2
import os

def my_print(obj, file=None):
    if not isinstance(obj, str):
        obj = pprint.pformat(obj)
    print(obj)
    if file is not None:
        print(obj, file=file)

def save_json(in_dict, save_path):
    json_str = json.dumps(in_dict, indent=4)
    with open(save_path, 'w') as json_file:
        json_file.write(json_str)
    json_file.close()


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data


def images2video(im_dir, out_video_path=None, ext='.jpg', size=None, fps=10):
    """
    Change some images to a avi video using opencv
    Args:
        im_dir: the dir contains the images
        out_video_path: the output path of the video
        ext: str, the extension of the images
        size: (h, w), the size of video
        fps: the fps of videos

    Returns:

    """
    im_list = glob.glob(os.path.join(im_dir, '*'+ext))
    im_list.sort()
    im_list = im_list[0:400]

    if out_video_path is None:
        out_video_path = os.path.join(im_dir, '..', 'video.avi')
    if size is None:
        im = cv2.imread(im_list[0])
        size = (im.shape[1], im.shape[0]) # [w, h]

    video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    for im_path in im_list:
        im = cv2.imread(im_path)
        im = cv2.resize(im, size)
        video.write(im)
    video.release()
    print('The images in {} are compressed to video {}'.format(im_dir, out_video_path))
    #cv2.destroyAllWindows()
