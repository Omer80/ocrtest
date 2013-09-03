# coding=utf-8
__author__ = 'Roman Podlinov'
import os
import logging
import sys
from file_helper import FileHelper


def cut_frames_from_video(path, videofile):
    videofile_abspath = os.path.join(path,videofile)
    if os.path.exists(videofile_abspath) == False:
        logging.error("File "+ videofile_abspath + " doesn't exists or permission denied")
        return None

    parts = videofile.split('.')
    frames_path = os.path.join(path, parts[0])
    if os.path.exists(frames_path):
        FileHelper.remove_files_in_dir(frames_path)
    else:
        os.mkdir(frames_path)

    try:
        images_template = frames_path + "/{0}_%05d.jpg".format(parts[0][:10])
        #ffmpeg -i input.flv -f image2 -vf fps=fps=1 out%04d.png
        cmd = "ffmpeg -i {0} -f image2 -vf fps=fps=1 {1}".format(videofile_abspath, images_template)
        logging.debug(cmd)
        os.system(cmd)
        return 1
    except:
        logging.error(sys.exc_info())
        return None


def main(path = './'):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(('.mp4')):
                cut_frames_from_video(path, filename)



if __name__ == "__main__":
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)

    if len(sys.argv) < 2:
        print 'USAGE:\n\t' + sys.argv[0] + ' inFolder'
        sys.exit(1)

    main(sys.argv[1])