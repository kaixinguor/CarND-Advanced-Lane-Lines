import cv2
from lane_fit import lane_fit
from matplotlib import pyplot as plt
from myutil import ensure_dir


if __name__ == '__main__':
    output_dir = 'output_images/project_video_result'
    ensure_dir(output_dir)

    vidcap = cv2.VideoCapture('project_video.mp4')
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_images/project_video_result.avi', fourcc, fps, (width, height))

    count = 0
    while True:
        success, image = vidcap.read()
        if success is False:
            break

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        result = lane_fit(image)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite('output_images/project_video_result/{:04d}.jpg'.format(count),result)
        out.write(result)
        count += 1

    print("width: ", width, ", height: ", height)
    print("frame: ", length)
    print("fps: ", fps)
    print("count:", count)
    vidcap.release()
    out.release()