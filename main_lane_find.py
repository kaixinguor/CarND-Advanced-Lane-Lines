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
        result, left_curverad, right_curverad, offset_dist = lane_fit(image)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.putText(result, "Radius of Curvature = {:4.0f} m".format(left_curverad), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),5)
        if offset_dist>0:
            cv2.putText(result, "Vehicle is {:.02f} m right of center".format(offset_dist), (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        else:
            cv2.putText(result, "Vehicle is {:.02f} m left of center".format(-offset_dist), (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        # ax.text(50, 30, "Radius of Curvature = {:4.0f} m".format(left_curverad), ha='left', va='top', color='white',
        #         fontsize=20)
        # ax.text(50, 100, "Vehicle is {:.02f} m right of center".format(offset_dist), ha='left', va='top', color='white',
        #         fontsize=20)

        cv2.imwrite('output_images/project_video_result/{:04d}.jpg'.format(count),result)
        out.write(result)
        count += 1

    print("width: ", width, ", height: ", height)
    print("frame: ", length)
    print("fps: ", fps)
    print("count:", count)
    vidcap.release()
    out.release()