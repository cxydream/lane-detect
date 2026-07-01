import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import glob
import copy
from moviepy import VideoFileClip
from IPython.display import HTML
from numpy.linalg import inv

# ==========================================
# 0. 全局配置与缓存初始化
# ==========================================
# 图像空间到现实世界空间（米）的像素转换系数
ym_per_pix = 30 / 720   # y轴方向：每像素对应30米
xm_per_pix = 3.7 / 700  # x轴方向：每像素对应3.7米

# 平滑滤波缓存列表
left_fitx_list = []
right_fitx_list = []
left_peak_list = []
right_peak_list = []
left_curverad_list = []
right_curverad_list = []
 
# ==========================================
# 1. 核心算法模块
# ==========================================

def undistort(img):
    """ 步骤 1: 相机畸变校正 """
    dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted


def color_gradient_thresh(img, s_thresh=(170, 255), l_thresh=(30, 255), sx_thresh=(65, 100)):
    """ 步骤 2: 色彩与梯度阈值过滤 (提取车道线特征) """
    img = np.copy(img)
    # 转换至 HLS 空间并分离通道
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    
    # 计算 X 方向的 Sobel 导数
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    # 结合 S 通道、L 通道和梯度阈值进行二值化
    color_gradient_binary = np.zeros_like(s_channel)
    condition = (((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])) & 
                 ((l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1]))) | \
                ((scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]))
    
    color_gradient_binary[condition] = 1
    return color_gradient_binary


def perspective_transform(undist):
     
    xoffset = 0 
    yoffset = 0
    img_size = (undist.shape[1], undist.shape[0]) # (w, h)

    # 设定原图的四个特征点与目标鸟瞰图的四个映射点
    src = np.float32([(550, 450), (730, 450), (1150, 700), (160, 700)])
    dst = np.float32([[xoffset, yoffset], 
                      [img_size[0] - xoffset, yoffset], 
                      [img_size[0] - xoffset, img_size[1] - yoffset], 
                      [xoffset, img_size[1] - yoffset]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped, M
  

def getCurvatureForLanes(processed_img):
    """ 步骤 4: 滑动窗口追踪车道线，并计算曲率 """
    yvals = []
    leftx = []
    rightx = []
    imageHeight = processed_img.shape[0]
    imageWidth = processed_img.shape[1]

    # 直方图统计底部区域的像素密度
    left_histogram = np.sum(processed_img[(imageHeight // 4):, :(imageWidth // 2)], axis=0)   
    right_histogram = np.sum(processed_img[(imageHeight // 4):, (imageWidth // 2):], axis=0)

    # 获取初始波峰位置
    starting_left_peak = np.argmax(left_histogram)
    leftx.append(starting_left_peak)
    
    # 【注意】右侧波峰必须加上局部偏移量，转为全局坐标
    starting_right_peak = np.argmax(right_histogram) + imageWidth // 2
    rightx.append(starting_right_peak)

    curH = imageHeight
    yvals.append(curH)
    increment = 25
    columnWidth = 150
    leftI = 0
    rightI = 0
    
    # 向上滑动窗口追踪车道线
    while (curH - increment >= imageHeight / 4):
        curH = curH - increment
        leftCenter = leftx[leftI]
        leftI += 1
        rightCenter = rightx[rightI]
        rightI += 1

        leftColumnL = max((leftCenter - columnWidth // 2), 0)
        rightColumnL = min((leftCenter + columnWidth // 2), imageWidth)

        leftColumnR = max((rightCenter - columnWidth // 2), 0)
        rightColumnR = min((rightCenter + columnWidth // 2), imageWidth)

        leftHistogram = np.sum(processed_img[curH - increment:curH, leftColumnL:rightColumnL], axis=0)
        rightHistogram = np.sum(processed_img[curH - increment:curH, leftColumnR:rightColumnR], axis=0)

        left_peak = np.argmax(leftHistogram)
        right_peak = np.argmax(rightHistogram)
        
        if left_peak:
            leftx.append(left_peak + leftColumnL)
        else:
            leftx.append(leftx[leftI - 1])

        if right_peak:
            rightx.append(right_peak + leftColumnR)
        else:
            rightx.append(rightx[rightI - 1])
            
        yvals.append(curH)

    yvals = np.array(yvals)
    rightx = np.array(rightx)
    leftx = np.array(leftx)
    
    # 计算现实世界空间（米）的多项式拟合
    left_fit_cr = np.polyfit(yvals * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(yvals * ym_per_pix, rightx * xm_per_pix, 2)
    
    # 计算车道底部（最靠近车辆位置）的曲率半径值
    y_eval = np.max(yvals)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    # 计算像素空间的多项式拟合
    left_fit = np.polyfit(yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2]
    
    right_fit = np.polyfit(yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2]
        
    return left_curverad, right_curverad, left_fitx, right_fitx, yvals, starting_right_peak, starting_left_peak


def drawLane(warped, M, undist, left_fitx, right_fitx, yvals):
    """ 步骤 5: 将检测到的车道区域逆透视投影回原图 """
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # 整合左右车道线边界点构成多边形
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # 填充车道内区域为绿色
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    # 逆透视变换还原回原视角
    Minv = inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
  
  
def drawCurvatureAndDistanceFromCenter(img, right_peak, left_peak, left_curverad, right_curverad):
    """ 步骤 6: 在图像上绘制曲率半径和中心偏移距离 """
    draw_img = img.copy()
    lane_offset = 50.0  # 原始标定补偿量

    # 1. 计算平均曲率半径
    avg_radius = (left_curverad + right_curverad) / 2
    radius_of_curvature = "%.2f" % avg_radius
    
    # 2. 计算车辆相对车道中心的偏移距离
    image_center = draw_img.shape[1] // 2
    lane_center = left_peak + (right_peak - left_peak) / 2  # 已修正为全局坐标计算
    offset_m = (image_center - lane_center) * xm_per_pix
    offset_cm = np.abs(offset_m * 100 + lane_offset)
    distance_from_center = "%.2f" % offset_cm

    # 3. 绘制文本到图层
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    white = (255, 255, 255)
    thickness = 3
    
    cv2.putText(draw_img, f"Dist from center: {distance_from_center} cm", (50, 100), 
                font, scale, white, thickness, cv2.LINE_AA)
    cv2.putText(draw_img, f"Radius of curvature: {radius_of_curvature} m", (50, 200), 
                font, scale, white, thickness, cv2.LINE_AA)

    return draw_img


def averageWithPrevious(stat, previousStats, nToAverage):
    """ 辅助函数: 滑动窗口均值滤波 """
    if len(previousStats) == 0:
        return stat
    if nToAverage > len(previousStats):
        nToAverage = len(previousStats)
    for i in range(len(previousStats) - nToAverage, len(previousStats)):
        stat = stat + previousStats[i]
    return stat / (nToAverage + 1)


# ==========================================
# 2. 视频帧主控制流水线 (Pipeline)
# ==========================================

def draw_lanes(img):
    global left_fitx_list, right_fitx_list
    global left_peak_list, right_peak_list
    global left_curverad_list, right_curverad_list
    
    # 备份原始图像
    original_img = copy.deepcopy(img)
    
    # P1: 图像畸变校正
    img = undistort(img)
    
    # P2: 梯度与色彩阈值化
    img = color_gradient_thresh(img)
    
    # P3: 透视变换提取鸟瞰图
    processed_img, M = perspective_transform(img)
    
    # P4: 滑动窗口像素追踪与车道参数计算
    left_curverad, right_curverad, left_fitx, right_fitx, yvals, right_peak, left_peak = getCurvatureForLanes(processed_img)
    
    # P5: 将当前帧数据推入全局缓存池
    left_fitx_list.append(left_fitx)
    right_fitx_list.append(right_fitx)
    left_peak_list.append(left_peak)
    right_peak_list.append(right_peak)
    left_curverad_list.append(left_curverad)
    right_curverad_list.append(right_curverad)
        
    # P6: 时序平滑滤波 (取前 10 帧做滑动平均)
    n = 10
    right_peak = averageWithPrevious(right_peak, right_peak_list, n)
    left_peak = averageWithPrevious(left_peak, left_peak_list, n)
    right_fitx = averageWithPrevious(right_fitx, right_fitx_list, n)
    left_fitx = averageWithPrevious(left_fitx, left_fitx_list, n)
    left_curverad = averageWithPrevious(left_curverad, left_curverad_list, n)
    right_curverad = averageWithPrevious(right_curverad, right_curverad_list, n)
    
    # P7: 渲染可视化（画车道线区域 + 写曲率和距离文本）
    img = drawLane(processed_img, M, original_img, left_fitx, right_fitx, yvals)
    img = drawCurvatureAndDistanceFromCenter(img, right_peak, left_peak, left_curverad, right_curverad)
    
    return img
 
# ==========================================
# 3. 视频输入/输出运行逻辑 | 实时看见处理画面
# ==========================================
if __name__ == "__main__":
    f = "output"
    video_path = "data/challenge_video.mp4"
    output_path = f"data/output/challenge_video_{f}.mp4"
    
    # 1. 初始化视频读取
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] 无法打开视频文件: {video_path}")
        exit()
        
    # 2. 初始化视频保存（可选：如果不想保存，可以注释掉下面三行）
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("====== 开始实时车道线检测 ======")
    print("提示: 在画面窗口上按 'q' 键可提前退出。")

    while True:
        success, frame = cap.read()
        if not success:
            print("[i] ==> 视频播放结束或处理完毕!!!")
            break

        # 3. 核心：调用你原有的车道线检测 Pipeline 函数处理当前帧 
        result_frame = draw_lanes(frame)

        # 4. 实时显示画面
        cv2.imshow("Lane Detection Real-Time", result_frame)
        
        # 5. 写入视频文件
        # video_writer.write(result_frame)

        # 6. 控制播放速度与退出机制
        # cv2.waitKey(1) 括号里的数字是每帧暂停的毫秒数。
        # 1 代表尽可能快地渲染；如果觉得太快，可以改成 int(1000 / fps)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[i] ==> 用户主动退出。")
            break

    # 7. 释放资源
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print("====== 资源已释放，运行结束 ======")


 