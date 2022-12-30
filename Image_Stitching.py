import os.path

import cv2
import numpy as np


class ImageStitching:
    def __init__(self):
        self.ratio = 0.85
        self.min_match = 10
        self.sift = cv2.SIFT_create()
        self.smoothing_window_size = 800

    def match_and_homography(self, img1, img2, save_path):
        # 我们使用尺度不变特征变换（SIFT）来提取输入图像的局部特征
        # 返回关键点信息和特征
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        # 使用KNN算法对特征进行匹配
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        # 获得最接近的两个匹配特征1，2，如果当前特征和匹配特征1，匹配特征2的距离太接近，不能说明特征1是一个好的匹配特征
        # 但是如果匹配特征1的距离远远小于特征2，则说明匹配特征1才是当前特征的正确匹配特征
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        # 把所有正确的匹配特征对画出来
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)

        if os.path.exists(save_path) is False:
            os.mkdir(save_path)
        cv2.imwrite(save_path + '/matching.jpg', img3)

        # 如果有足够的匹配对，那么就找到单应矩阵
        if len(good_points) > self.min_match:
            # image1中正确匹配特征集合中的第i个特征点的坐标
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            # image2中正确匹配特征集合中的第i个特征点的坐标
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            # 计算变换矩阵，将右图的特征坐标变换到左图的特征坐标
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        return H

    def create_mask(self, img1, img2, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        # 高度取第一张图像
        height_panorama = height_img1
        # 宽度取两张图像的和
        width_panorama = width_img1 + width_img2
        # 一张图像有完整的smoothing_window_size过渡窗口，offset是一半的过渡窗口
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left_image':
            # 左图的完整过渡窗口被mask覆盖，1到0，一共2*offset个数字（完整的过渡窗口）
            # tile，第一个参数表示第一个维度复制几次，第二个参数表示第二个维度复制几次
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
            # 左图的其余部分
            mask[:, :barrier - offset] = 1
            # 右图全部为0
        else:
            # 左图的完整过渡窗口被mask覆盖，1到0，一共2*offset个数字（完整的过渡窗口）
            # tile，第一个参数表示第一个维度复制几次，第二个参数表示第二个维度复制几次
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
            # 左图的其余部分为0
            # 右图全部为1
            mask[:, barrier + offset:] = 1
        # rgb三通道
        return cv2.merge([mask, mask, mask])

    # 在Python中使用SIFT、homography、KNN和Ransac的简单图像拼接算法。
    # 有关详细信息和解释，欢迎阅读“image_string.pdf”。
    # 该项目旨在实现一种基于特征的自动图像拼接算法。
    # 当我们输入两个具有重叠场的图像时，我们期望获得一个宽的无缝全景。
    # 我们使用尺度不变特征变换（SIFT）来提取输入图像的局部特征，
    # K个最近邻算法来匹配这些特征
    # 随机样本一致性（Ransac），用于计算单应图矩阵，该矩阵将用于图像扭曲。
    # 最后，我们应用加权矩阵作为图像混合的掩模。
    def blending(self, img1, img2, save_path):
        # 获得变换矩阵
        H = self.match_and_homography(img1, img2, save_path=save_path)
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        # 高度取第一张图像
        height_panorama = height_img1
        # 宽度取两张图像的和
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        # 创建过渡mask
        mask1 = self.create_mask(img1, img2, version='left_image')
        # 获得左图
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        # 创建过渡mask
        mask2 = self.create_mask(img1, img2, version='right_image')
        # 将右图按照矩阵进行变换，变换之后右图的特征点坐标就会与左图完全一致
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2
        # 拼接两张图像
        result = panorama1 + panorama2

        # 找到图像的边界
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result


def binary_stitching(img_path1, img_path2, result_dir):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img_result = ImageStitching().blending(img1, img2, save_path=result_dir)

    if os.path.exists(result_dir) is False:
        os.mkdir(result_dir)
    cv2.imwrite(result_dir + '/panorama.jpg', img_result)

    return img_result


def multi_stitching(img_paths, result_dir):
    img_num = len(img_paths)
    if img_num <= 2:
        print("the number og input images must > 2")
    binary_stitching(img_paths[0], img_paths[1], result_dir)
    for i in range(2, img_num):
        binary_stitching(result_dir + '/panorama.jpg', img_paths[i], result_dir)


if __name__ == '__main__':
    binary_stitching("examples/example1/bright/1.jpg", "examples/example1/bright/2.jpg", "examples/example1/bright/results")
    binary_stitching("examples/example1/origin/1.jpg", "examples/example1/origin/2.jpg", "examples/example1/origin/results")
    binary_stitching("examples/example1/rotate/1.jpg", "examples/example1/rotate/2.jpg", "examples/example1/rotate/results")
    binary_stitching("examples/example1/scale/1.jpg", "examples/example1/scale/2.jpg", "examples/example1/scale/results")
    binary_stitching("examples/example1/scene/1.jpg", "examples/example1/scene/3.jpg", "examples/example1/scene/results")

    binary_stitching("examples/example2/bright/4.jpg", "examples/example2/bright/5.jpg", "examples/example2/bright/results")
    binary_stitching("examples/example2/origin/4.jpg", "examples/example2/origin/5.jpg", "examples/example2/origin/results")
    binary_stitching("examples/example2/rotate/4.jpg", "examples/example2/rotate/5.jpg", "examples/example2/rotate/results")
    binary_stitching("examples/example2/scale/4.jpg", "examples/example2/scale/5.jpg", "examples/example2/scale/results")
    binary_stitching("examples/example2/scene/4.jpg", "examples/example2/scene/6.jpg", "examples/example2/scene/results")

    binary_stitching("examples/example3/bright/7.jpg", "examples/example3/bright/8.jpg", "examples/example3/bright/results")
    binary_stitching("examples/example3/origin/7.jpg", "examples/example3/origin/8.jpg", "examples/example3/origin/results")
    binary_stitching("examples/example3/rotate/7.jpg", "examples/example3/rotate/8.jpg", "examples/example3/rotate/results")
    binary_stitching("examples/example3/scale/7.jpg", "examples/example3/scale/8.jpg", "examples/example3/scale/results")
    binary_stitching("examples/example3/scene/7.jpg", "examples/example3/scene/9.jpg", "examples/example3/scene/results")

    binary_stitching("examples/example4/bright/10.jpg", "examples/example4/bright/11.jpg", "examples/example4/bright/results")
    binary_stitching("examples/example4/origin/10.jpg", "examples/example4/origin/11.jpg", "examples/example4/origin/results")
    binary_stitching("examples/example4/rotate/10.jpg", "examples/example4/rotate/11.jpg", "examples/example4/rotate/results")
    binary_stitching("examples/example4/scale/10.jpg", "examples/example4/scale/11.jpg", "examples/example4/scale/results")
    binary_stitching("examples/example4/scene/10.jpg", "examples/example4/scene/12.jpg", "examples/example4/scene/results")

    binary_stitching("examples/example5/bright/13.jpg", "examples/example5/bright/14.jpg", "examples/example5/bright/results")
    binary_stitching("examples/example5/origin/13.jpg", "examples/example5/origin/14.jpg", "examples/example5/origin/results")
    binary_stitching("examples/example5/rotate/13.jpg", "examples/example5/rotate/14.jpg", "examples/example5/rotate/results")
    binary_stitching("examples/example5/scale/13.jpg", "examples/example5/scale/14.jpg", "examples/example5/scale/results")
    binary_stitching("examples/example5/scene/13.jpg", "examples/example5/scene/15.jpg", "examples/example5/scene/results")

    multi_stitching(["examples/example1/multi/1.jpg",
                     "examples/example1/multi/2.jpg",
                     "examples/example1/multi/3.jpg"],
                    "examples/example1/multi/results")
    multi_stitching(["examples/example2/multi/4.jpg",
                     "examples/example2/multi/5.jpg",
                     "examples/example2/multi/6.jpg"],
                    "examples/example2/multi/results")
    multi_stitching(["examples/example3/multi/7.jpg",
                     "examples/example3/multi/8.jpg",
                     "examples/example3/multi/9.jpg"],
                    "examples/example3/multi/results")
    multi_stitching(["examples/example4/multi/10.jpg",
                     "examples/example4/multi/11.jpg",
                     "examples/example4/multi/12.jpg"],
                    "examples/example4/multi/results")
    multi_stitching(["examples/example5/multi/13.jpg",
                     "examples/example5/multi/14.jpg",
                     "examples/example5/multi/15.jpg"],
                     "examples/example5/multi/results")
