import cv2


def calculate_edge_strength(object_image):
    if len(object_image.shape) > 2:
        # TODO: Use sobel edge detectors for each channel, then build the matrix:
        # s = [[Rx^2 + Gx^2 + Bx^2, RxRy + GxGy +BxBy], [RxRy + GxGy +BxBy, Ry^2 + Gy^2 + By^2]]
        # The edge strength is the largest eigenvector,
        # if T is a + d, and D is ad - bc
        # L = T / 2 + sqrt(T^2/4 - D)
        # There's probably an analytical solution, but it's nasty
        # T = Rx^2 + Gx^2 + Bx^2 + Ry^2 + Gy^2 + By^2
        # D = Rx^2 Gy^2 + Rx^2 By^2 + Gx^2 Ry^2 + Gx^2 By^2 + Bx^2 Ry^2 + Bx^2 Gy^2 - 2 RxGxRyGy - 2 RxBxRyBy - 2 GxBxGyBy
        pass
    else:
        # 1-D image, normal sobel applies
        pass
    return 0


def calculate_sihlouette_strength(edge_strength, mask, edge_width=1):
    if edge_width < 1:
        edge_width = 1
    # TODO: Dilate and contract edges to produce a mask containing only the edge pixels
    edge_mask = cv2.dilate(mask, [], iterations=edge_width)
    return 0
