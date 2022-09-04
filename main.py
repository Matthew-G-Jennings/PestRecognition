import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

for i in range(2, 200):
    string_img = str(i)
    while len(string_img) <= 3:
        string_img = '0' + string_img
    string_img2 = str(i - 1)
    while len(string_img2) <= 3:
        string_img2 = '0' + string_img2
    rat_path = 'RATS/RC5/IMG_' + string_img + '.JPG'
    rat_path2 = 'RATS/RC5/IMG_' + string_img2 + '.JPG'
    img1 = cv2.imread(rat_path)

    img1 = img1[:2060, :]

    img2 = cv2.imread(rat_path2)

    img2 = img2[:2060, :]

    dif = cv2.subtract(img2, img1)
    dif_gs = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
    dif_gs_blur = cv2.GaussianBlur(dif_gs, (25, 25), 10)
    t_dif, dst = cv2.threshold(dif_gs_blur, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    R, C = np.shape(dif_gs_blur)
    row_vals = np.zeros((R,))
    col_vals = np.zeros((C,))
    for row in range(R - 50):
        row_vals[row] = np.sum(dif_gs_blur[row, :] > 20)

    for col in range(C):
        col_vals[col] = np.sum(dif_gs_blur[:, col] > 20)

    run_row_s = 0
    run_row_e = 0
    run_row_best_s = 0
    run_row_best_e = 0
    run_row_len = 0
    r = 0

    while r < len(row_vals):
        if row_vals[r] > 0:
            run_row_s = r
            while row_vals[r] > 0 and r < len(row_vals):
                r += 1
                run_row_e = r
            if run_row_e - run_row_s > run_row_len:
                run_row_best_s = run_row_s
                run_row_best_e = run_row_e
                run_row_len = run_row_e - run_row_s
        r += 1
    run_col_s = 0
    run_col_e = 0
    run_col_best_s = 0
    run_col_best_e = 0
    run_col_len = 0
    c = 0

    while c < len(col_vals):
        if col_vals[c] > 0:
            run_col_s = c
            while col_vals[c] > 0 and c < len(col_vals)-1:
                c += 1
                run_col_e = c
            if run_col_e - run_col_s > run_col_len:
                run_col_best_s = run_col_s
                run_col_best_e = run_col_e
                run_col_len = run_col_e - run_col_s
        c += 1

    start_point = run_col_best_e, run_row_best_s
    end_point = run_col_best_s, run_row_best_e

    output = cv2.rectangle(img1, start_point, end_point, (255, 0, 0), 10)

    cv2.imshow('image', dif_gs_blur)
    cv2.waitKey(0)
cv2.destroyAllWindows()
