import numpy as np
import openpyxl as px


def rec(r: float, theta: float) -> np.ndarray:
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def get_division_circ_coord(perc: float, r: float, x: np.ndarray) -> np.ndarray:
    angle = 2 * np.pi * perc
    return np.array([x[0] + r * np.cos(angle), x[1] + r * np.sin(angle)])


def get_vec_angle(vec1: np.ndarray, vec2: np.ndarray):
    return np.arccos((vec1 * vec2).sum() / np.sqrt((vec1 ** 2).sum() * (vec2 ** 2).sum()))


R: float = 1
MAX_BIAS: float = 0.13 * R

pos: list[np.ndarray] = []

for i in np.arange(0, 9):
    pos.append(rec(R, 40 * i * np.pi / 180))

wb=px.Workbook()
ws=wb.worksheets[0]
ws.append(["收信机编号","发信机编号","收信机至FY00连线与至发信机连线夹角范围(度数)"])
print("Angle look-up table:")
for i, ie in enumerate(pos):
    for j, je in enumerate(pos):
        if i == j:
            continue

        min_angle: float = 7
        max_angle: float = 0

        for k in np.linspace(0, 1, 100):
            coord = get_division_circ_coord(k, MAX_BIAS, pos[i])
            vec1 = np.array([0, 0]) - coord
            vec2 = je - coord

            angle = get_vec_angle(vec1, vec2)

            min_angle = np.minimum(min_angle, angle)
            max_angle = np.maximum(max_angle, angle)

        print("FY{0:02d} FY{1:02d} [{2:.5f},{3:.5f}]".format(
            i + 1, j + 1, np.rad2deg(min_angle), np.rad2deg(max_angle)))
        ws.append(["FY{:02d}".format(i+1),
                   "FY{:02d}".format(j+1),
                   "[{:.5f},{:.5f}]".format(np.rad2deg(min_angle), np.rad2deg(max_angle))])

wb.save("../data/至FY00连线夹角查找表.xlsx")

print("Positions in need of ALPHA transformation:")
for i in range(9, 18):
    # Min Known Drone is
    print("FY{:02d} ".format(i % 9 + 1), end="")
    for j in range(i - 4, i):
        print("FY{:02d},".format(j % 9 + 1), end="")
    print()

print("Positions in need of BETA transformation:")
for i in range(0, 9):
    # Max Known Drone is
    print("FY{:02d} ".format(i % 9 + 1), end="")
    for j in range(i + 1, i + 5):
        print("FY{:02d},".format(j % 9 + 1), end="")
    print()

print("Circ angles:")
# assume recv is FY01
# distance between known drones
for i in range(1, 5):
    min_circ_angle_acu = 7
    max_circ_angle_acu = 0

    min_circ_angle_obt = 7
    max_circ_angle_obt = 0
    # min known id
    for j in range(1, 9):
        kd1 = j
        kd2 = (j + i) % 9

        if kd2 + 1 == 1:
            continue

        cur_min_circ_angle_acu = 7
        cur_max_circ_angle_acu = 0

        cur_min_circ_angle_obt = 7
        cur_max_circ_angle_obt = 0

        for k in np.linspace(0, 1, 100):
            coord1 = get_division_circ_coord(k, MAX_BIAS, pos[kd1])
            coord2 = get_division_circ_coord(k, MAX_BIAS, pos[kd2])

            vec1 = coord1 - np.array([1, 0])
            vec2 = coord2 - np.array([1, 0])

            angle = get_vec_angle(vec1, vec2)

            if angle < np.pi / 2:
                cur_min_circ_angle_acu = np.minimum(cur_min_circ_angle_acu, angle)
                cur_max_circ_angle_acu = np.maximum(cur_max_circ_angle_acu, angle)
            else:
                cur_min_circ_angle_obt = np.minimum(cur_min_circ_angle_obt, angle)
                cur_max_circ_angle_obt = np.maximum(cur_max_circ_angle_obt, angle)

        min_circ_angle_acu = np.minimum(min_circ_angle_acu, cur_min_circ_angle_acu)
        max_circ_angle_acu = np.maximum(max_circ_angle_acu, cur_max_circ_angle_acu)

        min_circ_angle_obt = np.minimum(min_circ_angle_obt, cur_min_circ_angle_obt)
        max_circ_angle_obt = np.maximum(max_circ_angle_obt, cur_max_circ_angle_obt)

    print("DIST={:d} [{:5f},{:5f}] [{:5f},{:5f}]"
          .format(i,
                  np.rad2deg(min_circ_angle_acu),
                  np.rad2deg(max_circ_angle_acu),
                  np.rad2deg(min_circ_angle_obt),
                  np.rad2deg(max_circ_angle_obt)))

# two-acute LUT
wb = px.Workbook()
ws = wb.worksheets[0]
ws.append(["收信机编号", "圆周上发信机编号", "最小夹角(收信机位置准确时)", "第二小夹角(收信机位置准确时)"])
# current drone
for i in np.arange(1, 9):
    vec0: np.ndarray = np.array([0, 0]) - pos[i]
    vec1: np.ndarray = np.array([1, 0]) - pos[i]
    for j in np.arange(1, 9):
        if i == j:
            continue

        vec2: np.ndarray = pos[j] - pos[i]

        angles = [
            np.rad2deg(get_vec_angle(vec0, vec1)),
            np.rad2deg(get_vec_angle(vec1, vec2)),
            np.rad2deg(get_vec_angle(vec0, vec2))
        ]

        angles.sort()

        print("FY{:02d} | FY00 and FY{:02d}: {:.5f}, {:.5f}".format(
            i, j,
            angles[0], angles[1]
        ))

        ws.append(["FY{:02d}".format(i), "FY{:02d}".format(j), angles[0], angles[1]])

wb.save("../data/最小二夹角查找表.xlsx")
