import numpy as np
from matplotlib import pyplot as pl

pl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
pl.rcParams['axes.unicode_minus'] = False
pl.rcParams['figure.figsize'] = (7, 7)


def rec(x: np.ndarray, deg: str = "d"):
    return np.array([x[0] * np.cos(x[1] if str == "r" else np.deg2rad(x[1])),
                     x[0] * np.sin(x[1] if str == "r" else np.deg2rad(x[1]))])


class Calibrator:
    def __init__(self):
        ACTUAL_POS_POL: list[tuple[float, float]] = [
            (0, 0),
            (100, 0),
            (98, 40.10),
            (112, 80.21),
            (105, 119.75),
            (98, 159.86),
            (112, 199.96),
            (105, 240.07),
            (98, 280.17),
            (112, 320.28)
        ]

        IDEAL_POS_POL: list[list[float, float]] = [[0, 0]]
        for i in np.arange(0, 9):
            IDEAL_POS_POL.append([100, i * 40])

        self.ACTUAL_POS_REC: np.ndarray = np.array(ACTUAL_POS_POL, dtype=np.float64)
        for i, ie in enumerate(self.ACTUAL_POS_REC):
            self.ACTUAL_POS_REC[i] = rec(ie)

        self.IDEAL_POS_REC: np.ndarray = np.array(IDEAL_POS_POL, dtype=np.float64)
        for i, ie in enumerate(self.IDEAL_POS_REC):
            self.IDEAL_POS_REC[i] = rec(ie)

        self.calib_count: int = 0
        self.prev_stddev = 1000
        self.strategy: int = 1

    def _vec_angle(self, v1: np.ndarray, v2: np.ndarray):
        return np.arccos((v1 * v2).sum() / np.sqrt((v1 ** 2).sum() * (v2 ** 2).sum()))

    def dis_to_ideal_stddev(self):
        vecs = self.ACTUAL_POS_REC - self.IDEAL_POS_REC
        distances: np.ndarray = np.sqrt((vecs ** 2).sum(axis=1))
        return np.std(distances)

    def _dis_to_ideal(self, index):
        vec: np.ndarray = self.ACTUAL_POS_REC[index] - self.IDEAL_POS_REC[index]
        return np.sqrt((vec * vec).sum())

    def _get_actual_scatter_color(self, index):
        dis = self._dis_to_ideal(index)
        perc = dis / 12

        # trim to [0,1]
        perc = np.minimum(perc, 1)
        perc = np.maximum(perc, 0)

        # must start from 0, end with 1 and strictly increase.
        gradients: list[list[float, float, float, float]] = [
            [0, 33, 173, 0],
            [0.1, 84, 255, 0],
            [0.5, 255, 186, 0],
            [1, 255, 54, 0]
        ]

        color = [0, 0, 0]

        for i in range(0, len(gradients) - 1):
            if gradients[i][0] <= perc <= gradients[i + 1][0]:
                range_perc = \
                    (perc - gradients[i][0]) / (gradients[i + 1][0] - gradients[i][0])
                for j in range(0, 3):
                    color[j] = gradients[i][j + 1] \
                               + range_perc * (gradients[i + 1][j + 1] - gradients[i][j + 1])

                break

        return (np.array(color) / 255).reshape((1, -1))

    # angles in rad
    def _is_my_turn(self, index: int, angles: np.ndarray) -> tuple[bool, int]:
        ANGLE_CENTERS = np.deg2rad(np.array([
            [10, 10, 20],
            [30, 30, 60],
            [50, 50, 100],
            [70, 70, 140]
        ], dtype=np.float64))

        min_dist = 1000
        min_dist_index = -1

        for i, ie in enumerate(ANGLE_CENTERS):
            dist = np.sqrt(((angles - ie) ** 2).sum())
            if dist < min_dist:
                min_dist = dist
                min_dist_index = i

        # print(index, min_dist, min_dist_index, np.abs(angles[0] - angles[1]), np.rad2deg(angles))

        # for case [70,70,140], a threshold of 0.5 is chosen
        if min_dist_index < 3:
            if min_dist > 0.3:
                return False, -1
        else:
            if min_dist > 0.43:
                return False, -1

        if index == 0 or index == 1:
            pass
        elif index == 2:
            if min_dist_index == 3:
                return True, 3
        elif index == 3:
            if min_dist_index == 2:
                return True, 5
        elif index == 4:
            if min_dist_index == 1:
                return True, 7
        elif index == 5:
            if min_dist_index == 0:
                return True, 9
        elif index == 6:
            if min_dist_index == 0:
                return True, 2
        elif index == 7:
            if min_dist_index == 1:
                return True, 4
        elif index == 8:
            if min_dist_index == 2:
                return True, 6
        elif index == 9:
            if min_dist_index == 3:
                return True, 8

        return False, -1

    # angles in rad
    def _drone_calibrate(self, index: int, angles: np.ndarray):
        is_my_turn, drone_id = self._is_my_turn(index, angles)

        if not is_my_turn:
            return

        print("    FY{:02d} identified it's its turn to calibrate.".format(index))

        # now I know FY00, FY01 and FY_drone_id transmits signals.
        # also, I assume they are exactly at their ideal location.
        if 2 <= index <= 5:
            alpha = 2 * np.pi - angles[0]
        else:
            alpha = angles[0]

        if index in [6, 7, 8, 9]:
            beta = 2 * np.pi - angles[1]
        else:
            beta = angles[1]

        theta = np.deg2rad((drone_id - 1) * 40)

        # r = 1

        y = -(2 * np.sin(beta - alpha + theta)) / (np.sin(alpha) * (np.sin(beta + theta) - np.sin(beta)) * (
                (2 * (np.sin(beta) - np.cos(beta + theta) * np.tan(alpha)) ** 2) / (
                np.tan(alpha) ** 2 * (np.sin(beta + theta) - np.sin(beta)) ** 2) + 2))
        x = (y * (np.cos(beta + theta) / np.sin(beta) - 1 / np.tan(alpha))) / (np.sin(beta + theta) / np.sin(beta) - 1)

        x *= 100
        y *= 100

        print("    FY{:02d} calculated its location to be ({:.7f},{:.7f}). "
              "However, it is actually ({:.7f},{:.7f}).".format(
            index, x, y,
            self.ACTUAL_POS_REC[index][0],
            self.ACTUAL_POS_REC[index][1]))

        print("    FY{:02d}'s ideal location is ({:.7f},{:.7f}).".format(
            index,
            self.IDEAL_POS_REC[index][0],
            self.IDEAL_POS_REC[index][1],
        ))

        adjustment: np.ndarray = np.array([self.IDEAL_POS_REC[index][0] - x,
                                           self.IDEAL_POS_REC[index][1] - y])

        print("    FY{:02d} will adjust itself by ({:.7f},{:.7f}).".format(
            index, adjustment[0], adjustment[1]))

        if index in [2, 9]:
            self.ACTUAL_POS_REC[index] += adjustment / 2
        else:
            self.ACTUAL_POS_REC[index] += adjustment

        print("    FY{:02d} has adjust itself to ({:.7f},{:.7f}).".format(
            index,
            self.ACTUAL_POS_REC[index][0],
            self.ACTUAL_POS_REC[index][1]))

    def draw_current(self, show: bool = True, save: bool = False):
        pl.clf()
        pl.xlim(-120, 120)
        pl.ylim(-120, 120)
        # Actual pos scatters
        for i, ie in enumerate(self.ACTUAL_POS_REC):
            pl.scatter(ie[0], ie[1], c=self._get_actual_scatter_color(i), s=20)
            pl.annotate("FY{:02d}".format(i), (ie[0], ie[1] - 10))
            if i > 0:
                ideal_pos_rec = self.IDEAL_POS_REC[i]
                pl.plot([ideal_pos_rec[0], ie[0]], [ideal_pos_rec[1], ie[1]], c="r", lw=1, ls=":")

        # Ideal pos scatters
        for i, ie in enumerate(self.IDEAL_POS_REC[1:]):
            pl.scatter(ie[0], ie[1], c="b", s=20)
            pl.plot([0, ie[0]], [0, ie[1]], c="g", lw=1, ls=":")

        # Ideal circle
        circle_divs = np.linspace(0, 360, 100)
        for i, ie in enumerate(circle_divs):
            if i >= 99:
                break
            rec_coord = rec(np.array([100, ie]))
            rec_coord_next = rec(np.array([100, circle_divs[i + 1]]))
            pl.plot([rec_coord[0], rec_coord_next[0]],
                    [rec_coord[1], rec_coord_next[1]], c="b", lw=0.5)

        pl.title("校准次数={:d} σ(D)={:.3f}".format(
            self.calib_count, self.dis_to_ideal_stddev()))

        if save:
            pl.savefig(f"./t13_calib_imgs/cal{self.calib_count}.png")

        if show:
            pl.show()

    # strategy:
    # when stddev before previous calibration differs current more than 000,
    # then use STRATEGY 1:
    # select the top (param)count drones closest to ideal pos with FY00, FY01 to calibrate.
    # however, if the calibration target is more accurate, do not perform the calibration.
    # once that difference drops 000,
    # use STRATEGY 2:
    # select the drones that can calibrate the most inaccurate drones.
    def calibrate(self, count: int = 3):
        print(f"CALIBRATION {self.calib_count + 1} | STRATEGY = {self.strategy}")

        distances: list[list[float, int]] = []

        for i in range(2, 10):
            distances.append([self._dis_to_ideal(i), i])

        count = np.minimum(count, len(distances))

        drones_to_use: list[int] = []

        calibrator_lut: list[int] = [-1, -1, 3, 5, 7, 9, 2, 4, 6, 8]
        calibratee_lut: list[int] = [-1, -1, 6, 2, 7, 3, 8, 4, 9, 5]

        if self.strategy == 1:
            distances.sort(key=lambda x: x[0])
            for i in distances[0:count]:
                if self._dis_to_ideal(i[1]) <= self._dis_to_ideal(calibratee_lut[i[1]]):
                    drones_to_use.append(i[1])
                else:
                    print(f"Exclude {i[1]} because its calibratee is more accurate.")
        else:
            distances.sort(key=lambda x: x[0], reverse=True)
            print(f"This calibration intends to calibrate {[i[1] for i in distances[0:count]]}.")

            drones_to_use = [calibrator_lut[i[1]] for i in distances[0:count]]

        print(f"This calibration will use drones {drones_to_use} in turn.")

        for i in drones_to_use:
            print("Signals of FY00, FY01 and FY{:02d} transmitted to all others.".format(i))

            for j in range(2, 10):
                if i == j:
                    continue

                vecx0: np.ndarray = self.ACTUAL_POS_REC[0] - self.ACTUAL_POS_REC[j]
                vecx1: np.ndarray = self.ACTUAL_POS_REC[1] - self.ACTUAL_POS_REC[j]
                vecxi: np.ndarray = self.ACTUAL_POS_REC[i] - self.ACTUAL_POS_REC[j]

                angles: np.ndarray = np.array([
                    self._vec_angle(vecx0, vecx1),
                    self._vec_angle(vecx0, vecxi),
                    self._vec_angle(vecx1, vecxi)])

                self._drone_calibrate(j, angles)

        self.calib_count += 1

        current_stddev = self.dis_to_ideal_stddev()

        if np.abs(current_stddev - self.prev_stddev) < 1e-6 and self.strategy == 1:
            self.strategy = 2
            print("------------- STRATEGY CHANGED TO 2 -------------")

        self.prev_stddev = current_stddev

        print(f"STDDEV after calibration: {current_stddev}")
        print("===================================================")


# case [70,70,140] test
# for i in range(0,10):
#     print(cal._is_my_turn(i, np.array([1.0883722957, 1.0611017691, 2.1494740648])))
# cal._drone_calibrate(9, np.array([1.101618148,1.0316672065,2.1332853545]))
# cal._drone_calibrate(9, np.array([1.101862319, 1.048083218, 2.149945537]))

cal = Calibrator()
cal.draw_current(show=False, save=True)

counts = []
stddevs = []

for i in range(0, 50):
    cal.calibrate()
    cal.draw_current(show=False, save=True)
    counts.append(i)
    stddevs.append(cal.dis_to_ideal_stddev())

pl.clf()
pl.grid()
pl.title("σ(D)-迭代次数关系图")
pl.xlabel("迭代次数")
pl.ylabel("σ(D)/m")
pl.ylim(0.5, 2.3)
pl.plot(counts, stddevs, c="b", lw=0.7, marker="*")
pl.plot([15, 15], [0, 2.3], c="r", lw=1)
pl.annotate("策略切换", (15, 2), (17, 2), arrowprops=dict(arrowstyle='->', facecolor='black'))
pl.savefig("./t13_calib_imgs/stddev.png")
pl.show()
