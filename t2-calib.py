import numpy as np
from matplotlib import pyplot as pl

THREE_ACCURATE = True

pl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
pl.rcParams['axes.unicode_minus'] = False
pl.rcParams['figure.figsize'] = (7, 7)


class Calibrator:
    def __init__(self):
        IDEAL_POS: list[list[float, float]] = []

        self.EDGE_L = 50

        for i in range(0, 5):
            for j in range(0, i + 1):
                x = (2 - i) * np.sqrt(3) * self.EDGE_L / 2
                y = (i - 2 * j) * self.EDGE_L / 2
                IDEAL_POS.append([x, y])

        self.IDEAL_POS: np.ndarray = np.array(IDEAL_POS, dtype=np.float64)

        disturb: np.ndarray = (np.random.random((15, 2)) - 0.5) * self.EDGE_L * 0.24
        disturb[0] = [0, 0]
        disturb[1] = [0, 0]
        if THREE_ACCURATE:
            disturb[2] = [0, 0]

        self.ACTUAL_POS: np.ndarray = IDEAL_POS + disturb

        self.calib_count: int = 0
        self.prev_stddev = 1000

    def _rotate_coord(self, x, y, index: int):
        if index not in [3, 6, 10, 5, 9, 14]:
            return x, y

        ccw: bool = index in [5, 9, 14]
        rad = np.deg2rad(60)
        if ccw:
            return x * np.cos(rad) - y * np.sin(rad), y * np.cos(rad) + x * np.sin(rad)
        else:
            return x * np.cos(rad) + y * np.sin(rad), y * np.cos(rad) - x * np.sin(rad)

    def _translate_coord(self, x, y, index: int):
        TR_LUT = [
            [0, 0],
            [0, 0],
            [0, 0],

            [np.sqrt(3) / 2, 1 / 2],  # 4
            [np.sqrt(3), 0],  # 5
            [np.sqrt(3) / 2, -1 / 2],  # 6
            [-np.sqrt(3) / 2, 3 / 2],  # 7
            [0, 1],  # 8
            [0, -1],  # 9
            [-np.sqrt(3) / 2, -3 / 2],  # 10
            [-3 * np.sqrt(3) / 2, 5 / 2],  # 11
            [-np.sqrt(3), 2],  # 12
            [-np.sqrt(3), 0],  # 13
            [-np.sqrt(3), -2],  # 14
            [-3 * np.sqrt(3) / 2, -5 / 2],  # 15
        ]

        return x + TR_LUT[index][0], y + TR_LUT[index][1]

    def _vec_angle(self, v1: np.ndarray, v2: np.ndarray):
        return np.arccos((v1 * v2).sum() / np.sqrt((v1 ** 2).sum() * (v2 ** 2).sum()))

    def dis_to_ideal_stddev(self):
        vecs = self.ACTUAL_POS - self.IDEAL_POS
        distances: np.ndarray = np.sqrt((vecs ** 2).sum(axis=1))
        return np.std(distances)

    def dis_to_near_stddev(self):
        dis = []

        dis.append(self._dis_to_another(0, 1))
        dis.append(self._dis_to_another(1, 3))
        dis.append(self._dis_to_another(3, 6))
        dis.append(self._dis_to_another(6, 10))
        dis.append(self._dis_to_another(2, 4))
        dis.append(self._dis_to_another(4, 7))
        dis.append(self._dis_to_another(7, 11))
        dis.append(self._dis_to_another(5, 8))
        dis.append(self._dis_to_another(8, 12))
        dis.append(self._dis_to_another(9, 13))
        dis.append(self._dis_to_another(10, 11))
        dis.append(self._dis_to_another(11, 12))
        dis.append(self._dis_to_another(12, 13))
        dis.append(self._dis_to_another(13, 14))
        dis.append(self._dis_to_another(6, 7))
        dis.append(self._dis_to_another(7, 8))
        dis.append(self._dis_to_another(8, 9))
        dis.append(self._dis_to_another(3, 4))
        dis.append(self._dis_to_another(4, 5))
        dis.append(self._dis_to_another(1, 2))
        dis.append(self._dis_to_another(0, 2))
        dis.append(self._dis_to_another(2, 5))
        dis.append(self._dis_to_another(5, 9))
        dis.append(self._dis_to_another(9, 14))
        dis.append(self._dis_to_another(1, 4))
        dis.append(self._dis_to_another(4, 8))
        dis.append(self._dis_to_another(8, 13))
        dis.append(self._dis_to_another(3, 7))
        dis.append(self._dis_to_another(7, 12))
        dis.append(self._dis_to_another(6, 11))

        return np.std(dis)

    def _dis_to_another(self, index1: int, index2: int):
        vec: np.ndarray = self.ACTUAL_POS[index1] - self.ACTUAL_POS[index2]
        return np.sqrt((vec * vec).sum())

    def _dis_to_ideal(self, index: int):
        vec: np.ndarray = self.ACTUAL_POS[index] - self.IDEAL_POS[index]
        return np.sqrt((vec * vec).sum())

    def _get_actual_scatter_color(self, index: int):
        dis = self._dis_to_ideal(index)
        perc = dis / (self.EDGE_L * 0.12)

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
    def _is_my_turn(self, index: int, angles: np.ndarray) -> tuple[bool, list[int, int, int]]:
        ANGLE_CENTER = np.deg2rad(np.array([30, 30, 60], dtype=np.float64))

        CALIBRATOR_LUT: list[list[int, int, int]] = [
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [1, 2, 4],
            [1, 0, 2],
            [4, 1, 2],
            [3, 4, 7],
            [3, 1, 4],
            [4, 2, 5],
            [8, 4, 5],
            [6, 7, 11],
            [6, 3, 7],
            [7, 4, 8],
            [8, 5, 9],
            [13, 8, 9]
        ]

        min_dist = np.sqrt(((angles - ANGLE_CENTER) ** 2).sum())

        # print(index, min_dist, np.abs(angles[0] - angles[1]), np.rad2deg(angles))
        # print(index, min_dist, min_dist_index, np.abs(angles[0] - angles[1]), np.rad2deg(angles))

        if min_dist < 0.3:
            return True, CALIBRATOR_LUT[index]
        else:
            return False, [-1, -1, -1]

    # angles in rad
    def _drone_calibrate(self, index: int, angles: np.ndarray):
        is_my_turn, drone_id = self._is_my_turn(index, angles)

        if not is_my_turn:
            return

        print("    FY{:02d} identified it's its turn to calibrate.".format(index + 1))

        # now I know the three drones that transmits signals.
        # also, I assume they are exactly at their ideal location.
        a = angles[0]

        b = angles[1]

        y = -(np.cos(a) ** 2 - np.cos(b) ** 2 - 3 ** (1 / 2) * np.cos(a) * (-1 / (np.cos(a) ** 2 - 1)) ** (
                1 / 2) + 3 ** (1 / 2) * np.cos(b) * (-1 / (np.cos(b) ** 2 - 1)) ** (1 / 2) + 3 ** (1 / 2) * np.cos(
            a) ** 3 * (-1 / (np.cos(a) ** 2 - 1)) ** (1 / 2) - 3 ** (1 / 2) * np.cos(b) ** 3 * (
                      -1 / (np.cos(b) ** 2 - 1)) ** (1 / 2)) / (2 * (
                np.cos(a) ** 2 * np.cos(b) ** 2 + 3 ** (1 / 2) * np.cos(a) * (-1 / (np.cos(a) ** 2 - 1)) ** (
                1 / 2) + 3 ** (1 / 2) * np.cos(b) * (-1 / (np.cos(b) ** 2 - 1)) ** (1 / 2) - 3 ** (
                        1 / 2) * np.cos(a) ** 3 * (-1 / (np.cos(a) ** 2 - 1)) ** (1 / 2) - 3 ** (
                        1 / 2) * np.cos(b) ** 3 * (-1 / (np.cos(b) ** 2 - 1)) ** (1 / 2) - np.cos(a) * np.cos(
            b) * (-1 / (np.cos(a) ** 2 - 1)) ** (1 / 2) * (-1 / (np.cos(b) ** 2 - 1)) ** (1 / 2) - 3 ** (
                        1 / 2) * np.cos(a) * np.cos(b) ** 2 * (-1 / (np.cos(a) ** 2 - 1)) ** (1 / 2) - 3 ** (
                        1 / 2) * np.cos(a) ** 2 * np.cos(b) * (-1 / (np.cos(b) ** 2 - 1)) ** (1 / 2) + np.cos(
            a) * np.cos(b) ** 3 * (-1 / (np.cos(a) ** 2 - 1)) ** (1 / 2) * (-1 / (np.cos(b) ** 2 - 1)) ** (
                        1 / 2) + np.cos(a) ** 3 * np.cos(b) * (-1 / (np.cos(a) ** 2 - 1)) ** (1 / 2) * (
                        -1 / (np.cos(b) ** 2 - 1)) ** (1 / 2) + 3 ** (1 / 2) * np.cos(a) ** 3 * np.cos(
            b) ** 2 * (-1 / (np.cos(a) ** 2 - 1)) ** (1 / 2) + 3 ** (1 / 2) * np.cos(a) ** 2 * np.cos(b) ** 3 * (
                        -1 / (np.cos(b) ** 2 - 1)) ** (1 / 2) - np.cos(a) ** 3 * np.cos(b) ** 3 * (
                        -1 / (np.cos(a) ** 2 - 1)) ** (1 / 2) * (-1 / (np.cos(b) ** 2 - 1)) ** (1 / 2) - 1))
        x = -(y * (3 ** (1 / 2) * np.cos(a) * (-1 / (np.cos(a) ** 2 - 1)) ** (1 / 2) + 3 ** (1 / 2) * np.cos(b) * (
                -1 / (np.cos(b) ** 2 - 1)) ** (1 / 2) - 2) - 3 ** (1 / 2) * np.cos(a) * (
                      -1 / (np.cos(a) ** 2 - 1)) ** (1 / 2) + 3 ** (1 / 2) * np.cos(b) * (
                      -1 / (np.cos(b) ** 2 - 1)) ** (1 / 2)) / (
                    np.cos(a) * (-1 / (np.cos(a) ** 2 - 1)) ** (1 / 2) - np.cos(b) * (
                    -1 / (np.cos(b) ** 2 - 1)) ** (1 / 2))

        x, y = self._rotate_coord(x, y, index)
        x, y = self._translate_coord(x, y, index)

        x *= self.EDGE_L/2
        y *= self.EDGE_L/2

        print("    FY{:02d} calculated its location to be ({:.7f},{:.7f}). "
              "However, it is actually ({:.7f},{:.7f}).".format(
            index + 1, x, y,
            self.ACTUAL_POS[index][0],
            self.ACTUAL_POS[index][1]))

        print("    FY{:02d}'s ideal location is ({:.7f},{:.7f}).".format(
            index + 1,
            self.IDEAL_POS[index][0],
            self.IDEAL_POS[index][1],
        ))

        adjustment: np.ndarray = np.array([self.IDEAL_POS[index][0] - x,
                                           self.IDEAL_POS[index][1] - y])

        print("    FY{:02d} will adjust itself by ({:.7f},{:.7f}).".format(
            index + 1, adjustment[0], adjustment[1]))

        if index in [2, 9]:
            self.ACTUAL_POS[index] += adjustment / 2
        else:
            self.ACTUAL_POS[index] += adjustment

        print("    FY{:02d} has adjust itself to ({:.7f},{:.7f}).".format(
            index + 1,
            self.ACTUAL_POS[index][0],
            self.ACTUAL_POS[index][1]))

    def _draw_line_between(self, index1, index2):
        pl.plot([self.ACTUAL_POS[index1][0], self.ACTUAL_POS[index2][0]],
                [self.ACTUAL_POS[index1][1], self.ACTUAL_POS[index2][1]],
                c="r", lw=1, ls=":")

    def draw_current(self, show: bool = True, save: bool = False, draw_actual_grid: bool = True):
        pl.clf()
        pl.xlim(-120, 120)
        pl.ylim(-120, 120)

        # triangle grid
        for i in range(0, 4):
            x1 = [(i - 2) * np.sqrt(3) * self.EDGE_L / 2] * 2
            y1 = [(-4 + i) * self.EDGE_L / 2, (4 - i) * self.EDGE_L / 2]

            x2 = [-self.EDGE_L * np.sqrt(3), (2 - i) * np.sqrt(3) * self.EDGE_L / 2]
            y2 = [(2 - i) * self.EDGE_L, -i * self.EDGE_L / 2]

            x3 = x2
            y3 = [(-2 + i) * self.EDGE_L, i * self.EDGE_L / 2]

            pl.plot(x1, y1, x2, y2, x3, y3, c="b", lw=1, ls=":")

        # Actual pos scatters
        for i, ie in enumerate(self.ACTUAL_POS):
            pl.scatter(ie[0], ie[1], c=self._get_actual_scatter_color(i), s=20)
            pl.annotate("FY{:02d}".format(i + 1), (ie[0], ie[1] - 10))
            # if i > 2:
            #     ideal_pos = self.IDEAL_POS[i]
            #     pl.plot([ideal_pos[0], ie[0]], [ideal_pos[1], ie[1]], c="r", lw=1, ls=":")

        # Ideal pos scatters
        for i, ie in enumerate(self.IDEAL_POS[3:]):
            pl.scatter(ie[0], ie[1], c="b", s=20)

        if draw_actual_grid:
            self._draw_line_between(0, 1)
            self._draw_line_between(1, 3)
            self._draw_line_between(3, 6)
            self._draw_line_between(6, 10)
            self._draw_line_between(2, 4)
            self._draw_line_between(4, 7)
            self._draw_line_between(7, 11)
            self._draw_line_between(5, 8)
            self._draw_line_between(8, 12)
            self._draw_line_between(9, 13)
            self._draw_line_between(10, 11)
            self._draw_line_between(11, 12)
            self._draw_line_between(12, 13)
            self._draw_line_between(13, 14)
            self._draw_line_between(6, 7)
            self._draw_line_between(7, 8)
            self._draw_line_between(8, 9)
            self._draw_line_between(3, 4)
            self._draw_line_between(4, 5)
            self._draw_line_between(1, 2)
            self._draw_line_between(0, 2)
            self._draw_line_between(2, 5)
            self._draw_line_between(5, 9)
            self._draw_line_between(9, 14)
            self._draw_line_between(1, 4)
            self._draw_line_between(4, 8)
            self._draw_line_between(8, 13)
            self._draw_line_between(3, 7)
            self._draw_line_between(7, 12)
            self._draw_line_between(6, 11)

        pl.title("校准次数={:d} σ(D)={:.3f} σ(Dn)={:.3f}".format(
            self.calib_count,
            self.dis_to_ideal_stddev(),
            self.dis_to_near_stddev()))

        if save:
            pl.savefig(f"./t2_calib_imgs{'_3accu' if THREE_ACCURATE else ''}/cal{self.calib_count}.png")

        if show:
            pl.show()

    def calibrate(self, transmitters: list[int]):
        if len(transmitters) != 3:
            raise RuntimeError("Calibration must use 3 transmitters")

        print(f"CALIBRATION {self.calib_count + 1}")

        print(f"This calibration will use drones {[i + 1 for i in transmitters]} in turn.")

        print("Signals of FY{:02d}, FY{:02d} and FY{:02d} transmitted to all others.".format(
            transmitters[0] + 1,
            transmitters[1] + 1,
            transmitters[2] + 1
        ))

        for i in range(0, 15):
            if i in transmitters:
                continue

            vecxu: np.ndarray = self.ACTUAL_POS[transmitters[0]] - self.ACTUAL_POS[i]
            vecxc: np.ndarray = self.ACTUAL_POS[transmitters[1]] - self.ACTUAL_POS[i]
            vecxd: np.ndarray = self.ACTUAL_POS[transmitters[2]] - self.ACTUAL_POS[i]

            angles: np.ndarray = np.array([
                self._vec_angle(vecxu, vecxc),
                self._vec_angle(vecxd, vecxc),
                self._vec_angle(vecxu, vecxd)])

            self._drone_calibrate(i, angles)

        current_stddev = self.dis_to_ideal_stddev()
        self.prev_stddev = current_stddev

        print(f"STDDEV after calibration: {current_stddev}")
        print("===================================================")


TRANSMITTER_PRESETS: list[list[int, int]] = [
    [1, 0, 2],
    [1, 2, 4],
    [4, 1, 2],
    [3, 1, 4],
    [4, 2, 5],
    [3, 4, 7],
    [8, 4, 5],
    [6, 3, 7],
    [7, 4, 8],
    [8, 5, 9],
    [6, 7, 11],
    [13, 8, 9]
]

cal = Calibrator()
cal.draw_current(show=False, save=True)

counts = []
ideal_stddevs = []
near_stddevs = []

if THREE_ACCURATE:
    for i, ie in enumerate(TRANSMITTER_PRESETS):
        cal.calibrate(ie)
        cal.draw_current(show=False, save=True)
        counts.append(i)
        ideal_stddevs.append(cal.dis_to_ideal_stddev())
        cal.calib_count += 1

    pl.clf()
    pl.grid()
    pl.title("σ(D)-迭代次数关系图")
    pl.xlabel("迭代次数")
    pl.plot(counts, ideal_stddevs, c="b", lw=0.7, marker="*")
    pl.savefig("./t2_calib_imgs_3accu/stddev.png")
    pl.show()
else:
    for i in range(0, 50):
        for j, je in enumerate(TRANSMITTER_PRESETS):
            cal.calibrate(je)
        counts.append(i)
        ideal_stddevs.append(cal.dis_to_ideal_stddev())
        near_stddevs.append(cal.dis_to_near_stddev())
        cal.calib_count += 1

        cal.draw_current(show=False, save=True)

    pl.clf()
    pl.grid()
    pl.title("σ(D),σ(Dn)-迭代次数关系图")
    pl.xlabel("迭代次数")

    pl.plot(counts, ideal_stddevs, c="b", lw=0.7, marker="*", label="σ(D)")
    pl.plot(counts, near_stddevs, c="m", lw=0.7, marker="*", label="σ(Dn)")

    pl.legend(ncol=2)

    pl.savefig("./t2_calib_imgs/stddev.png")
    pl.show()

