import matplotlib.pyplot as plt
import numpy as np

class DataDrawer:
    def __init__(self, point_cnt = 1000):
        fig, self.ax = plt.subplots()
        self.points = np.random.uniform(low=-2, high=2, size=(point_cnt, 2))
        self.x_sorted_indices = np.argsort(self.points[:,0])
        self.x_sorted_values = self.points[self.x_sorted_indices,0]

        self.classes = np.zeros(self.points.shape[0], dtype=int)
        self.click_radius = 0.2
        self.ax.scatter(self.points[:,0], self.points[:,1], c=self.classes, s=3)

        self.selected_class = 0
        self.total_class_cnt = 1
        self.is_dragging = False
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid = fig.canvas.mpl_connect('button_release_event', self.onrelease)
        cid = fig.canvas.mpl_connect('motion_notify_event', self.onmove)
        cid = fig.canvas.mpl_connect('key_press_event', self.onkeypress)

    def get_data(self):
        return self.points, self.classes

    def start(self):
        plt.show()

    def onclick(self,event):
        if event.button == 1:
            self.is_dragging = True
            print('left click')

    def onrelease(self,event):
        if event.button == 1:
            self.is_dragging = False
            print('left release')

    def onmove(self,event):
        if self.is_dragging:
            pos = np.array([event.xdata, event.ydata])
            l = self.x_sorted_values.searchsorted(pos[0]-self.click_radius)
            r = self.x_sorted_values.searchsorted(pos[0]+self.click_radius)
            points_to_check = self.points[self.x_sorted_indices[l:r]]

            dist = np.linalg.norm(points_to_check - pos, axis=1)
            mask = dist <= self.click_radius
            self.classes[self.x_sorted_indices[l:r][mask]] = self.selected_class
            self.ax.cla()
            self.ax.scatter(self.points[:,0], self.points[:,1], c=self.classes, s=3)
            plt.draw()

    def onkeypress(self,event):
        self.selected_class
        self.total_class_cnt
        self.click_radius
        if event.key == 'up':
            self.selected_class = (self.selected_class + 1) % self.total_class_cnt
            print('selected class:', self.selected_class)
        elif event.key == 'down':
            self.selected_class = (self.selected_class - 1 + self.total_class_cnt) % self.total_class_cnt
            print('selected class:', self.selected_class)
        elif event.key == 'left':
            self.click_radius *= 0.9
            print('click radius:', self.click_radius)
        elif event.key == 'right':
            self.click_radius *= 1.1
            print('click radius:', self.click_radius)
        elif event.key == '+':
            self.total_class_cnt += 1
            print('Added new class')
