# Double generator definition
class XYGenerator:
    def __init__(self, x_generator, y_generator):
        self.x_generator = x_generator
        self.y_generator = y_generator

    def __iter__(self):
        return self

    def __next__(self):
        def map_row(r):
            r[int(r[-1])] = 1
            return r

        x_next = self.x_generator.next()
        y_next_rgb = self.y_generator.next()

        num_classes = 58
        n = y_next_rgb.shape[0]
        y_next = np.zeros((n, num_classes))
        y_next[np.arange(n), y_next_rgb] = 1

        # s = y_next_rgb.shape
        # y_next = np.zeros((s[0],s[1],s[2],59))
        # y_next[:,:,:,-1:] = y_next_rgb[:,:,:,:1]
        # y_next = np.apply_along_axis(map_row, 3, y_next)[:,:,:,:-1]

        return (x_next, y_next)

    def next(self):
        return self.__next__()