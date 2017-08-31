# import the relevant libraries
import time
import pygame
import pygame.camera
import skimage
from pygame.locals import *

from glld.modules.trid_net import TridNet
from glld.datasets.clopema import ClopemaLoader
from glld.util.display import Display
from keras import backend as K
from skimage.transform import resize, rotate
from skimage.data import imread
import numpy as np

with K.get_session():
    dataset = ClopemaLoader()
    resolution = (352, 288)
    ratio = (352 / 224, 288 / 224)
    m = TridNet((224, 224, 3), dataset.n_garment_cats(), dataset.n_landmark_cats())
    data = dataset.fetch_data('val')
    plot = Display(dataset)

    # this is where one sets how long the script
    # sleeps for, between frames.sleeptime__in_seconds = 0.05
    # initialise the display window
    screen = pygame.display.set_mode(list(resolution))
    pygame.init()
    pygame.camera.init()
    # set up a camera object
    cam = pygame.camera.Camera("/dev/video0", resolution)
    # start the camera
    cam.start()

    m.load_weights(m.predict_global, 'global_training_3')
    while 1:
        # sleep between every frame
        # time.sleep(0.05)
        # fetch the camera image


        image = cam.get_image()
        np_img = pygame.surfarray.array3d(pygame.transform.rotate(image, 90))
        np_img = imread('/home/danfergo/cloC00024.png')
        # np_img = imread('/home/danfergo/cloC01287.png')
        np_img = resize(np_img, (224, 224, 3))
        # np_img = rotate(np_img, -90, resize=True)
        # print(np_img.shape)

        import matplotlib.pyplot as plt

        plt.imshow(np_img)
        plt.show()

        pred = m.predict_global(np.array([np_img]))[0]
        # plot.show_results(np.array([np_img]), pred)

        for p in pred[3]:
            center = (int(p[0] * ratio[0]), int(p[1] * ratio[1]))
            # print(p[2])
            # print(p)
            color = (255, 0, 0) if p[2] == 0 else (0, 255, 0)
            pygame.draw.circle(image, color, center, 5)

        # print(m[0].shape)
        # ground_truths = m.get_ground_truths(data)

        pygame.display.set_caption(dataset.categories[pred[0]])

        # print(type(image.map_array()))
        # blank out the screen
        screen.fill([0, 0, 0])
        # copy the camera image to the screen
        screen.blit(image, (0, 0))
        # update the screen to show the latest screen image





        pygame.display.update()


# cv2.destroyAllWindows()


# cv2.destroyAllWindows()

# from glld import feature_extractor_model, global_localizer
#
#
# def show_webcam(mirror=False):
#     cam = cv2.VideoCapture(0)
#     while True:
#         ret_val, img = cam.read()
#         if mirror:
#             img = cv2.resize(img, (0, 0), fx=0.35, fy=0.466666667)
#             print(img.shape)
#
#             img = cv2.flip(img, 1)
#             cv2.imshow('my webcam', img)
#             if cv2.waitKey(1) == 27:
#                 return img
#                 # break  # esc to quit
#
#
# cv2.destroyAllWindows()
#
#
# def main():
#
#     input_layer = keras.layers.Input(shape=(224, 224, 3))
#     feature_extractor = feature_extractor_model(input_layer)
#
#     global_localizer_ = global_localizer(feature_extractor[1])
#
#     model = keras.models.Model(
#         inputs=[input_layer],
#         outputs=global_localizer_
#     )
#
#     try:
#         model.load_weights('global_model_weights')
#         print('Loaded weights')
#     except:
#         print('Failed to load model weights')
#
#     model.compile(
#         # optimizer=Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.0),
#         optimizer='adadelta',
#         loss={
#             'cls': 'categorical_crossentropy',
#             'reg': 'mean_squared_error',
#         },
#     )
#
#     x = show_webcam(mirror=True)
#     pred_y = model.predict(np.array([x]), batch_size=1)
#
#
#
#
# if __name__ == '__main__':
#     main()
#
#
#
