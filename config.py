NUM_OF_CLASSES = 34
WIDTH = 128
HEIGHT = 128
TRAIN_IMAGES_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/Semantic Segmentation Datasets/CityScapes/Dataset/Train/Images"
TRAIN_ANNOTATIONS_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/Semantic Segmentation Datasets/CityScapes/Dataset/Train/Annotations"
COLOR_DEPTH = 3
GRAYSCALE = False
BATCH_SIZE = 10
EPOCHS = 20
VALIDATION_SPLIT = 0.2

Cityscapes_Colors = [(128, 64, 128), (244, 35, 231), (69, 69, 69), (128,128,128)
                # 0 = road, 1 = sidewalk, 2 = parking, 3= rail track
                ,(102, 102, 156), (190, 153, 153)
                # 4 = person, 5 = rider
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35), (50,50,50), (0,120,20), (200,0,200), (0,200,40), (50, 50, 255)
                # 6 = car, 7 = truck, 8 = bus, 9 = on rails, 10 = motor cycle, 11 = bicycle, 12 = caravan, 13 = trailer
                ,(152, 250, 152), (69, 129, 180), (219, 19, 60), (255,255, 20), (200,10,150), (10,20,200)
                # 14 = building, 15 = wall, 16 = fence, 17 = guard rail, 18 = bridge, 19 = tunnel
                ,(255, 0, 0), (0, 0, 142), (0, 0, 69), (20,20,0)
                # 20 = pole, 21 = pole group, 22 = traffic sign, 23 = traffic light
                ,(0, 60, 100), (0, 79, 100)
                # 24 = vegetation, 25 = terrain
                ,(119, 10, 32)
                # 26 = sky
                ,(0, 0, 230)
                #27 = ground, 28 = dynamic, 29 = static
                ,(20, 162, 187), (162, 250, 133), (50, 150, 210), (62, 68, 122), (52,78,78), (72, 84, 180)
                #30, 31, 32, 33, 34, 35
                ]
