import cv2
import numpy as np
from random import randint
import pickle


def see_image_annotations(image_folder, image_name, annotation_file_name):
    images = get_image_annotations(image_folder, image_name, annotation_file_name)
    for img, top_left, bottom_right in images:
        print(img.shape)
        print(top_left)
        print(bottom_right)
        cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_training_data(image_folder, image_name, annotation_file_name):
    images = get_image_annotations(image_folder, image_name, annotation_file_name)
    concat = []
    for i in range(len(images) - 1):
        concat.append((images[i], images[i + 1]))
    return concat

def show_training_data(data):
    length = len(data)
    while True:
        i = randint(0, length - 1)
        pair = data[i]
        first = pair[0]
        second = pair[1]
        print("first")
        img, top_left, bottom_right = first
        cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("second")
        img, top_left, bottom_right = second
        cv2.rectangle(img, top_left, bottom_right, (255,0,0), 2)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_image_annotations(image_folder, image_name, annotation_file_name):
    """
    Returns list of tuples. Each tuple contains ground truth image and two corners of image
    """
    annotation_full_path = annotation_path + image_folder + annotation_file_name
    with open(annotation_full_path) as f:
        content = f.readlines()
    image_full_path = image_path + image_folder + image_name
    all_annotations = []
    for i in range(len(content)):
        coordinates = content[i].split()
        frame = coordinates[0].zfill(8)
        coordinates = [int(float(x)) for x in coordinates]
        top_left = min(coordinates[1], coordinates[3], coordinates[5], coordinates[7]), \
                    min(coordinates[2], coordinates[4], coordinates[6], coordinates[8])
        bottom_right = max(coordinates[1], coordinates[3], coordinates[5], coordinates[7]), \
                    max(coordinates[2], coordinates[4], coordinates[6], coordinates[8])
        img = cv2.imread(image_full_path + frame + ".jpg", flags=cv2.IMREAD_COLOR)
        all_annotations.append((img, top_left, bottom_right))
    return all_annotations
        
def get_directories():
    """
    Returns list of lists.
    The outside list is the length of the number of folders
    Inner list contains videos within that category
    """
    print("Getting directories..")
    import os
    directories = []
    for image_folder in os.listdir(image_path):
        image_folder_type = image_path + image_folder
        image_folder = image_folder + "/"
        inner_list = []
        for image_name in os.listdir(image_folder_type):
            annotation_file_name = image_name + ".ann"
            # print(image_folder, image_name + "/", annotation_file_name)
            inner_list.append((image_folder, image_name + "/", annotation_file_name))
        directories.append(inner_list)
    return directories

def split_data(directories, training_frac = 0.5, min_samples = 5):
    """
    Returns two lists of lists. 
    The first is the training data and the second is the validation data.
    """
    print("Splitting data into training and validation..")
    training = []
    validation = []
    for image_folder in directories:
        num_files = len(image_folder)
        num_samples = int(num_files * training_frac)
        if num_samples < min_samples:
            num_samples = min(num_files, min_samples)
        training_indices = np.random.choice(num_files, num_samples, replace = False)
        train_list = []
        val_list = []

        # construct training set and validation set
        for i in range(num_files):
            if i in training_indices:
                train_list.append(image_folder[i])
            else:
                val_list.append(image_folder[i])
        training.append(train_list)
        validation.append(val_list)
    return training, validation


prefix = "sandbox/michael/"
image_path = prefix + "tracking_data/imagedata++/"
annotation_path = prefix + "tracking_data/alov300++_rectangleAnnotation_full/"
if __name__ == "__main__":
    prefix = ""
    image_path = prefix + "tracking_data/imagedata++/"
    annotation_path = prefix + "tracking_data/alov300++_rectangleAnnotation_full/"
    directories = get_directories()
    train, val = split_data(directories, training_frac = 0.2, min_samples = 2)
    data = []
    for folder in train:
        print(folder)
        for image_folder, image_name, annotation_file_name in folder:
            data.append(get_image_annotations(image_folder, image_name, annotation_file_name))



# Sample inputs to function
# image_folder = "01-Light/"
# image_name = "01-Light_video00003/"
# annotation_file_name = "01-Light_video00003.ann"
# get_training_data(image_folder, image_name, annotation_file_name)


        # data.append(get_image_annotations(image_folder, image_name + "/", annotation_file_name))
        # data.extend(get_training_data(image_folder, image_name + "/", annotation_file_name))


# pickle.dump(directories, open("directory_structure.p", "wb"))
# pickle.load(open( "directory_structure.p", "rb"))

# pickle.dump(data, open("light.pkl", "wb"))
# pickle.load(open( "directory_structure.p", "rb"))

# data = []
# directory_structure = pickle.load(open("directory_structure.p", "rb"))
# for image_path, image_name, image_folder in directory_structure:
#     if image_folder == "01-Light/":
#         print(image_path, image_name, image_folder)
#         print(len(data))
#         data.extend(get_training_data(image_folder, image_name + "/", annotation_file_name))
# pickle.dump(data, open("light.pkl", "wb"))

# for root, dirs, files in os.walk(image_path):
#     for file in files:
#         if file.endswith(".txt"):
#              print(os.path.join(root, file))

# see_image_annotations(image_folder, image_name, annotation_file_name)

# numpy uses height width, cv2 uses width height
# x = get_training_data(image_folder, image_name, annotation_file_name)
# img, tl, br = x[0][0]
# new_img, new_tl, new_br = x[0][0]

# box_x = br[1] - tl[1]
# box_y = br[0] - tl[0]
# center_x = (br[1] + tl[1]) / 2 
# center_y = (br[0] + tl[0]) / 2

# crop = img[tl[1]:br[1], tl[0]:br[0], :]
# alternate_crop = img[center_x - box_x / 2:center_x + box_x / 2, center_y - box_y / 2:center_y + box_y / 2, :]
# # big_crop = img[center_x - box_x:center_x + box_x, center_y - box_y:center_y + box_y, :]
# big_crop = img[0:900, 0:1000, :]
# print(img.shape)
# cv2.imshow('crop', crop)
# cv2.imshow('alternate_crop', alternate_crop)
# cv2.imshow('big_crop', big_crop)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

