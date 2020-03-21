import config
from imutils import paths
import shutil
import os
from pathlib import Path

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def prepare_dataset():
    for split in (config.TRAIN, config.TEST, config.VAL):
        # grab all images paths in the current split
        logger.info(f"Processing {split}")

        p = os.path.sep.join([config.ORIG_INPUT_DATASET, split])
        imagePaths = list(paths.list_images(p))

        for imagePath in imagePaths:
            # extract class label [0,1] from the filename
            # filenames are of the form:  0_<num>.jpg or 1_<num>.jpg
            # 0 - non-food, and 1 - food
            filename = imagePath.split(os.path.sep)[-1]
            label = config.CLASSES[int(filename.split('_')[0])]

            # construct the path to the output directory
            # creates a directory like:
            # base/dataset/training/non_food
            # base/dataset/testing/food
            dirPath = os.path.sep.join([config.BASE_PATH, split, label])

            # make the output directory
            Path(dirPath).mkdir(parents=True, exist_ok=True)

            # construct the path to the output image file and copy it
            p = os.path.sep.join([dirPath, filename])
            shutil.copy2(imagePath, p)


if __name__ == '__main__':
    logger.debug(f'Preparing dataset from source: {config.ORIG_INPUT_DATASET} to {config.BASE_PATH}')
    prepare_dataset()
    logger.debug("DONE!")
