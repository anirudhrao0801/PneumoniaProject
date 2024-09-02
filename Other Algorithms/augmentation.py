import imgaug.augmenters as augmenters

def augment(data, augmenter):
    if len(data.shape) == 3:
        return augmenter.augment_image(data)
    if len(data.shape) == 4:
        return augmenter.augment_images(data)

def rotate(data, rotate):
    fun = augmenters.Affine(rotate=rotate)
    return augment(data, fun)

def shear(data, shear):
    fun = augmenters.Affine(shear=shear)
    return augment(data, fun)

def scale(data, scale):
    fun = augmenters.Affine(scale=scale)
    return augment(data, fun)

def flip_left_right(data):
    fun = augmenters.Fliplr()
    return augment(data, fun)

def flip_up_down(data):
    fun = augmenters.Flipud()
    return augment(data, fun)

def remove_color(data, channel):
    new_data = data.copy()
    if len(data.shape) == 3:
        new_data[:,:,channel] = 0
        return new_data
    if len(data.shape) == 4:
        new_data[:,:,:,channel] = 0
        return new_data
