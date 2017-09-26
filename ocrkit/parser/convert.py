


def page_to_unit(pagewise):
    images, truths = [], []
    for im, tr in pagewise:
        images.extend(im)
        truths.extend(tr)
    #units = tuple(zip(images, truths)) 
    return (images,truths) #(list of images, list of truths)
