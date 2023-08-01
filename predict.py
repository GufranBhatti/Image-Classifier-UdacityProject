import argparse
import torch
import json
from PIL import Image

def load_model(model_checkpoint):
    model_info = torch.load(model_checkpoint)
    model = model_info['model']
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])
    return model

def process_image(image):
    im = Image.open(image)
    width, height = im.size
    picture_coords = [width, height]
    max_span = max(picture_coords)
    max_element = picture_coords.index(max_span)
    if max_element == 0:
        min_element = 1
    else:
        min_element = 0
    aspect_ratio = picture_coords[max_element] / picture_coords[min_element]
    new_picture_coords = [0, 0]
    new_picture_coords[min_element] = 256
    new_picture_coords[max_element] = int(256 * aspect_ratio)
    im = im.resize(new_picture_coords)
    width, height = new_picture_coords
    left = (width - 244) / 2
    top = (height - 244) / 2
    right = (width + 244) / 2
    bottom = (height + 244) / 2
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)
    np_image = np_image.astype('float64')
    np_image = np_image / [255, 255, 255]
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def classify_image(image_path, model_checkpoint, top_k=5, category_names=None, gpu=False):
    top_k = int(top_k)
    model = load_model(model_checkpoint)
    model.eval()

    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()

        if gpu and torch.cuda.is_available():
            image = image.cuda()
            model = model.cuda()

        outputs = model(image)
        probs, indices = torch.exp(outputs).topk(top_k)
        probs, indices = probs[0].tolist(), indices[0].tolist()

        if category_names is not None:
            with open(category_names, 'r') as f:
                cat_to_name = json.load(f)
            top_classes = [cat_to_name[str(idx)] for idx in indices]
        else:
            top_classes = [str(idx) for idx in indices]

        return list(zip(probs, top_classes))

def display_prediction(results):
    i = 0
    for p, c in results:
        i = i + 1
        p = str(round(p, 4) * 100.) + '%'
        print("{}.{} ({})".format(i, c, p))
    return None

def parse():
    parser = argparse.ArgumentParser(description='use a neural network to classify an image!')
    parser.add_argument('image_input', help='image file to classify (required)')
    parser.add_argument('model_checkpoint', help='model used for classification (required)')
    parser.add_argument('--top_k', help='how many prediction categories to show [default 5].')
    parser.add_argument('--category_names', help='file for category names')
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    args = parser.parse_args()
    return args

def main():
    args = parse()
    if args.gpu and not torch.cuda.is_available():
        raise Exception("--gpu option enabled...but no GPU detected")
    if args.top_k is None:
        top_k = 5
    else:
        top_k = args.top_k
    image_path = args.image_input
    prediction = classify_image(image_path, args.model_checkpoint, top_k, args.category_names, args.gpu)
    display_prediction(prediction)
    return prediction

if __name__ == "__main__":
    main()
