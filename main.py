from PIL import Image, ImageDraw, ImageFont
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import argparse

MODEL = "facebook/detr-resnet-50"
REVISION = "no_timm"
DEFAULT_IMAGE_PATH = "/home/maganab/Tec8/Ai/test/dog.jpeg"


def get_image(path: str):
    return Image.open(rf"{path}")


def process_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    return model(**inputs)


def convert_model_output_to_coco(model_output,
                                 processor,
                                 image_size,
                                 detection_threshold=0.9):
    target_sizes = torch.tensor([image_size[::-1]])
    return processor.post_process_object_detection(model_output,
                                                   target_sizes=target_sizes,
                                                   threshold=0.9)[0]


def get_detected_objects(results, model):
    processed_result = []
    for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]):
        detected_object = {}
        detected_object["box"] = [round(i, 2) for i in box.tolist()]
        detected_object["label"] = model.config.id2label[label.item()]
        detected_object["confidence"] = round(score.item(), 3)
        processed_result.append(detected_object)
    return processed_result


def draw_boxes_on_detected_objects(image, detected_objects):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(100)

    for detected_object in detected_objects:
        label = detected_object["label"]
        box = detected_object["box"]
        draw.rectangle(box, outline="black", width=5)
        draw.text((box[0], box[1]), label, font=font)
    return draw


def print_output_to_console(detected_objects):
    for detected_object in detected_objects:
        label = detected_object["label"]
        box = detected_object["box"]
        confidence = detected_object["confidence"]
        print(
                f"Detected {label} with confidence "
                f"{confidence} at location {box}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-image", "--image", help="Path to image", type=str)
    parser.add_argument("-v",
                        "--verbose",
                        help="Verbose option",
                        nargs="?",
                        type=int,
                        const=1)

    args = parser.parse_args()

    image_path = args.image if args.image else DEFAULT_IMAGE_PATH
    verbose = args.verbose if args.verbose else 0

    image = get_image(image_path)

    processor = DetrImageProcessor.from_pretrained(MODEL, revision=REVISION)
    model = DetrForObjectDetection.from_pretrained(MODEL, revision=REVISION)

    model_output = process_image(image, processor, model)
    results = convert_model_output_to_coco(model_output, processor, image.size)

    detected_objects = get_detected_objects(results, model)

    draw_boxes_on_detected_objects(image, detected_objects)

    if verbose:
        print_output_to_console(detected_objects)

    image.show()


if __name__ == "__main__":
    main()
