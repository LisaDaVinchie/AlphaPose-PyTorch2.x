from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from pathlib import Path

class VisualizeResults:
    def __init__(self, thumbnails_dir: Path = None, font_size = 20):
        """Initialise the class to show the predicted classes.

        Args:
            thumbnails_dir (Path): path to the directory containing the images to use as thumbnails.
            Use None to not print thumbnails. Defaults to None.
        """
        self.colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
        ]

        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            self.font = ImageFont.load_default()

        self.class_to_img = self.get_thumbnail_images(thumbnails_dir) if thumbnails_dir is not None else None

    def get_thumbnail_images(self, thumbnails_dir: Path) -> dict:
        """Get a list of paths of thumbnail images.

        Args:
            thumbnails_dir (Path): directory where the images are stored, with names of the form "<class_name>*"

        Returns:
            dict: dictionary with class names as keys and image paths as elements. One path per class.
            None if no image is found or the directory does not exist.
        """
        if not thumbnails_dir.exists():
            print("Warning: no thumbnail dir found")
            return None

        mini_images_list = list(thumbnails_dir.glob("*.jpg")) + \
                            list(thumbnails_dir.glob("*.jpeg")) + \
                            list(thumbnails_dir.glob("*.png"))
        
        class_to_img = defaultdict(str)

        for path in mini_images_list:
            class_name = "_".join(path.stem.split("_")[0:2])
            if class_name in list(class_to_img.keys()):
                continue
            class_to_img[class_name] = path
        
        return class_to_img if class_to_img else None

    def visualize(
            self,
            result: dict,
            save_path: Path = None,
            show: bool = False,
            base_offset: tuple = (50, 100),
            img_spacing: tuple = (50, 100)
        ) -> Image:
        """Visualize the predicted result by drawing bounding boxes, labels and scores.

        Args:
            result (dict): dictionary with the results. 
            save_path (Path, optional): path to save the created image. Defaults to None.
            show (bool, optional): choose to show the image. Defaults to False.
            base_offset (tuple, optional): distance between the thumbnail and the image border, in pixels. Defaults to (50, 100).
            img_spacing (tuple, optional): distance between the shown thumbnails, if more than one is shown. Defaults to (50, 100).

        Returns:
            Image: modified image
        """
        image = result["image"].copy()
        draw  = ImageDraw.Draw(image)
        
        offset = [base_offset[0], base_offset[1]] # width, height
        for box, names, scores, yconf in zip(
            result["boxes"], result["names"],
            result["scores"], result["yolo_conf"]
        ):
            
            # Draw the bounding box with the most probable label
            self.draw_bbox_w_label(draw, box, names, scores)
            
            if self.class_to_img is not None:
                for name, score in zip(names, scores):
                    image, (new_h, new_w) = self.add_predicted_classes_thumbnails(image, draw, offset, name, score)
                    offset[1] += new_h + img_spacing[1]
                
                offset[1] = base_offset[1]
                offset[0] += base_offset[0] + img_spacing[0] + new_w

        if save_path is not None:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            save_path.parent.mkdir(exist_ok=True, parents=True)
            image.save(save_path)
            print(f"Saved to {save_path}")

        if show:
            image.show()

        return image
    
    def draw_bbox_w_label(self, draw, box: list, name: str, score: float, text_align = 'l'):

        x1, y1, x2, y2 = map(int, box)

        # Draw label rectangle
        label = f"{name} ({score:.2f})"
        # label = f"{y2 - y1}"
        bbox  = self.font.getbbox(label)
        box_w, box_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        color = self.colors[hash(name) % len(self.colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4) # Draw object bounding box

        # Draw text rectangle and text
        x1_t = x2 - box_w - 6 if text_align == 'r' else x1
        x2_t = x2 if text_align == 'r' else x1 + box_w + 6
        y1_t = y1 - box_h - 4
        y2_t = y1
        draw.rectangle([x1_t, y1_t, x2_t, y2_t], fill=color)
        draw.text((x1_t, y1_t), label,
                    fill="white", font=self.font,
                    stroke_width=2,
                    stroke_fill="black")

    def add_predicted_classes_thumbnails(self, image: Image, draw, offset: tuple, name: str, score: float):

        if name not in list(self.class_to_img.keys()):
            print(f"No image found for class {name}")
            return image, (0, 0)

        mini_image_path = self.class_to_img[name]

        mini_image = Image.open(mini_image_path)

        image, (new_h, new_w), (mini_x, mini_y) = self.draw_thumbnail(image, mini_image, offset=offset, prop=1/7, pos="tr")

        label = f"{score:.2f} {name}"

        draw.text((mini_x, mini_y - 30), label,
                                fill="white", font=self.font,
                                stroke_width=2,
                                stroke_fill="black")
                    
        return image, (new_h, new_w)

    def draw_thumbnail(self, image: Image, mini_image: Image, offset: tuple = (20, 20), prop:int = 1/10, pos = "tl") -> Image:

        w, h = image.width, image.height
        mini_w, mini_h = mini_image.width, mini_image.height

        # Resize with to match original image fraction and height to keep proportions
        new_w = int(w * prop)
        new_h = int(new_w * mini_h / mini_w)

        new_image = mini_image.resize((new_w, new_h))

        possible_positions = ["tr", "tl", "br", "bl"]
        if pos not in possible_positions:
            pos = "tr"

        # Paste mini image in the chosen corner
        mini_x = w - (new_w + offset[0]) if pos in ["tr", "br"] else offset[0]
        mini_y = h - (new_h + offset[1]) if pos in ["bl", "br"] else offset[1]
        
        image.paste(new_image, (mini_x, mini_y))

        return image, (new_h, new_w), (mini_x, mini_y)
