from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from pathlib import Path

class VisualizeResults:
    def __init__(self, font_size = 20):
        """Initialise the class to show the predicted classes.

        Args:
            font_size (int): size of the font. Dfaults to 20.
        """
        self.colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
        ]

        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            self.font = ImageFont.load_default()

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