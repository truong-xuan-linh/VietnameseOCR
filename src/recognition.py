from typing import Any
from paddleocr import PaddleOCR

from src.swin_text_spotter import VisualizationDemo, setup_cfg

class BaseRecognition:
    def __init__(self):
        pass
    
    @classmethod
    def find_box(self, image: Any):
        pass

class PaddleOCRRecognition(BaseRecognition):
    def __init__(self) -> None:
        self.paddle_ocr = PaddleOCR(lang='en', use_angle_cls=False)
                
    def find_box(self, image):
        '''Xác định box dựa vào mô hình paddle_ocr'''
        result = self.paddle_ocr.ocr(image, cls = False)
        result = result[0]
        # Extracting detected components
        boxes = [res[0] for res in result] 
        texts = [{"text": res[1][0], "score": res[1][1]} for res in result]
        
        # scores = [res[1][1] for res in result]
        return boxes, texts

class SwinTextSpotterRecognition(BaseRecognition):
    def __init__(self) -> None:
        cfg = setup_cfg()
        self.demo = VisualizationDemo(cfg)
                
    def find_box(self, image):
        '''Xác định box dựa vào mô hình paddle_ocr'''
        predictions, vis_output = self.demo.run_on_image(image, 0.4, "path")
        # Extracting detected components
        boxes = []
        for i in range(len(predictions["instances"])):
            box = predictions["instances"][i].pred_boxes.tensor.numpy().tolist()[0]
            boxes.append(
                [[box[0], box[1]], [box[0], box[3]], [box[2], box[3]], [box[2], box[1]]]
            )
        texts = [{"text": "", "score": predictions["instances"][i].scores.numpy().tolist()[0]} for i in range(len(predictions["instances"]))]
        
        # scores = [res[1][1] for res in result]
        return boxes, texts
