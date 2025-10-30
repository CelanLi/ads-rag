import os
from pathlib import Path

from rag.data_preprocessing.process_img import ImgProcessor


SUMO_MAP_FILE_NAME = "*.net.xml" # the map file
SUMO_ROUTE_FILE_NAME = "*.rou.xml" # the trip file
MAP_IMG_NAME = "map.png" # the screenshot of the map

class SumoMapProcessor:
    def __init__(self):
        self.img_processor = ImgProcessor()

    def process_sumo_maps(self, input_dir: str = "data/raw/sumo-scenario/", output_dir: str = "data/processed/sumo-scenario/"):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for scenario_dir in input_path.iterdir():
            if scenario_dir.is_dir():
                map_img_file = scenario_dir / MAP_IMG_NAME
                if map_img_file.exists():
                    print(f"Processing {map_img_file}...")
                    scenario_output_dir = os.path.join(output_dir, scenario_dir.name)
                    print(f"Output directory: {scenario_output_dir}")
                    self.process_sumo_map(map_img_file=map_img_file, output_dir=scenario_output_dir)
                    print(f"Processed {map_img_file}")

    def process_sumo_map(self, map_img_file: Path, output_dir: str):
        self.img_processor.process_img(img_file=map_img_file, type="vlm_description", output_dir=output_dir)

# usage:
#     python -m rag.data_preprocessing.process_sumo_map
if __name__ == "__main__":
    sumo_map_processor = SumoMapProcessor()
    sumo_map_processor.process_sumo_maps()