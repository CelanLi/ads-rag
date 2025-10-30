import os
from pathlib import Path

from rag.config import DEFAULT_VISION_MODEL
from rag.vlm.vision_models import get_vision_model
from utils.file_utils import export_json


SUMO_MAP_FILE_NAME = "*.net.xml"  # the map file
SUMO_ROUTE_FILE_NAME = "*.rou.xml"  # the trip file
MAP_IMG_NAME = "map.png"  # the screenshot of the map


class SumoMapProcessor:
    def __init__(self):
        self.vlm = get_vision_model(model_name=DEFAULT_VISION_MODEL)

    def process_sumo_maps(
        self,
        input_dir: str = "data/raw/sumo-scenario/",
        output_dir: str = "data/processed/sumo-scenario/",
    ):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for scenario_dir in input_path.iterdir():
            if scenario_dir.is_dir():
                map_img_file = scenario_dir / MAP_IMG_NAME

                # find the first .net.xml file that matches
                matches = list[Path](scenario_dir.glob(SUMO_MAP_FILE_NAME))
                if matches:
                    map_xml_file = matches[0]
                    print("Found map:", map_xml_file)
                else:
                    print("No map XML found in:", scenario_dir)

                if map_img_file.exists() and map_xml_file.exists():
                    print(f"Processing {map_img_file} and {map_xml_file}...")
                    scenario_output_dir = os.path.join(output_dir, scenario_dir.name)
                    print(f"Output directory: {scenario_output_dir}")
                    self.process_sumo_map(
                        map_img_file=map_img_file,
                        map_xml_file=map_xml_file,
                        output_dir=scenario_output_dir,
                    )
                    print(f"Processed {map_img_file}")

    def process_sumo_map(self, map_img_file: Path, map_xml_file: Path, output_dir: str):
        description = self.vlm.generate_sumo_map_description(
            img_path=map_img_file,
            xml_path=map_xml_file,
        )
        export_json(
            output_dir=output_dir,
            file_name=map_img_file.name,
            content={"filename": map_img_file.name, "text": description},
        )


# usage:
#     python -m rag.data_preprocessing.process_sumo_map
if __name__ == "__main__":
    sumo_map_processor = SumoMapProcessor()
    sumo_map_processor.process_sumo_maps()
