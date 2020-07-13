from pathlib import Path
import shutil
from pymodm.connection import connect
import arvet.database.image_manager as im_manager
from arvet.database.tests.mock_image_manager import MockImageManager


__connected = False
IMAGES_DIR = Path(__file__).parent / 'test_images'


def connect_to_test_db():
    global __connected
    if not __connected:
        connect("mongodb://localhost:27017/arvet-test-db")
        __connected = True


def setup_image_manager(mock: bool = True):
    IMAGES_DIR.mkdir(exist_ok=True)
    if mock:
        image_manager = MockImageManager()
    else:
        image_manager = im_manager.DefaultImageManager(IMAGES_DIR)
    im_manager.set_image_manager(image_manager)


def tear_down_image_manager():
    im_manager.set_image_manager(None)
    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
