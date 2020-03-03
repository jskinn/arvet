import os
from pathlib import Path
import numpy as np
from arvet.util.test_helpers import ExtendedTestCase
import arvet.util.image_utils as image_utils


_test_dir = 'temp-test_image_utils'

# An RGB image in array and PNG form, to test image reading
_demo_image_rgb = np.array([
    [[255 * r, 255 * g, 127.5 * (2 - r - g)] for r in np.arange(0, 1, 0.1)]
    for g in np.arange(0, 1, 0.1)
], dtype=np.uint8)
_demo_image_rgb_binary = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\n\x00\x00\x00\n\x08\x02\x00\x00\x00\x02PX" \
                         b"\xea\x00\x00\x00\x94IDAT\x18\x19\x05\xc1QA\x02P\x00\x04\xc1}1\xeejP\xc3\x1c\xd4\xb0\x80" \
                         b"\x05\x08`\x01\nP\x80\x00\x14\xa0\x00{\xff:s\xe0/X\x0c\x96\x05\x8b\xc1\xb2\xe0!\x9f`1X\x16" \
                         b",\x06\xcb\x82\x87\xcb;X\x16,\x06\xcb\x82\xc5\xe0\xe1\xeb\x15,\x0b\x16\xc3\x8a\xc1b\xd8\xe1" \
                         b"\xfa\x0c\x16\x83e\xc1b\xb0\x18v\xf8~\x04\x8ba\xc5`1X\x16<\xdc\xee\xc1\xb2`1X\x0c+\x06\x0f" \
                         b"\xf7\xdf`Y\xb0\x18,\x0b\x16\xc3\x0e\xcf[\xb0\x18,\x0b\x16\x83\xc5\xb0\xc3\xfb'X\x0c\x96" \
                         b"\x05\x8b\xc1\xb2\xe0?\xdf\xe5j\xaf\xb6\x15\xb3\xd2\x00\x00\x00\x00IEND\xaeB`\x82"

# An depth image in array and PNG form with 16-bit depth, to test image reading
_demo_image_depth = np.array([
    [65535 * (1 - 0.5 * x - 0.5 * y) for x in np.arange(0, 1, 0.1)]
    for y in np.arange(0, 1, 0.1)
], dtype=np.uint16)
_demo_image_depth_binary = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\n\x00\x00\x00\n\x10\x00\x00\x00\x00\xf8' \
                           b'\xc9L"\x00\x00\x00kIDAT\x08\x1d\x05\xc1QU\x03Q\x10@\xb1\xf3,\xf5\x8e\xa5\xbaA\x005\x80' \
                           b'\x81\n\xa8\x815\x80\x01\x0cd\xfe\x97\xe4\xdc\xf7&\x19m2:\x1e\x92\xcc&\x19\x9d\xbf\xa7' \
                           b'\x8c6\x19i;\xbf\xdf2\x9b\x8c\xb4\x99s]\xdad\xa4M\xe6|\xeeMF\xdadt\xde\x0f\x19i\x93\x91' \
                           b'\xce\xcfSF\x9b\x8c\xb4s^/i3\x926s\xbe.m2\xd2f\xf4\x0f\xa6Qu>\x91\xc9#\xb1' \
                           b'\x00\x00\x00\x00IEND\xaeB`\x82'


class TestImageUtils(ExtendedTestCase):
    rgb_image_path = None
    depth_image_path = None

    @classmethod
    def setUpClass(cls):
        # Create a test dir with the desired image files in it
        if not os.path.isdir(_test_dir):
            os.makedirs(_test_dir, exist_ok=True)
        cls.rgb_image_path = os.path.join(_test_dir, 'demo_image_rgb.png')
        with open(cls.rgb_image_path, "wb") as image_file:
            image_file.write(_demo_image_rgb_binary)

        cls.depth_image_path = os.path.join(_test_dir, 'demo_image_depth.png')
        with open(cls.depth_image_path, "wb") as image_file:
            image_file.write(_demo_image_depth_binary)

    @classmethod
    def tearDownClass(cls):
        # Clean up the image files and test dir when we're done
        if cls.rgb_image_path is not None:
            os.remove(cls.rgb_image_path)
        if cls.depth_image_path is not None:
            os.remove(cls.depth_image_path)
        os.removedirs(_test_dir)

    def test_read_colour(self):
        image_data = image_utils.read_colour(self.rgb_image_path)
        self.assertNPEqual(_demo_image_rgb, image_data)

    def test_read_colour_path(self):
        image_data = image_utils.read_colour(Path(self.rgb_image_path))
        self.assertNPEqual(_demo_image_rgb, image_data)

    def test_read_depth(self):
        image_data = image_utils.read_depth(self.depth_image_path)
        self.assertNPEqual(_demo_image_depth, image_data)

    def test_read_depth_path(self):
        image_data = image_utils.read_depth(Path(self.depth_image_path))
        self.assertNPEqual(_demo_image_depth, image_data)

    def test_convert_to_grey_do_nothing(self):
        image = np.array([[16 * y + x for x in range(0, 16)] for y in range(0, 16)], dtype=np.uint8)
        result = image_utils.convert_to_grey(image)
        self.assertNPEqual(image, result)

    def test_convert_to_grey_monochrome(self):
        monochrome = np.array([[[16 * y + x, 16 * y + x, 16 * y + x]
                                for x in range(0, 16)]
                               for y in range(0, 16)], dtype=np.uint8)
        grey_image = image_utils.convert_to_grey(monochrome)
        self.assertNPEqual(monochrome[:, :, 0], grey_image)

    def test_convert_to_grey(self):
        rgb_image = np.array([
            [[255 * r, 255 * g, 127.5 * (2 - r - g)] for r in np.arange(0, 1, 0.01)]
            for g in np.arange(0, 1, 0.01)
        ], dtype=np.uint8)
        result = image_utils.convert_to_grey(rgb_image)
        self.assertEqual(2, len(result.shape))
        self.assertEqual(rgb_image.shape[0:2], result.shape)
        self.assertNotNPEqual(rgb_image[:, :, 0], result)
        self.assertNotNPEqual(rgb_image[:, :, 1], result)
        self.assertNotNPEqual(rgb_image[:, :, 2], result)

    def test_get_bounding_box_logical(self):
        logical_image = np.array([[False for _ in range(100)] for _ in range(100)])
        left = 12
        top = 32
        right = 73
        bottom = 56
        logical_image[top:bottom, left:right] = True
        self.assertEqual((left, top, right, bottom), image_utils.get_bounding_box(logical_image))

    def test_get_bounding_box_integer(self):
        logical_image = np.array([[0 for _ in range(100)] for _ in range(100)], dtype=np.uint8)
        left = 12
        top = 32
        right = 73
        bottom = 56
        logical_image[top:bottom, left:right] = 255
        self.assertEqual((left, top, right, bottom), image_utils.get_bounding_box(logical_image))

    def test_get_bounding_box_float(self):
        logical_image = np.array([[0.0 for _ in range(100)] for _ in range(100)], dtype=np.float32)
        left = 12
        top = 32
        right = 73
        bottom = 56
        logical_image[top:bottom, left:right] = 1.0
        self.assertEqual((left, top, right, bottom), image_utils.get_bounding_box(logical_image))

    def test_resize_up_nearest(self):
        image = np.array([[1, 64], [128, 255]], np.uint8)
        result = image_utils.resize(image, (4, 4), interpolation=image_utils.Interpolation.NEAREST)
        self.assertNPEqual([
            [1, 1, 64, 64],
            [1, 1, 64, 64],
            [128, 128, 255, 255],
            [128, 128, 255, 255]
        ], result)

    def test_resize_down_nearest(self):
        image = np.array([
            [1, 1, 64, 64],
            [1, 1, 64, 64],
            [128, 128, 255, 255],
            [128, 128, 255, 255]
        ], np.uint8)
        result = image_utils.resize(image, (2, 2), interpolation=image_utils.Interpolation.NEAREST)
        self.assertNPEqual([[1, 64], [128, 255]], result)

    def test_resize_box_up(self):
        image = np.array([[1, 64], [128, 255]], np.uint8)
        result = image_utils.resize(image, (4, 4), interpolation=image_utils.Interpolation.BOX)
        self.assertNPEqual([
            [1, 1, 64, 64],
            [1, 1, 64, 64],
            [128, 128, 255, 255],
            [128, 128, 255, 255]
        ], result)

    def test_resize_box_down(self):
        image = np.array([list(range(i, i+10)) for i in range(0, 100, 10)], np.uint8)
        result = image_utils.resize(image, (2, 2), interpolation=image_utils.Interpolation.BOX)
        self.assertNPEqual([
            [22, 27],
            [72, 77]
        ], result)

    def test_resize_binlear_up(self):
        image = np.array([[1, 63], [127, 255]], np.uint8)
        result = image_utils.resize(image, (3, 3), interpolation=image_utils.Interpolation.BILINEAR)
        self.assertNPEqual([
            [1, 32, 63],
            [64, 112, 159],
            [127, 191, 255]
        ], result)

    def test_resize_binlear_down(self):
        image = np.array([
            [10, 15, 20, 25],
            [30, 35, 40, 45],
            [50, 55, 60, 65],
            [70, 75, 80, 85]
        ], np.uint8)

        result = image_utils.resize(image, (3, 3), interpolation=image_utils.Interpolation.BILINEAR)
        self.assertNPEqual([
            [17, 24, 30],
            [41, 48, 54],
            [65, 72, 78]
        ], result)

    def test_resize_preserves_dtype(self):
        for dtype in [np.float, np.float16, np.float32, np.int32]:
            image = np.asarray(np.random.uniform(0, 10, (32, 32)), dtype=dtype)
            result = image_utils.resize(image, (24, 24), interpolation=image_utils.Interpolation.BILINEAR)
            self.assertEqual(dtype, result.dtype)

    def test_resize_preserves_precision_up_to_float_32(self):
        # Check we maintain float16 precision
        image = np.array([
            [256.5, 1/256],
            [257.5, 255 / 512]
        ], dtype=np.float16)
        result = image_utils.resize(image, (1, 1), interpolation=image_utils.Interpolation.BILINEAR)
        self.assertEqual(np.mean([256.5, 1/256, 257.5, 255 / 512], dtype=np.float16), result[0, 0])

        image = np.array([
            [(256.5 * 256.5), 1 / (256 * 256)],
            [(257.5 * 257.5), 255 / (512 * 512)]
        ], dtype=np.float32)
        result = image_utils.resize(image, (1, 1), interpolation=image_utils.Interpolation.BILINEAR)
        self.assertEqual(np.mean([(256.5 * 256.5), 1 / (256 * 256),
                                  (257.5 * 257.5), 255 / (512 * 512)], dtype=np.float32), result[0, 0])

    def test_show_image_float(self):
        image_data = np.array([[0.0 for _ in range(100)] for _ in range(100)], dtype=np.float32)
        image_data[12:56, 34:75] = 1.0
        image_utils.show_image(image_data, 'test window')

    def test_to_uint_image_handles_grey_floats(self):
        for dtype in [
            np.float,
            np.float16,
            np.float32,
            np.float64,
            np.double
        ]:
            image_data = np.array([[1 / (0.001 * x * y + 1) for y in range(100)] for x in range(100)], dtype=dtype)
            result = image_utils.to_uint_image(image_data)
            self.assertEqual(np.uint8, result.dtype)
            self.assertEqual((100, 100), result.shape)
            self.assertEqual(255, np.max(result))

    def test_to_uint_image_handles_color_floats(self):
        for dtype in [
            np.float,
            np.float16,
            np.float32,
            np.float64,
            np.double
        ]:
            image_data = np.array([[[1 / (0.001 * x * y + 1),
                                     1 / (0.001 * (100 - x) * y + 1),
                                     1 / (0.001 * x * (100 - y) + 1)]
                                    for y in range(100)] for x in range(100)], dtype=dtype)
            result = image_utils.to_uint_image(image_data)
            self.assertEqual(np.uint8, result.dtype)
            self.assertEqual((100, 100, 3), result.shape)
            self.assertEqual(255, np.max(result))

    def test_to_uint_image_handles_bools(self):
        image_data = np.array([[1 / (0.001 * x * y + 1) for y in range(100)] for x in range(100)])
        image_data = image_data > 0.5
        result = image_utils.to_uint_image(image_data)
        self.assertEqual(np.uint8, result.dtype)
        self.assertEqual((100, 100), result.shape)
        self.assertEqual(255, np.max(result))
        self.assertEqual(0, np.min(result))

    def test_to_uint_image_handles_grey_integer(self):
        for dtype in [
            np.int,
            np.uint,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.long,
            np.ulonglong
        ]:
            image_data = np.array([[255 / (0.001 * x * y + 1) for y in range(100)] for x in range(100)], dtype=dtype)
            result = image_utils.to_uint_image(image_data)
            self.assertEqual(np.uint8, result.dtype)
            self.assertEqual((100, 100), result.shape)
            self.assertEqual(255, np.max(result))

    def test_to_uint_image_handles_color_integer(self):
        for dtype in [
            np.int,
            np.uint,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.long,
            np.ulonglong
        ]:
            image_data = np.array([[[255 / (0.001 * x * y + 1),
                                     255 / (0.001 * (100 - x) * y + 1),
                                     255 / (0.001 * x * (100 - y) + 1)]
                                    for y in range(100)] for x in range(100)], dtype=dtype)
            result = image_utils.to_uint_image(image_data)
            self.assertEqual(np.uint8, result.dtype)
            self.assertEqual((100, 100, 3), result.shape)
            self.assertEqual(255, np.max(result))
