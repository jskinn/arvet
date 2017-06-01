"""
Fix the metadata for my first iteration of data generation
I'm missing some keys I now realize I need, both at the dataset level and the
"""
import math
import packaging.version as vs


def update_dataset_metadata(metadata):
    if 'Version' in metadata:
        version = vs.parse(metadata['Version'])
    else:
        version = vs.parse('0.0.0')

    if version < vs.parse('0.1.1'):
        _dataset_to_0_1_1(metadata)
    if version < vs.parse('0.1.2'):
        _dataset_to_0_1_2(metadata)
    if version <= vs.parse('0.1.2') and 'Geometry Detail' not in metadata:
        metadata['Geometry Detail'] = {'Forced LOD level': 0}
    if version <= vs.parse('0.1.2') and 'Sequence Type' not in metadata:
        metadata['Sequence Type'] = 'Sequential'


def update_image_metadata(metadata):
    if 'Version' in metadata:
        version = vs.parse(metadata['Version'])
    else:
        version = vs.parse('0.0.0')

    if version < vs.parse('0.1.0'):
        _image_to_0_1_0(metadata)


def _dataset_to_0_1_1(metadata):
    metadata['Version'] = '0.1.1'
    if 'Dataset Metadata Version' in metadata:
        del metadata['Dataset Metadata Version']
    if 'World name' in metadata:
        metadata['World Name'] = metadata['World name']
        del metadata['World name']
    if 'Index Padding' not in metadata:
        metadata['Index Padding'] = 4


def _dataset_to_0_1_2(metadata):
    metadata['Version'] = '0.1.2'
    if 'World Information' not in metadata:
        metadata['World Information'] = {}
    if 'Image Filename Format' not in metadata:
        metadata['Image Filename Format'] = '{world}.{frame}'
    if 'Image Filename Format Mappings' not in metadata:
        metadata['Image Filename Format Mappings'] = {'world': metadata['World Name']}
    if 'Available Frame Metadata' not in metadata:
        metadata['Available Frame Metadata'] = ["Camera Location", "Camera Orientation"]
    if 'Path Generation' not in metadata:
        metadata['Path Generation'] = {'Generation Type': 'Manual'}


def _image_to_0_1_0(metadata):
    metadata['Version'] = '0.1.0'
    if 'W' not in metadata['Camera Orientation']:
        x, y, z, w = euler_to_quarternion(
            metadata['Camera Orientation']['X'],
            metadata['Camera Orientation']['Y'],
            metadata['Camera Orientation']['Z'])
        metadata['Camera Orientation'] = {'X': x, 'Y': y, 'Z': z, 'W': w}


def euler_to_quarternion(x, y, z):
    # I got this math from here:
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/
    # np array order is x y z w, which I'm trying to be consistent as the format for quarternions
    c1 = math.cos(y / 2)
    c2 = math.cos(z / 2)
    c3 = math.cos(x / 2)
    s1 = math.sin(y / 2)
    s2 = math.sin(z / 2)
    s3 = math.sin(x / 2)
    return (s1 * s2 * c3 + c1 * c2 * s3,
            s1 * c2 * c3 + c1 * s2 * s3,
            c1 * s2 * c3 - s1 * c2 * s3,
            c1 * c2 * c3 - s1 * s2 * s3)
