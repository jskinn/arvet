import os.path
import pymodm
import numpy as np
from arvet.util.test_helpers import ExtendedTestCase
import arvet.database.tests.database_connection as dbconn
from arvet.database.image_field import ImageField, ImageManager
import arvet.database.image_manager as im_manager


class TestImageFieldModel(pymodm.MongoModel):
    image = ImageField()
    objects = ImageManager()


class TestImageFieldEmbeddedModel(pymodm.EmbeddedMongoModel):
    image = ImageField()


class TestImageInEmbeddedDocumentModel(pymodm.MongoModel):
    inner = pymodm.fields.EmbeddedDocumentField(TestImageFieldEmbeddedModel)
    objects = ImageManager()


class TestImageInEmbeddedDocumentListModel(pymodm.MongoModel):
    inner_list = pymodm.fields.EmbeddedDocumentListField(TestImageFieldEmbeddedModel)
    objects = ImageManager()


class TestNestedEmbeddedDoc(pymodm.EmbeddedMongoModel):
    inner = pymodm.fields.EmbeddedDocumentField(TestImageFieldEmbeddedModel)


class TestComplexImageModel(pymodm.MongoModel):
    image = ImageField()
    inner = pymodm.fields.EmbeddedDocumentField(TestImageFieldEmbeddedModel)
    nested_inner = pymodm.fields.EmbeddedDocumentField(TestNestedEmbeddedDoc)
    inner_list = pymodm.fields.EmbeddedDocumentListField(TestImageFieldEmbeddedModel)
    nested_inner_list = pymodm.fields.EmbeddedDocumentListField(TestNestedEmbeddedDoc)
    objects = ImageManager()


class TestImageField(ExtendedTestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model and removing the image file
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)
        TestImageFieldModel._mongometa.collection.drop()
        TestImageInEmbeddedDocumentModel._mongometa.collection.drop()
        TestImageInEmbeddedDocumentListModel._mongometa.collection.drop()
        TestComplexImageModel._mongometa.collection.drop()

    def test_image_field_stores_and_loads(self):
        image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        # Save the image
        model = TestImageFieldModel()
        model.image = image
        model.save()

        # Load all the entities
        all_entities = list(TestImageFieldModel.objects.all())
        self.assertEqual(len(all_entities), 1)
        self.assertNPEqual(all_entities[0].image, image)
        all_entities[0].delete()

    def test_group_modifies_storage_path(self):
        image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        # Define a custom model class
        class GroupedImageModel(pymodm.MongoModel):
            image = ImageField(group='mygroup')

        model = GroupedImageModel(image=image)
        model.save()

        # Get the unserialised documents
        try:
            documents = list(GroupedImageModel.objects.all().values())
            self.assertEqual(len(documents), 1)
            self.assertTrue(documents[0]['image'].startswith('mygroup/'))
        finally:
            GroupedImageModel._mongometa.collection.drop()

    def test_delete_removes(self):
        image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)
        path = image_manager.find_path_for_image(image)

        # Save the image
        model = TestImageFieldModel()
        model.image = image
        model.save()
        self.assertTrue(image_manager.is_valid_path(path))

        # Delete the model
        model.delete()
        self.assertFalse(image_manager.is_valid_path(path))

    def test_delete_removes_from_embedded_model(self):
        image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)
        path = image_manager.find_path_for_image(image)

        # Save the image
        model = TestImageInEmbeddedDocumentModel()
        model.inner = TestImageFieldEmbeddedModel()
        model.inner.image = image
        model.save()
        self.assertTrue(image_manager.is_valid_path(path))

        # Delete the model
        model.delete()
        self.assertFalse(image_manager.is_valid_path(path))

    def test_delete_removes_from_embedded_model_list(self):
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        image1 = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        path1 = image_manager.find_path_for_image(image1)
        path2 = image_manager.find_path_for_image(image2)

        # Build the model
        inner1 = TestImageFieldEmbeddedModel(image=image1)
        inner2 = TestImageFieldEmbeddedModel(image=image2)
        model = TestImageInEmbeddedDocumentListModel(inner_list=[inner1, inner2])
        model.save()
        self.assertTrue(image_manager.is_valid_path(path1))
        self.assertTrue(image_manager.is_valid_path(path2))

        # Delete the model
        model.delete()
        self.assertFalse(image_manager.is_valid_path(path1))
        self.assertFalse(image_manager.is_valid_path(path2))

    def test_delete_removes_from_complex_model(self):
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        images = [
            np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
            for _ in range(7)
        ]
        paths = [
            image_manager.find_path_for_image(image)
            for image in images
        ]

        # Build the model
        inners = [
            TestImageFieldEmbeddedModel(image=images[idx])
            for idx in range(6)
        ]

        nested1 = TestNestedEmbeddedDoc(inner=inners[0])
        nested2 = TestNestedEmbeddedDoc(inner=inners[1])
        nested3 = TestNestedEmbeddedDoc(inner=inners[2])

        model = TestComplexImageModel(
            image=images[6],
            inner=inners[3],
            nested_inner=nested1,
            inner_list=[inners[4], inners[5]],
            nested_inner_list=[nested2, nested3]
        )
        model.save()
        for path in paths:
            self.assertTrue(image_manager.is_valid_path(path))

        # Delete the model
        model.delete()
        for path in paths:
            self.assertFalse(image_manager.is_valid_path(path))

    def test_delete_removes_from_lots_of_complex_models(self):
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        ids_to_delete = []
        kept_paths = []
        deleted_paths = []
        for idx in range(10):
            # Make the images
            images = [
                np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
                for _ in range(7)
            ]
            if idx % 2 == 0:
                deleted_paths.extend(
                    image_manager.find_path_for_image(image)
                    for image in images
                )
            else:
                kept_paths.extend(
                    image_manager.find_path_for_image(image)
                    for image in images
                )

            # Build the inner models
            inners = [
                TestImageFieldEmbeddedModel(image=images[idx])
                for idx in range(6)
            ]

            nested1 = TestNestedEmbeddedDoc(inner=inners[0])
            nested2 = TestNestedEmbeddedDoc(inner=inners[1])
            nested3 = TestNestedEmbeddedDoc(inner=inners[2])

            # Make a complex model
            model = TestComplexImageModel(
                image=images[6],
                inner=inners[3],
                nested_inner=nested1,
                inner_list=[inners[4], inners[5]],
                nested_inner_list=[nested2, nested3]
            )
            model.save()
            if idx % 2 == 0:
                ids_to_delete.append(model.pk)

        for path in kept_paths:
            self.assertTrue(image_manager.is_valid_path(path))
        for path in deleted_paths:
            self.assertTrue(image_manager.is_valid_path(path))

        # Delete some of the models
        TestComplexImageModel.objects.raw({'_id': {'$in': ids_to_delete}}).delete()
        for path in kept_paths:
            self.assertTrue(image_manager.is_valid_path(path))
        for path in deleted_paths:
            self.assertFalse(image_manager.is_valid_path(path))
