# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import arvet.database.tests.database_connection as dbconn
import arvet.core.tests.mock_types as mock_core
from arvet.batch_analysis.simple_experiment import SimpleExperiment


# ------------------------- HELPER TYPES -------------------------

class CountedMetricResult(mock_core.MockMetricResult):
    instances = 0

    def __init__(self, *args, **kwargs):
        super(CountedMetricResult, self).__init__(*args, **kwargs)
        self._inc_counter()

    @classmethod
    def _inc_counter(cls):
        cls.instances += 1


class PlottedMetricResult1(CountedMetricResult):

    @classmethod
    def get_available_plots(cls):
        return {'my_plot_1', 'my_plot_2'}

    # @classmethod
    # def visualize_results(cls, results, plots, display=True, output=''):
    #     print("show me the money")


class PlottedMetricResult2(CountedMetricResult):

    @classmethod
    def get_available_plots(cls):
        return {'plot_2', 'custom_demo_plot_newest'}


class MetricResultSubclass(PlottedMetricResult1):

    @classmethod
    def get_available_plots(cls):
        return PlottedMetricResult1.get_available_plots() | {'plot_new_newer', 'other_plot'}


# ------------------------- DATABASE TESTS -------------------------

@mock.patch.object(PlottedMetricResult1, 'visualize_results', autospec=True)
@mock.patch.object(PlottedMetricResult2, 'visualize_results', autospec=True)
@mock.patch.object(MetricResultSubclass, 'visualize_results', autospec=True)
class TestPlotResultsDatabase(unittest.TestCase):
    system = None
    image_source = None
    trial_result = None
    metric_result_1 = None
    metric_result_2 = None
    metric_result_3 = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

        # Build the heirarchy of references necessary to save a metric result
        cls.system = mock_core.MockSystem()
        cls.system.save()

        cls.image_source = mock_core.MockImageSource()
        cls.image_source.save()

        cls.metric = mock_core.MockMetric()
        cls.metric.save()

        cls.trial_result = mock_core.MockTrialResult(system=cls.system, image_source=cls.image_source, success=True)
        cls.trial_result.save()

        # Make some metric results that provide plots
        cls.metric_result_1 = PlottedMetricResult1(metric=cls.metric, trial_results=[cls.trial_result], success=True)
        cls.metric_result_1.save()

        cls.metric_result_2 = PlottedMetricResult2(metric=cls.metric, trial_results=[cls.trial_result], success=True)
        cls.metric_result_2.save()

        cls.metric_result_3 = MetricResultSubclass(metric=cls.metric, trial_results=[cls.trial_result], success=True)
        cls.metric_result_3.save()

        # Don't store the experiment to force the relevant references to load from the database.
        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            metric_results=[cls.metric_result_1, cls.metric_result_2, cls.metric_result_3]
        )
        obj.save()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        SimpleExperiment._mongometa.collection.drop()
        mock_core.MockMetric._mongometa.collection.drop()
        mock_core.MockTrialResult._mongometa.collection.drop()
        mock_core.MockSystem._mongometa.collection.drop()
        mock_core.MockImageSource._mongometa.collection.drop()
        mock_core.MockMetricResult._mongometa.collection.drop()

    def test_plot_results_calls_visualize_results_with_all_instances_of_that_class(self, *_):
        experiment = SimpleExperiment.objects.all().first()
        experiment.plot_results(
            PlottedMetricResult1.get_available_plots() |
            PlottedMetricResult2.get_available_plots() |
            MetricResultSubclass.get_available_plots()
        )

        self.assertEqual(1, PlottedMetricResult1.visualize_results.call_count)
        self.assertEqual(1, PlottedMetricResult2.visualize_results.call_count)
        self.assertEqual(1, MetricResultSubclass.visualize_results.call_count)

        results = PlottedMetricResult1.visualize_results.call_args[1]['results']
        plot_names = PlottedMetricResult1.visualize_results.call_args[1]['plots']
        self.assertEqual(2, len(results))
        keys = set()
        for result in results:
            self.assertIsInstance(result, PlottedMetricResult1)
            keys.add(result.pk)
        self.assertEqual(keys, {self.metric_result_1.pk, self.metric_result_3.pk})
        self.assertEqual(PlottedMetricResult1.get_available_plots(), set(plot_names))

        results = PlottedMetricResult2.visualize_results.call_args[1]['results']
        plot_names = PlottedMetricResult2.visualize_results.call_args[1]['plots']
        self.assertEqual(1, len(results))
        self.assertIsInstance(results[0], PlottedMetricResult2)
        self.assertEqual(results[0].pk, self.metric_result_2.pk)
        self.assertEqual(PlottedMetricResult2.get_available_plots(), set(plot_names))

        results = MetricResultSubclass.visualize_results.call_args[1]['results']
        plot_names = MetricResultSubclass.visualize_results.call_args[1]['plots']
        self.assertEqual(1, len(results))
        self.assertIsInstance(results[0], MetricResultSubclass)
        self.assertEqual(results[0].pk, self.metric_result_3.pk)
        self.assertEqual(MetricResultSubclass.get_available_plots(), set(plot_names))

    def test_only_passes_plot_names_applicable_to_that_model(self, *_):
        experiment = SimpleExperiment.objects.all().first()

        # Choose a subset of the plot names
        experiment.plot_results({'my_plot_2', 'custom_demo_plot_newest', 'other_plot', 'not_a_real_plot'})

        self.assertEqual(1, PlottedMetricResult1.visualize_results.call_count)
        self.assertEqual(1, PlottedMetricResult2.visualize_results.call_count)
        self.assertEqual(1, MetricResultSubclass.visualize_results.call_count)

        plot_names = PlottedMetricResult1.visualize_results.call_args[1]['plots']
        self.assertEqual({'my_plot_2'}, set(plot_names))

        plot_names = PlottedMetricResult2.visualize_results.call_args[1]['plots']
        self.assertEqual({'custom_demo_plot_newest'}, set(plot_names))

        plot_names = MetricResultSubclass.visualize_results.call_args[1]['plots']
        self.assertEqual({'my_plot_2', 'other_plot'}, set(plot_names))

    def test_passes_through_display_and_output_parameters(self, *_):
        experiment = SimpleExperiment.objects.all().first()
        for display in [True, False]:
            for output in {'/dev/null', 'mytestoutput'}:
                PlottedMetricResult1.visualize_results.reset_mock()
                PlottedMetricResult2.visualize_results.reset_mock()
                MetricResultSubclass.visualize_results.reset_mock()

                experiment.plot_results(
                    plot_names={'my_plot_2', 'custom_demo_plot_newest', 'other_plot', 'not_a_real_plot'},
                    display=display,
                    output=output
                )

                self.assertTrue(PlottedMetricResult1.visualize_results.called)
                self.assertTrue(PlottedMetricResult2.visualize_results.called)
                self.assertTrue(MetricResultSubclass.visualize_results.called)

                self.assertEqual(display, PlottedMetricResult1.visualize_results.call_args[1]['display'])
                self.assertEqual(output, PlottedMetricResult1.visualize_results.call_args[1]['output'])
                self.assertEqual(display, PlottedMetricResult2.visualize_results.call_args[1]['display'])
                self.assertEqual(output, PlottedMetricResult2.visualize_results.call_args[1]['output'])
                self.assertEqual(display, MetricResultSubclass.visualize_results.call_args[1]['display'])
                self.assertEqual(output, MetricResultSubclass.visualize_results.call_args[1]['output'])

    def test_doesnt_load_results_if_not_needed_for_plot(self, *_):
        experiment = SimpleExperiment.objects.all().first()

        PlottedMetricResult1.instances = 0
        PlottedMetricResult2.instances = 0
        MetricResultSubclass.instances = 0

        experiment.plot_results(PlottedMetricResult1.get_available_plots())

        self.assertEqual(1, PlottedMetricResult1.instances)
        self.assertEqual(0, PlottedMetricResult2.instances)
        # The one instance gets loaded twice, once for the parent, once for the subclass
        self.assertEqual(2, MetricResultSubclass.instances)
