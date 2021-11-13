<?php
ini_set("display_errors","off");
include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;

/* Reports */
$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

echo 'Loading data into memory ...' . PHP_EOL;

/* Train with the dataset */
$training = Labeled::fromIterator(new CSV(dirname(__FILE__).'/datasets/uoc_train_balanced.csv', true));
$training->apply(new NumericStringConverter())->apply(new MinMaxNormalizer());

$training_predictions = Labeled::fromIterator(new CSV(dirname(__FILE__).'/datasets/uoc_train.csv', true));
$training_predictions->apply(new NumericStringConverter())->apply(new MinMaxNormalizer());
$testing = $training_predictions->randomize()->take(100);

$estimator = new RandomForest(new ClassificationTree(20), 300, 0.3, true);

echo 'Training ...' . PHP_EOL;

$estimator->setBackend(new Amp(10));
$estimator->train($training);

echo 'Making predictions ...' . PHP_EOL;

$predictions = $estimator->predict($testing);

/* Make predictions for uoc_X_test.csv */
echo 'Example predictions:' . PHP_EOL;

$estimations = $estimator->predict(Unlabeled::fromIterator(new CSV(dirname(__FILE__).'/datasets/uoc_X_test.csv', true)));
print_r(array_slice($estimations, 0, 5));

/* Estimate the score */
$metric = new Accuracy();
$score = $metric->score($predictions, $testing->labels());

echo 'Accuracy is ' . (string) ($score * 100.0) . '% '.PHP_EOL;

$results = $report->generate($predictions, $testing->labels());
echo 'f1-score is ' . $results[0]["overall"]["f1_score"] .'%'. PHP_EOL;