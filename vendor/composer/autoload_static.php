<?php

// autoload_static.php @generated by Composer

namespace Composer\Autoload;

class ComposerStaticInit6d62affc2c3dfcd096df482a22d3f443
{
    public static $files = array (
        'e8aa6e4b5a1db2f56ae794f1505391a8' => __DIR__ . '/..' . '/amphp/amp/lib/functions.php',
        '76cd0796156622033397994f25b0d8fc' => __DIR__ . '/..' . '/amphp/amp/lib/Internal/functions.php',
        '6cd5651c4fef5ed6b63e8d8b8ffbf3cc' => __DIR__ . '/..' . '/amphp/byte-stream/lib/functions.php',
        '8dc56fe697ca93c4b40d876df1c94584' => __DIR__ . '/..' . '/amphp/process/lib/functions.php',
        '3da389f428d8ee50333e4391c3f45046' => __DIR__ . '/..' . '/amphp/serialization/src/functions.php',
        'bcb7d4fc55f4b1a7e10f5806723e9892' => __DIR__ . '/..' . '/amphp/sync/src/functions.php',
        'e187e371b30897d6dc51cac6a8c94ff6' => __DIR__ . '/..' . '/amphp/sync/src/ConcurrentIterator/functions.php',
        '430de19db8b7ee88fdbe5c545d82d33d' => __DIR__ . '/..' . '/amphp/parallel/lib/Context/functions.php',
        '888e1afeed2e8d13ef5a662692091e6e' => __DIR__ . '/..' . '/amphp/parallel/lib/Sync/functions.php',
        '384cf4f2eb4d2f896db72315a76066ad' => __DIR__ . '/..' . '/amphp/parallel/lib/Worker/functions.php',
        '8f758069bf9eb3411d096c10be343745' => __DIR__ . '/..' . '/rubix/tensor/src/constants.php',
        '0e6d7bf4a5811bfa5cf40c5ccd6fae6a' => __DIR__ . '/..' . '/symfony/polyfill-mbstring/bootstrap.php',
        '0d59ee240a4cd96ddbb4ff164fccea4d' => __DIR__ . '/..' . '/symfony/polyfill-php73/bootstrap.php',
        'a4a119a56e50fbb293281d9a48007e0e' => __DIR__ . '/..' . '/symfony/polyfill-php80/bootstrap.php',
        '702239352e6628be5dc71b6fd029e72e' => __DIR__ . '/..' . '/rubix/ml/src/constants.php',
        '0315e8fd3e479309d097647b8ef2920b' => __DIR__ . '/..' . '/rubix/ml/src/functions.php',
    );

    public static $prefixLengthsPsr4 = array (
        'Z' => 
        array (
            'Zephir\\Optimizers\\FunctionCall\\' => 31,
        ),
        'T' => 
        array (
            'Tensor\\' => 7,
        ),
        'S' => 
        array (
            'Symfony\\Polyfill\\Php80\\' => 23,
            'Symfony\\Polyfill\\Php73\\' => 23,
            'Symfony\\Polyfill\\Mbstring\\' => 26,
        ),
        'R' => 
        array (
            'Rubix\\ML\\' => 9,
        ),
        'P' => 
        array (
            'Psr\\Log\\' => 8,
        ),
        'L' => 
        array (
            'League\\MimeTypeDetection\\' => 25,
            'League\\Flysystem\\' => 17,
        ),
        'J' => 
        array (
            'JAMA\\' => 5,
        ),
        'A' => 
        array (
            'Amp\\Sync\\' => 9,
            'Amp\\Serialization\\' => 18,
            'Amp\\Process\\' => 12,
            'Amp\\Parser\\' => 11,
            'Amp\\Parallel\\' => 13,
            'Amp\\ByteStream\\' => 15,
            'Amp\\' => 4,
        ),
    );

    public static $prefixDirsPsr4 = array (
        'Zephir\\Optimizers\\FunctionCall\\' => 
        array (
            0 => __DIR__ . '/..' . '/rubix/tensor/optimizers',
        ),
        'Tensor\\' => 
        array (
            0 => __DIR__ . '/..' . '/rubix/tensor/src',
        ),
        'Symfony\\Polyfill\\Php80\\' => 
        array (
            0 => __DIR__ . '/..' . '/symfony/polyfill-php80',
        ),
        'Symfony\\Polyfill\\Php73\\' => 
        array (
            0 => __DIR__ . '/..' . '/symfony/polyfill-php73',
        ),
        'Symfony\\Polyfill\\Mbstring\\' => 
        array (
            0 => __DIR__ . '/..' . '/symfony/polyfill-mbstring',
        ),
        'Rubix\\ML\\' => 
        array (
            0 => __DIR__ . '/..' . '/rubix/ml/src',
        ),
        'Psr\\Log\\' => 
        array (
            0 => __DIR__ . '/..' . '/psr/log/Psr/Log',
        ),
        'League\\MimeTypeDetection\\' => 
        array (
            0 => __DIR__ . '/..' . '/league/mime-type-detection/src',
        ),
        'League\\Flysystem\\' => 
        array (
            0 => __DIR__ . '/..' . '/league/flysystem/src',
        ),
        'JAMA\\' => 
        array (
            0 => __DIR__ . '/..' . '/rubix/tensor/lib/JAMA',
        ),
        'Amp\\Sync\\' => 
        array (
            0 => __DIR__ . '/..' . '/amphp/sync/src',
        ),
        'Amp\\Serialization\\' => 
        array (
            0 => __DIR__ . '/..' . '/amphp/serialization/src',
        ),
        'Amp\\Process\\' => 
        array (
            0 => __DIR__ . '/..' . '/amphp/process/lib',
        ),
        'Amp\\Parser\\' => 
        array (
            0 => __DIR__ . '/..' . '/amphp/parser/lib',
        ),
        'Amp\\Parallel\\' => 
        array (
            0 => __DIR__ . '/..' . '/amphp/parallel/lib',
        ),
        'Amp\\ByteStream\\' => 
        array (
            0 => __DIR__ . '/..' . '/amphp/byte-stream/lib',
        ),
        'Amp\\' => 
        array (
            0 => __DIR__ . '/..' . '/amphp/amp/lib',
        ),
    );

    public static $classMap = array (
        'Attribute' => __DIR__ . '/..' . '/symfony/polyfill-php80/Resources/stubs/Attribute.php',
        'JsonException' => __DIR__ . '/..' . '/symfony/polyfill-php73/Resources/stubs/JsonException.php',
        'Stringable' => __DIR__ . '/..' . '/symfony/polyfill-php80/Resources/stubs/Stringable.php',
        'UnhandledMatchError' => __DIR__ . '/..' . '/symfony/polyfill-php80/Resources/stubs/UnhandledMatchError.php',
        'ValueError' => __DIR__ . '/..' . '/symfony/polyfill-php80/Resources/stubs/ValueError.php',
    );

    public static function getInitializer(ClassLoader $loader)
    {
        return \Closure::bind(function () use ($loader) {
            $loader->prefixLengthsPsr4 = ComposerStaticInit6d62affc2c3dfcd096df482a22d3f443::$prefixLengthsPsr4;
            $loader->prefixDirsPsr4 = ComposerStaticInit6d62affc2c3dfcd096df482a22d3f443::$prefixDirsPsr4;
            $loader->classMap = ComposerStaticInit6d62affc2c3dfcd096df482a22d3f443::$classMap;

        }, null, ClassLoader::class);
    }
}